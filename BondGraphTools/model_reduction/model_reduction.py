from collections import namedtuple
import logging


from sympy.core import Lambda
from sympy.solvers.solveset import invert_real
logger = logging.getLogger(__name__)

from BondGraphTools.model_reduction.algebra import *
from BondGraphTools.model_reduction.symbols import *
from BondGraphTools.exceptions import SymbolicException

DAE = namedtuple("DAE", ["X", "P", "L", "M", "J"])


def as_dict(sparse_matrix):
    """Converts an instance of `sympy.SparseMatrix` into a `dict`-of-`dict`s.

    Args:
        sparse_matrix: The input matrix

    Returns:
        The matrix in dictionary form.
    """
    output = {}
    n, m = sparse_matrix.shape
    for i in range(n):
        row = {
            j: sparse_matrix[i, j] for j in range(m)
            if sparse_matrix[i, j] != 0
        }
        if row:
            output.update({i:row})
    return output


def _sympy_to_dict(equation, coordinates):

    L = {}
    M = {}
    J = []

    remainder = equation
    partials = [remainder.diff(x) for x in coordinates]
    for i, r_i in enumerate(partials):
        if not (r_i.atoms() & set(coordinates)) and not r_i.is_zero:
            L[i] = r_i
            remainder -= r_i * coordinates[i]

    remainder = remainder.expand()

    if remainder.is_Add:
        terms = remainder.args
    elif remainder.is_zero:
        terms = []
    else:
        terms = [remainder]
    logger.info("Nonlinear terms %s are %s", type(terms), terms)
    for term in terms:
        coeff = sympy.Number("1")
        nonlinearity = sympy.Number("1")
        logger.info("Checking factors %s\n", term.as_coeff_mul())

        for factor in flatten(term.as_coeff_mul()):
            if factor.atoms() & set(coordinates):
                nonlinearity = factor * nonlinearity
            else:
                coeff = factor * coeff
        logger.info("Coefficients: %s of nonlinearity: %s", coeff, nonlinearity)
        try:
            index = J.index(nonlinearity)
        except ValueError:
            index = len(J)
            J.append(nonlinearity)

        M[index] = coeff

    return L, M, J


def parse_relation(
        equation: str,
        coordinates: list,
        parameters: set = None,
        substitutions: set = None) -> tuple:
    """

    Args:
        equation: The equation in string format
        coordinates: a list of symbolic variables for the coordinate system
        parameters: a set of symbolic varibales that should be treated as
                    non-zero parameters.
        substitutions: A set tuples (p, v) where p is a symbolic variable and
                       v it's value

    Returns:
        tuple (L, M, J) such that $LX + MJ(X) =0$

    Parses the input string into canonical implicit form.
    - $L$ is a sparse row vector (in dict form) of the same length as the
    co-oridinates (dict form)
    - $M$ is a sparse row vector that is the same size as $J$ (dict form)
    containing the coefficients of each unique nonlinear term.
    - $J$ is a column vector of of unique nonlinear terms.
    """

    namespace = {str(x): x for x in coordinates}
    logger.info("Got coords: %s", [(c, c.__class__) for c in coordinates])
    if parameters:
        namespace.update({str(x): x for x in parameters})
    try:
        p, q = equation.split("=")
        relation = f"({p}) -({q})"
    except (ValueError, AttributeError):
        relation = equation

    logger.info(f"Trying to sympify \'{relation}\' with locals={namespace}")

    remainder = sympy.sympify(relation, locals=namespace).expand()

    logger.info(f"Got {remainder}")

    if substitutions:
        remainder = remainder.subs(substitutions)

    unknowns = []
    for a in remainder.atoms():
        if a in coordinates:
            continue
        if a.is_number:
            continue
        if parameters and str(a) in {str(p) for p in parameters}:
            continue

        # TODO: hack to get around weird behaviour with sympy
        if a.name in namespace:
            remainder = remainder.subs(a, namespace[a.name])
            continue

        logger.info(f"Don't know what to do with {a} of type f{a.__class__} ")
        unknowns.append(a)

    if unknowns:
        raise SymbolicException(f"While parsing {relation} found unknown " 
                                f"terms {unknowns} in namespace {namespace}")

    return _sympy_to_dict(remainder, coordinates)


def _is_number(value):
    """
    Returns: True if the value is a number or a number-like vaiable
    """
    if isinstance(value, (float, complex, int)):
        return True
    try:
        return value.is_number
    except AttributeError:
        pass
    return False


def _make_coords(model):

    state = [Variable(x) for x in model.state_vars]
    derivatives = [DVariable(x) for x in state]

    inputs = [Control(u) for u in model.control_vars]
    outputs = [Output(y) for y in model.output_vars]

    ports = []
    for p in model.ports:

        ports.append(Effort(f"e_{p.index}"))
        ports.append(Flow(f"f_{p.index}"))

    params = set()
    substitutions = set()

    for param in model.params:
        value = model.params[param]

        if (not value or param in model.control_vars
                or param in model.output_vars):
            continue
        elif isinstance(value, dict):
            try:
                value = value['value']
            except AttributeError:
                continue

        if isinstance(value, Parameter):
            params.add(value)
        elif _is_number(value):
            substitutions.add((sympy.Symbol(param), value))
        elif isinstance(value, sympy.Expr):
            pass
        elif isinstance(value, str):
            pass
        else:
            raise NotImplementedError(
                f"Don't know how to treat {model.uri}.{param} "
                f"with Value {value}"
            )

    return outputs + derivatives + ports + state + inputs, params, substitutions


def _generate_atomics_system(model):
    """
    Args:
          model: Instance of `BondGraphBase` from which to generate matrix
                 equation.

    Returns:
        tuple $(coordinates, parameters, L, M, J)$

    Such that $L_pX + M_p*J(X) = 0$.

    """
    # coordinates is list
    # parameters is a set

    coordinates, parameters, substitutions = _make_coords(model)

    relations = model.constitutive_relations

    # Matrix for linear part.
    L = sympy.SparseMatrix(len(relations), len(coordinates), {})

    # Matrix for nonlinear part {row:  {column: value }}
    M = sympy.SparseMatrix(len(relations), 0, {})
    J = [] # nonlinear terms

    for i, relation in enumerate(model.constitutive_relations):
        L_1, M_1, J_1 = parse_relation(relation, coordinates,
                                       parameters, substitutions)
        for j, v in L_1.items():
            L[i, j] = v

        mappings = {}

        for m_i, term in enumerate(J_1):
            try:
                index = J.index(term)
                mappings[m_i] = index
            except ValueError:
                index = len(J)
                J.append(term)
                mappings[m_i] = index
        if len(J) > M.cols:
            M = M.row_join(
                sympy.SparseMatrix(len(relations), len(J) - M.cols, {})
            )

        logger.info("Nonlinear Matrix M: %s", str(M))
        logger.info("Nonlinear Terms J: %s", str(J))
        for col, value in M_1.items():
            M[i, mappings[col]] = value

    return coordinates, parameters, L, M, J


def merge_coordinates(*pairs):
    """Merges coordinate spaces and parameter spaces together

    This function takes a list of coordinates and parameters and builds a new
    coordinate space by simply taking the direct of the relavent spaces and
    returns the result along with a list of inverse maps (dictionaries)
    identifying how to get get back

    For example::

        c_pair = [dx_0, e_0, f_0, x_0], [C]
        r_pair = [e_0, f_0], [R]

        new_pair, maps = merge_coordinates(c_pair, r_pair)

    would return a new coordinate system::

        new_pair == [dx_0, e_0, f_0, e_1, f_1, x_0], ['C','R']

    with maps::

        maps == ({0:0, 1:1, 2:2, 5:3}, {0:0}), ({3:0, 4:1}, {1:0})

    which identifies how the index of the new coordinate system (the keys)
    relate to the index of the old coordinate system (the values)
    for both the state space (first of the pair) and the parameter space
    (second of the pair).

    Args:
        *pairs: iterable of state space and parameter space pairs.

    Returns:
        tuple, list of tuples.

    """

    new_coordinates = []
    counters = {
        DVariable: 0,
        Variable: 0,
        Effort: 0,
        Flow: 0,
        Output: 0,
        Control: 0,
    }

    new_parameters = set()

    x_projectors = {}
    logger.info("Merging coordinates..")
    for index, (coords, params) in enumerate(pairs):
        x_inverse = {}

        logger.info(
            "Coordinates: %s, Params %s:", coords, params
        )
        # Parameters can be shared; needs to be many-to-one
        # So we need to check if they're in the parameter set before adding
        # them
        for old_p_index, param in enumerate(params):
            new_parameters.add(param)

        for idx, x in enumerate(coords):

            new_idx = len(new_coordinates)
            x_inverse.update({new_idx: idx})

            cls = x.__class__
            new_x = cls(f"{cls.default_prefix}_{counters[cls]}")

            counters[cls] += 1
            new_coordinates.append(new_x)

        x_projectors[index] = x_inverse

    new_coordinates, permuation_map = permutation(
        new_coordinates, canonical_order
    )
    # the permutation map that $x_i -> x_j$ then (i,j) in p_map^T
    permuation = {i: j for i, j in permuation_map}

    for index in x_projectors:
        x_projectors[index] = {
            permuation[i]: j for i, j in x_projectors[index].items()
        }

    projectors = [x_projectors[i] for i in x_projectors]
    return (new_coordinates, new_parameters), projectors


def merge_systems(*systems):
    """
    Args:
        systems: An order lists of system to merge

    Returns:
        A new system, and an inverse mapping.

    See Also:
        _generate_atomics_system

    Merges a set of systems together. Each system should be of the form
    `X,P,L,M,J` where
    - `X` is a `list` of local cordinates
    - `P` is a `set` of local parameters
    - `L` is a Sparse Matrix
    - `M` is a Sparse Matrix of the nonlinear contribution weighting.
    - 'J' is nonlinear atomic terms.

    The resulting merged system is of the same form.

    """

    logger.info("Merging systems")

    coord_list, param_list, L_list, M_list, nonlinear_terms = zip(*systems)

    # create block diagrams
    L = sparse_block_diag(L_list)
    M = sparse_block_diag(M_list)

    (coords, params), maps = merge_coordinates(*zip(coord_list, param_list))
    P = sympy.SparseMatrix(len(coords), len(coords),{})

    logging.info("New coordinates: %s", str(coords))
    logging.info("Mappings: %s", str(maps))
    J = []
    offset = 0

    for new_to_old,  old_coords, nlin in zip(maps, coord_list, nonlinear_terms):

        # New_to_Old is a dictionary of ints i:j such that
        # newx_i = oldx_j + offset
        #
        # Hence, we need to swap the columns of L to reflect this change of
        # coordinates. We also need to substitute the nonlinear terms.
        # but we can't just to a straight swap because they might share the
        # so we pass to intermediates first

        if nlin:
            logger.info("Substituting nonlinear terms: %s", J)
            intermediates = {
                x: sympy.Dummy(f"i_{i}") for i, x in enumerate(old_coords)
            }
            targets = list(intermediates.values())
            substitutions = [
                (targets[j], coords[i]) for i, j in new_to_old.items()
            ]
            assert substitutions
            for term in nlin:
                temp = term.subs(intermediates.items())
                J.append(temp.subs(substitutions))

        for i, j in new_to_old.items():
            P[i, j + offset] = 1

        offset += len(old_coords)

    L = L * P.T
    return coords, params, L, M, J, maps


def generate_system_from(model):
    """Generates an implicit dynamical system from an instance of
    `BondGraphBase`.

    Args:
        model:

    Returns:

    """
    try:
        systems = {
            component: generate_system_from(component)
            for component in model.components
        }
    except AttributeError:
        return _generate_atomics_system(model)

    X, P, L, M, J, maps = merge_systems(*systems.values())

    map_dictionary = {c: M for c, M in zip(systems.keys(), maps)}

    L_bonds = sympy.SparseMatrix(2*len(model.bonds), L.cols, {})

    # Add the bonds:
    for row, (head_port, tail_port) in enumerate(model.bonds):
        # 1. Get the respective systems
        X_head = systems[head_port.component][0]
        head_to_local_map = {
            j: i for i, j in map_dictionary[head_port.component].items()
        }

        X_tail = systems[tail_port.component][0]

        tail_to_local_map = {
            j: i for i, j in map_dictionary[tail_port.component].items()
        }
        # 2. Find the respetive pairs of coorindates.
        e_1, = [tail_to_local_map[i] for i, x in enumerate(X_tail)
                if x.index == tail_port.index and isinstance(x, Effort)]
        f_1, = [tail_to_local_map[i] for i, x in enumerate(X_tail)
                if x.index == tail_port.index and isinstance(x, Flow)]
        e_2, = [head_to_local_map[i] for i, x in enumerate(X_head)
                if x.index == head_port.index and isinstance(x, Effort)]
        f_2, = [head_to_local_map[i] for i, x in enumerate(X_head)
                if x.index == head_port.index and isinstance(x, Flow)]

        # 2. add as a row in the linear matrix.
        L_bonds[2 * row, e_1] = 1
        L_bonds[2 * row, e_2] = -1
        L_bonds[2 * row + 1, f_1] = 1
        L_bonds[2 * row + 1, f_2] = 1

    L = L.col_join(L_bonds)

    return X, P, L, M, J


def sparse_eye(n):
    """Returns a sparse representation of the identity matrix"""
    return sympy.SparseMatrix(n, n, {(i, i): 1 for i in range(n)})


class InversionError(SymbolicException):
    pass


def invert(eqn, var):

    soln = sympy.solve(eqn, var, dict=True)
    if var in soln and len(soln) == 1:
        return soln[var]
    else:
        return None


def _get_next_eq(rows, system):

    X, P, L, M, J = system

    for row, atoms in rows:

        nonlinearity = sum(M[row, c] * J[c] for c in M.cols)
        for atom in atoms:
            # try and invert it with respect to the target variable
            eqn = invert(L[row, :] * X + nonlinearity, atom)
            if eqn:
                return row, atom, eqn

    raise InversionError


def _merge_in(system, row, eqn):
    X, P, L, M, J = system

    Lp, Mp, Jp = _sympy_to_dict(eqn, X)

    atoms = set()
    for col, val in Lp:
        L[row, col] = val

    for idx, term in enumerate(Jp):
        atoms |= term.atoms()
        col = J.index(term)
        if col < 0:
            col = len(J)
            M = M.row_join(sympy.SparseMatrix(M.rows, 1, {}))
            J.append(term)

        try:
            M[row, col] = Mp[col]
        except AttributeError:
            pass

    return row, atoms


def _make_ef_invertible(system):
    """Assumes Linear part is in Smith Normal Form"""

    X, P, L, M, J = system

    targets = {
        x for i, x in enumerate(X)
        if L[i, i] == 0 and isinstance(x, (Effort, Flow))
    }

    rows = []
    for row in (i for i, x in enumerate(X)
                if not isinstance(x, (DVariable, Output))):

        nonlinearity = sum(M[row, j]*J[j] for j in M.cols)
        rows.append(
            (row, nonlinearity.atoms() & targets)
        )

    while targets and rows:
        # find the row with the smallest number of target variables.
        rows.sort(key=lambda _, atoms: len(atoms))

        # if we can, remove it from the target list and substitute through

        try:
            row, atom, eqn = _get_next_eq(rows, system)
        except InversionError:
            # whatever is left are algebraic
            raise NotImplementedError

        J = [term.subs(atom, eqn) for term in J]

        r, a = _merge_in(system, row, atom - eqn)

        targets.remove(atom)

        rows = [
            (row, atoms & targets) for (row, atoms) in rows
            if atoms & targets and row != r
        ]
        rows.append((r, a))

    return X, P, L, M, J


def reduce(system):
    """ Performs basic symbolic reduction on the given system.

    We assume that the system coordinates are stack so that::

        X = (y, dx, e, f, x, u)

    Args:
        system:

    Returns: system (tuple)
    """
    X, P, L, M, J = system

    J_vect = sympy.Matrix(len(J), 1, J)

    lin_matrix, nlin_matrix = smith_normal_form(L, M)

    _make_ef_invertible((X, P, lin_matrix, nlin_matrix, J))





    # if any(isinstance(x, DVariable) for term in J for x in term.atoms()):
    #     # System is Nonholonomic
    #     raise NotImplementedError("System is non-holonomic")
    # elif any(lin_matrix[i, i] == 0
    #          for i, x in enumerate(X) if isinstance(x, DVariable)):
    #     raise NotImplementedError("System has conserved quantities")
    # else:
    #     return
