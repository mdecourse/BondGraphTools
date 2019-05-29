from collections import namedtuple
import logging
import sympy

from BondGraphTools.model_reduction.algebra import *
from BondGraphTools.model_reduction.symbols import *
from BondGraphTools.exceptions import SymbolicException

logger = logging.getLogger(__name__)

DAE = namedtuple("DAE", ["X", "P", "L", "M", "J"])


class System(object):
    __slots__ = ["X", "P", "L", "M", "J"]

    def __init__(self,
                 coordinates,
                 parameters,
                 linear_matrix,
                 nonlinear_matrix,
                 nonlinear_terms):
        self.X = coordinates
        self.P = parameters
        self.L = linear_matrix
        self.M = nonlinear_matrix
        self.J = nonlinear_terms

    def __iter__(self):
        for t in (self.X, self.P, self.L, self.M, self.J):
            yield t


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
            output.update({i: row})
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
        parameters: list,
        namespace: dict) -> tuple:
    """

    Args:
        equation: The equation in string format
        coordinates: a list of symbolic variables for the coordinate system
        parameters: a set of symbolic varibales that should be treated as
                    non-zero parameters.
        namespace: A dict tuples (p, var): value where p is a string,
                       var is a Dummy variable, and v

    Returns:
        tuple (L, M, J) such that $LX + MJ(X) =0$

    Parses the input string into canonical implicit form.
    - $L$ is a sparse row vector (in dict form) of the same length as the
    co-oridinates (dict form)
    - $M$ is a sparse row vector that is the same size as $J$ (dict form)
    containing the coefficients of each unique nonlinear term.
    - $J$ is a column vector of of unique nonlinear terms.
    """

    try:
        p, q = equation.split("=")
        relation = f"({p}) -({q})"
    except (ValueError, AttributeError):
        relation = equation

    logger.info(f"Trying to sympify \'{relation}\' with locals={namespace}")

    remainder = sympy.sympify(relation, namespace).expand()

    logger.info(f"Got {remainder}")

    unknowns = []
    for a in remainder.atoms():
        if a in coordinates or a in parameters or a.is_number:
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
    """
    Generates the coordinates for the model:
    [y, dx, e, f ,x, u]

    the parameters:
    [k_1, k_2, ...]


    and the mapping between symbols and values
    {
        local_value: k_i
    }
    which are immediately substituted.


    For parameters, we have 3 cases::

        1. Parameter Values are specified numerically
           Here, we interpret this as a 'universal constant' and simply
           discard the name
        2. Parameter Values are an existing symbol or symbolic expression.
           An instance of `Parameter` is created for each unique symbol.
           If values are not specified, they are given a local name.
        3. Parameter Value is an instance of `Parameter`.
           If it does not exist in the
    """

    def _update_parameters(parameters, atom):
        if atom in parameters:
            return atom
        if isinstance(atom, Parameter):
            parameters.append(atom)
            return atom
        else:
            p = Parameter(name=str(atom), value=atom)
            parameters.append(p)
            return p

    coordinates = []
    params = []
    namespace = dict()

    state = []
    for x in model.state_vars:
        v = Variable(x)
        namespace[x] = v
        state.append(v)

    for y in model.output_vars:
        v = Output(y)
        namespace[y] = v
        coordinates.append(v)

    for x in state:
        v = DVariable(x)
        namespace[str(v)] = v
        coordinates.append(v)

    for p in model.ports:
        e_str = f"e_{p.index}"
        f_str = f"f_{p.index}"
        e = Effort(e_str)
        f = Flow(f_str)
        namespace.update({e_str: e, f_str: f})
        coordinates += [e, f]

    coordinates += state

    for u in model.control_vars:
        v = Control(u)
        namespace[u] = v
        coordinates.append(v)

    for param in (p for p in model.params if
                  p not in model.control_vars and
                  p not in model.output_vars):

        assert isinstance(param, str), \
            f"{model.uri}.{param} is must be a string"

        value = model.params[param]
        if isinstance(value, dict) and 'value' in value:
            value = value['value']

        # Parameter values are given as a number.
        if isinstance(value, (float, int)):
            value = sympy.Number(value)
        # Parameters are symbolic
        elif isinstance(value, (sympy.Symbol, Parameter)):
            value = _update_parameters(params, value)
        # Parameters are equations
        elif isinstance(value, sympy.Expr):
            for atom in {v for v in value.atoms()}:
                if atom.is_number and not isinstance(atom, Parameter):
                    continue
                v = _update_parameters(params, atom)
                value.subs(atom, v)
        else:
            raise NotImplementedError(
                f"Unknown parameter value {model.uri}.{param} = {value}"
            )
        namespace.update({param: value})

    return coordinates, params, namespace


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


    relations = model.equations
    # parameters is a set

    coordinates, parameters, namespace = _make_coords(model)

    # Matrix for linear part.
    L = sympy.SparseMatrix(len(relations), len(coordinates), {})

    # Matrix for nonlinear part {row:  {column: value }}
    M = sympy.SparseMatrix(len(relations), 0, {})
    J = []  # nonlinear terms

    for i, relation in enumerate(relations):
        L_1, M_1, J_1 = parse_relation(relation, coordinates,
                                       parameters, namespace)
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

    return System(coordinates, parameters, L, M, J)


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

    new_parameters = []

    def _get_param_index(p):
        values = [P.value for P in new_parameters]
        if p.value is None or p.value not in values:
            idx = len(new_parameters)
            param = Parameter(f"k_{idx}", value=p.value)
            new_parameters.append(param)
        else:
            idx = values.index(p.value)
        return idx

    x_projectors = {}
    p_projectors = {}
    logger.info("Merging coordinates..")
    for index, (coords, params) in enumerate(pairs):
        x_inverse = {}
        p_inverse = {}

        logger.info(
            "Coordinates: %s, Params %s:", coords, params
        )
        # Parameters can be shared; needs to be many-to-one
        # So we need to check if they're in the parameter set before adding
        # them

        for old_p_index, p in enumerate(params):
            p_idx = _get_param_index(p)
            p_inverse.update({p_idx: old_p_index})

        for idx, x in enumerate(coords):
            new_idx = len(new_coordinates)
            x_inverse.update({new_idx: idx})

            cls = x.__class__
            new_x = cls(f"{cls.default_prefix}_{counters[cls]}")

            counters[cls] += 1
            new_coordinates.append(new_x)

        x_projectors[index] = x_inverse
        p_projectors[index] = p_inverse
    new_coordinates, permuation_map = permutation(
        new_coordinates, canonical_order
    )
    # the permutation map that $x_i -> x_j$ then (i,j) in p_map^T
    permuation = {i: j for i, j in permuation_map}

    for index in x_projectors:
        x_projectors[index] = {
            permuation[i]: j for i, j in x_projectors[index].items()
        }

    projectors = [(x_projectors[i], p_projectors[i]) for i in x_projectors]
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
    - `X` is a `list` of local coordinates
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

    for (new_to_old, param_map), old_coords, nlin in zip(
            maps, coord_list, nonlinear_terms):

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
    return System(coords, params, L, M, J), maps


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

    system, maps = merge_systems(*systems.values())

    map_dictionary = {c: M for c, M in zip(systems.keys(), maps)}

    L_bonds = sympy.SparseMatrix(2*len(model.bonds), system.L.cols, {})

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

    system.L = system.L.col_join(L_bonds)

    return system


def sparse_eye(n):
    """Returns a sparse representation of the identity matrix"""
    return sympy.SparseMatrix(n, n, {(i, i): 1 for i in range(n)})


class InversionError(SymbolicException):
    pass


def _replace_row(system, row, eqn):
    Lp, Mp, Jp = _sympy_to_dict(eqn, system.X)

    for col in range(system.L.cols):
        if col not in Lp:
            system.L[row, col] = 0
        else:
            system.L[row, col] = Lp[col]

    replaced_cols = []
    for idx, term in enumerate(Jp):
        col = _get_or_make_index(system, term)
        system.M[row, col] = Mp[idx]
        replaced_cols.append(col)

    system.M.row_op(row, lambda v, col: v if col in replaced_cols else 0)


def _get_or_make_index(system, term):
    try:
        col = system.J.index(term)
    except ValueError:
        col = len(system.J)
        system.M = system.M.row_join(
            sympy.SparseMatrix(system.M.rows, 1, {})
        )
        system.J.append(term)

    return col


def _get_dummy_vars(f):
    atoms = (a for a in f.atoms() if isinstance(a, BondGraphVariables))
    return [(x, sympy.Dummy(f"_X_{i}")) for i, x in enumerate(atoms)]


def _invert_row(system, row, atoms):

    X, P, L, M, J = system

    nonlinearity = sum(M[row, c] * J[c] for c in range(M.cols))

    for a in atoms:
        f = sum(L[row, i] * X[i] for i in range(L.cols)) + nonlinearity

        dummy_vars = _get_dummy_vars(f)
        try:
            dummy_a, = (v for x, v in dummy_vars if x == a)
        except ValueError:
            continue
        f = f.subs(dummy_vars)
        g = solve_implicit(f, dummy_a)

        if g:
            dummy_inverse = [(v, x) for (x, v) in dummy_vars]
            g = g.subs(dummy_inverse)

            return a, g

    raise InversionError


def _reduce_row(system, row):

    for col in range(system.L.cols):
        v = system.L[row, col]
        if v == 0:
            continue

        if row == col:
            if v != 1:
                system.L.row_op(row, lambda vv, c: vv / v)
                system.M.row_op(row, lambda vv, c: vv / v)
            continue

        d = system.L[col, col]

        if d == 0:
            if col < row:
                system.L.row_swap(col, row)
                system.M.row_swap(col, row)
                _reduce_row(system, col)
                return
            else:
                continue

        coeff = v / d
        system.L.row_op(row, lambda vv, c: vv - coeff * system.L[col, c])
        system.M.row_op(row, lambda vv, c: vv - coeff * system.M[col, c])


def _merge_row(system, row, eqn):
    X, P, L, M, J = system
    logger.info("merging eqn %s", str(eqn))

    Lp, Mp, Jp = _sympy_to_dict(eqn, X)

    atoms = set()

    try:
        pivot_val = Lp[row]
    except KeyError:
        pivot_val = 0

    for col, val in Lp.items():
            L[row, col] = val

    for idx, term in enumerate(Jp):
        atoms |= term.atoms()
        col = _get_or_make_index(system, term)
        try:
            M[row, col] = Mp[idx]
        except AttributeError:
            pass

    for col in range(L.cols):
        v_rj = L[row, col]
        v_jj = L[col, col]
        if not v_jj or not v_rj: continue

        L[row, :] = v_jj*L[row, :] - v_rj*L[col, :]
        M[row, :] = v_jj*M[row, :] - v_rj*M[col, :]

    pivot_val = L[row, row]
    if not (pivot_val == 0) or (pivot_val == 1):
        L.row_op(row, lambda v, _: v / pivot_val)
        M.row_op(row, lambda v, _: v / pivot_val)

    return row, atoms


def _simplify(term):

    out = sympy.expand_log(term, force=True)
    out = sympy.expand_trig(out)

    return out


def _substitute_and_reduce(system, atom, equation):

    # for each element of J we have to substitute the atom for the equation.
    # when we expand the substitution as a sum one of four things can happen
    # a) the term is linear
    #    so we have to move it into L
    # b) the term is an existing nonlinear term
    #    so we have to merge and scale M accordingly
    # c) the term does not exist in J
    #    so we must add it in.

    for idx in range(len(system.J)):
        term = system.J[idx]

        if atom not in term.atoms():
            continue
        logger.info("Reducing %s with %s=%s ", term, atom, equation)
        system.J[idx] = 0
        term = term.subs(atom, equation)
        term = _simplify(term)
        # as an example; consider

        # J = [  x^2 ]
        #     [   xy ]
        #     [exp(u)]
        # with
        # u = log(x + x^2)
        #
        # after substitution we have the third row
        # J_3 = Jl X + JjJ
        #               [x]            [ x^2]
        #     = [1, 0,0][y] + [1, 0 ,0][ xy ]
        #               [u]            [ 0  ]

        Jlin, Jnlin, Jterms = _sympy_to_dict(term, system.X)

        # for the linear part L
        # L := (L + MJ_l)X
        system.L += system.M * sympy.SparseMatrix(
            len(system.J), len(system.X),
            {(idx, col): val for col, val in Jlin.items()}
        )

        # for the nonlinear part
        # M := M(I + J_nlin)J
        for Jn_idx, t in enumerate(Jterms):
            col = _get_or_make_index(system, t)
            system.M.col_op(
                col, lambda v, r: v + Jnlin[Jn_idx] * system.M[r, idx]
            )
    idx = 0
    while idx < len(system.J):
        if system.J[idx] == 0:
            system.M.col_del(idx)
            system.J.remove(idx)
        else:
            idx += 1
    return


def _reduce_constraints(system):
    """Assumes Linear part is in Smith Normal Form"""

    X, P, L, M, J = system

    targets = {
        x for i, x in enumerate(X)
        if L[i, i] == 0 and isinstance(x, (Effort, Flow, DVariable, Output))
    }

    rows = []
    for row in (i for i, x in enumerate(X)
                if not isinstance(x, (DVariable, Output))):

        nonlinearity = sum(M[row, j] * J[j] for j in range(M.cols))
        atoms = nonlinearity.atoms() & targets
        if atoms:
            rows.append((row, atoms))

    # we now have a list or rows that have our target atoms.
    while targets and rows:
        # find the row with the smallest number of target variables.
        rows.sort(key=lambda row_pair: len(row_pair[1]), reverse=True)
        row, atoms = rows.pop()

        # if we can, remove it from the target list and substitute through
        try:
            atom, eqn = _invert_row(system, row, atoms)
        except InversionError:
            continue
        _replace_row(system, row, eqn)
        _reduce_row(system, row)
        targets.remove(atom)

        rows = [(r, a & targets) for (r, a) in rows if a & targets]


def _normalise(system):
    system.L, system.M = smith_normal_form(system.L, system.M)


def _simplify_nonlinear_terms(system, skip=None):

    # need to reset each time since we can reduce nonlinear
    # to linear
    if skip is None:
        skip = set()
    atoms = set(a for term in system.J for a in term.atoms())

    indicies = [i for i, x in enumerate(system.X)
                if system.L[i, i] == 1 and x in atoms and
                i not in skip]
    indicies.sort()
    if not indicies: return

    index = indicies.pop()

    atom = system.X[index]
    # The system is LX + MJ(X) =0
    # Hence, set L = (I - P) -> P = (I - L)
    # Then IX + (L - I)X + MJ(X) = 0
    # and hence X = (I - L)X - MJ(X)

    remainder = -sum(
        system.M[index, c] * system.J[c] for c in range(system.M.cols)
    )

    if atom in remainder.atoms():
        remainder -= sum(system.L[index, c] * system.X[c]
                         for c in range(index + 1, system.L.cols))

    _substitute_and_reduce(system, atom, remainder)

    skip.add(index)
    # tail recurse!
    return _simplify_nonlinear_terms(system, skip)


def reduce(system):
    """ Performs inplace basic symbolic reduction on the given system.

    We assume that the system coordinates are stack so that::

        X = (y, dx, e, f, x, u)

    Args:
        system:

    Returns: system (tuple)
    """
    _normalise(system)
    _reduce_constraints(system)
    _simplify_nonlinear_terms(system)
