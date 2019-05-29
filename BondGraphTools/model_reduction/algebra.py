"""Algebra.py

Symbolic Algebra Helper functions.
"""

import sympy

__all__ = [
    "permutation",
    "sparse_block_diag",
    "adjacency_to_dict",
    "smith_normal_form",
    "flatten",
    "solve_implicit"
]


def permutation(vector, key=None):
    """
    Args:
        vector: The vector to sort
        key: Optional sorting key (See: `sorted`)

    Returns:
        (vector, list)

    For a given iterable, produces a list of tuples representing the
    permutation that maps sorts the list.

    Examples:
        >>> permutation([3,2,1])
        outputs `[1,2,3], [(0,2),(1,1),(2,0)]`
    """
    sorted_vector = sorted(vector, key=key)
    permutations = [
        (vector.index(v), j) for (j, v) in enumerate(sorted_vector)
    ]

    return sorted_vector, permutations


def adjacency_to_dict(nodes, edges, offset=0):
    """
    matrix has 2*#bonds rows
    and 2*#ports columes
    so that MX = 0 and X^T = (e_1,f_1,e_2,f_2)

    Args:
        nodes:
        edges:
        offset

    Returns:
        `dict` with keys (row, column)

    """
    M = dict()

    for i, (node_1, node_2) in enumerate(edges):
        j_1 = offset + 2 * nodes[node_1]
        j_2 = offset + 2 * nodes[node_2]
        # effort variables
        M[(2 * i, j_1)] = - 1
        M[(2 * i, j_2)] = 1
        # flow variables
        M[(2 * i + 1, j_1 + 1)] = 1
        M[(2 * i + 1, j_2 + 1)] = 1

    return M


def smith_normal_form(matrix, augment=None):
    """Computes the Smith normal form of the given matrix.

    Args:
        matrix:
        augment:

    Returns:
        n x n smith normal form of the matrix.
        Particularly for projection onto the nullspace of M and the orthogonal
        complement that is, for a matrix M,
        P = _smith_normal_form(M) is a projection operator onto the nullspace of M
    """
    if augment:
        M = matrix.row_join(augment)
        k = augment.cols
    else:
        M = matrix
        k = 0
    rows = matrix.cols

    M = M.rref(pivots=False)
    M = M.col_join(
        sympy.SparseMatrix(matrix.cols - matrix.rows, M.cols, {})
    )

    for row in reversed(range(rows)):
        pivot_col = next((col for col in range(rows) if M[row, col] != 0), -1)

        if pivot_col > 0:
            M.row_swap(row, pivot_col)

    if augment:
        return M[:, :-k], M[:, -k:]
    else:
        return M


def flatten(sequence):
    """
    Gets a first visit iterator for the given tree.
    Args:
        sequence: The iterable that is to be flattened

    Returns: iterable
    """
    for item in sequence:
        if isinstance(item, (list, tuple)):
            for subitem in flatten(item):
                yield subitem
        else:
            yield item


def sparse_block_diag(matricies):
    """Creates a block diagonal matrix from the given matricies

    Args:
        matricies (list): A list of matricies to block diagonalise. Assumed to
                          be of type `sympy.SparseMatrix`

    Returns:
        `sympy.SparseMatrix`

    """
    values = {}
    rows = 0
    cols = 0

    for matrix in matricies:
        values.update(
            {(r + rows, c + cols): v for r, c, v in matrix.RL}
        )
        n, m = matrix.shape
        rows += n
        cols += m

    return sympy.SparseMatrix(rows, cols, values)


def _parse_finite_sets(var, *args):

    if len(args) == 1 and isinstance(args[0], sympy.Expr):
        result = var - args[0]
        result.simplify()
        return result
    elif len(args) == 2 and isinstance(args[0], sympy.Expr):
        a1, a2 = args
        arg_sum = (a1 + a2).expand()
        if arg_sum == 0:
            result = a1*a2 + var ** 2
            result.simplify()
            return result

    return None


def _parse_set_complement(var, universe, divisor):
    if not divisor & sympy.Reals:
        if isinstance(universe, sympy.FiniteSet):
            return _parse_finite_sets(var, *universe)

    return None


def solve_implicit(eqn, var):

    soln = sympy.solveset(eqn, var)

    if isinstance(soln, sympy.FiniteSet):
        return _parse_finite_sets(var, *soln.args)
    if isinstance(soln, sympy.Complement):
        return _parse_set_complement(var, soln.args[0], soln.args[1])
    else:
        return None

