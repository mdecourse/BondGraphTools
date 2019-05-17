import sympy


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
    sorted_vect = sorted(vector, key=key)
    return sorted_vect, [(vector.index(v), j) for (j,v) in enumerate(sorted_vect)]


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
    m, n = M.shape
    M = augmented_rref(M, k)

    Mp = sympy.MutableSparseMatrix(n-k, n, {})

    constraints = []
    for row in range(m):
        leading_coeff = -1
        for col in range(row, n-k):
            if M[row, col] != 0:
                leading_coeff = col
                break
        if leading_coeff < 0:
            if not M[row, n-k:].is_zero:
                constraints.append(sum(M[row,:]))
        else:
            Mp[leading_coeff, :] = M[row, :]

    if augment:
        return Mp[:,:-k], Mp[:, -k:], constraints
    else:
        return Mp, sympy.SparseMatrix(m,k,{}), constraints


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


def augmented_rref(matrix, augmented_rows=0, inplace=False):
    """ Computes the reduced row-echelon form (rref) of the given augmented
    matrix.

    That is for the augmented  [ A | B ], we fine the reduced row echelon form
    of A.

    Args:
        matrix (sympy.MutableSparseMatrix): The augmented matrix
        augmented_rows (int): The number of rows that have been augmented onto
         the matrix.
        inplace: Whether or not to do the operations inplace or not

    Returns: a matrix M =  [A' | B'] such that A' is in rref.

    """
    pivot = 0
    m = matrix.cols - augmented_rows
    for col in range(m):
        if matrix[pivot, col] == 0:
            j = None
            v_max = 0
            for row in range(pivot, matrix.rows):
                val = matrix[row, col]
                v = abs(val)
                try:
                    if v > v_max:
                        j = row
                        v_max = v
                except TypeError: # symbolic variable
                    j = row
                    v_max = v
            if not j:
                continue  # all zeros below, skip on to next column
            else:
                matrix.row_swap(pivot, j)

        a = matrix[pivot, col]

        for i in range(matrix.rows):
            if i != pivot and matrix[i, col] != 0:
                b = matrix[i, col]/a
                matrix[i, :] += - b * matrix[pivot, :]

        matrix[pivot, :] *= 1 / a

        pivot += 1

        if pivot >= matrix.rows:
            break
    if inplace:
        return
    else:
        return matrix


def sparse_block_diag(matricies):
    """Creates a block diagonal matrix from the given matricies

    Args:
        matricies (list): A list of matricies to block diagonalise. Assumed to be of type
                         `sympy.SparseMatrix`

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
