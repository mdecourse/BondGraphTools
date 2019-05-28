import pytest


import logging
logging.basicConfig(level=logging.INFO)
import sympy
import BondGraphTools as bgt
from BondGraphTools import connect
from BondGraphTools.model_reduction import *
from test.conftest import assert_implicit

class TestHelpersFunctions:
    def test_as_dict(self):
        matrix = sympy.SparseMatrix(5, 5, {(0, 1):1, (3,4):-1})
        assert as_dict(matrix) == {
            0: {1:1 },
            3: {4: -1}
        }


class TestParameter:
    def test_parameter_creation(self):
        P = Parameter("K")
        P2 = Parameter("K", value=10)
        k = Parameter("k")

        assert P is P2
        assert P.value == 10
        assert P2.evalf() == 10
        assert P.is_number
        assert str(k) == "k"
        P2.value = 1
        assert P.value == 1

    def test_symbolic_cmp(self):
        P = Parameter('K')
        K = sympy.Symbol('K')

        assert str(P) == str(K)
        assert P is not K

        P.value = 10
        assert P.value == sympy.Number(10)
        assert P != K
        assert P != 'K'

    def test_parameter_with_symbolic_value(self):
        v = sympy.sympify("exp(a)")
        P = Parameter('P', value=v)

        assert P.value == v
        assert P.atoms() == {P}

        #
        # assert (sympy.log(P).simplify()) == sympy.Symbol('a')

    def test_funciton_of_param(self):
        P = Parameter('K')
        assert sympy.exp(P) == sympy.exp(P)

class TestBGVariables:

    # def test_variable(self):
    #     x =  Variable('x_1')
    #     dx = Derivative('dx_1')
    #
    #     assert x == Symbol('x_1')

    def test_effort(self):

        e = Effort("e_0")
        e_test = sympy.Symbol("e_0")
        coords = [e, e_test]
        assert e is not e_test
        # assert e == e_test
        locals = {str(e): e}

        eq = sympy.sympify("e_0 - 10", locals=locals)

        assert eq.atoms() == {e, -10}
        assert len(eq.atoms() - set(coords)) == 1

    def test_sort(self):
        syms = list(sympy.symbols("e_1,e_2,f_2,f_1,x_1,dx_1,u_1,y_1"))

        s_symbols = sympy.symbols("y_1, dx_1, e_1,f_1, e_2,f_2,x_1,u_1")

        assert sorted(syms, key=canonical_order) == list(s_symbols)

    def test_permutation_matrix(self):
        syms = list(sympy.symbols("e_1,e_2,f_2,f_1,x_1,dx_1,u_1,y_1"))
        s_symbols = list(sympy.symbols("y_1, dx_1, e_1,f_1, e_2,f_2,x_1,u_1"))

        _, matrix = permutation(syms, key=canonical_order)

        for i,j in matrix:
            assert s_symbols[j] == syms[i]

    def test_product_eq(self):
        P = Parameter('K')
        X = Variable("x_0")
        assert X == X
        assert P * X == P * X
        assert P * X == X *P
        assert sympy.exp(P*X) == sympy.exp(P*X)
        assert sympy.exp(sympy.Mul(P, X)) == sympy.exp(X*P)
        eqn = sympy.exp(X*P)

        assert eqn in [sympy.exp(P*X)]



def test_build_junction_dict():
    c = bgt.new("C")
    kvl = bgt.new("0")
    bg = bgt.new()
    bg.add([c, kvl])
    connect(kvl, c)
    cp,kp = list(c.ports) + list(kvl.ports)
    index_map = {cp:0, kp:1}
    M = adjacency_to_dict(index_map, bg.bonds, offset=1)
    assert M[(0, 1)] == 1
    assert M[(0, 3)] == -1
    assert M[(1, 2)] == 1
    assert M[(1, 4)] == 1


class TestSmithNormalForm(object):
    def test_1(self):

        m = sympy.SparseMatrix(2,3,{(0,2):2, (1,1):1})
        mp = smith_normal_form(m)
        assert mp.shape == (3, 3)

        assert mp[2, 2] != 0

    def test_2(self):
        matrix = sympy.eye(3)
        matrix.row_del(1)

        m = smith_normal_form(matrix)

        diff = sympy.Matrix([[0,0,0],[0,1,0], [0,0,0]])

        assert (sympy.eye(3) - m) == diff

    def test_3(self):
        m = sympy.SparseMatrix(5,3,{(0,1):1, (1,0):1,
                                    (4,2):1})
        mp = smith_normal_form(m)
        assert mp.shape == (3, 3)
        assert (mp - sympy.eye(3)).is_zero


class TestImplicitInversions:

    def test_basic(self):
        eqn = sympy.sympify("k * x -  log(y)")
        var = sympy.S('y')

        result = solve_implicit(eqn, var)
        assert_implicit(result, sympy.sympify('y - exp(k*x)'))

    def test_quadratic(self):
        eqn = sympy.sympify("log(y**2 + x**2)")
        var = sympy.S('y')

        result = solve_implicit(eqn, var)
        assert_implicit(result, sympy.sympify("x**2 + y**2 - 1"))

    def test__exp_product(self):
        eqn = sympy.sympify(" x / (1 + y**2) - z")
        var = sympy.S('x')

        result = solve_implicit(eqn, var)
        outcome = sympy.sympify("x - z*(1+y**2)")

        assert_implicit(result, outcome)


