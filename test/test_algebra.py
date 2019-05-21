import pytest


import logging
logging.basicConfig(level=logging.INFO)
import sympy
import BondGraphTools as bgt
from BondGraphTools import connect, new, expose
from BondGraphTools.model_reduction import *
from BondGraphTools.model_reduction.model_reduction import (_make_coords,
    _generate_atomics_system)

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


class TestParseRelation:
    def test_basic(self):
        ## test 1

        eqn = 'e-R*f'
        X = [Effort('e'), Flow('f')]
        with pytest.raises(SymbolicException):
            parse_relation(eqn, X)
        R = Parameter('R')
        P = [R]

        L, M, J = parse_relation(eqn, X, P)

        assert L == {0:1, 1:-R}
        assert M == {}
        assert J == []

    def test_extended_array(self):
        eqn = "f = dx"
        X = sympy.symbols('dx,e,f,x')

        L, M, J = parse_relation(eqn,X)

        assert L == {0:-1,2:1}
        assert M == {}
        assert J == []

    def test_nonlinear_function(self):
        eqn = "f - I_s*exp(e/V_t) "
        X = sympy.symbols('e,f')
        Is = Parameter('I_s')
        V_t = Parameter('V_t')
        P = [Is, V_t]

        L, M, J = parse_relation(eqn, X, P)

        assert L == {1:1}
        assert M == {0:-Is}
        assert J == [sympy.exp(X[0]/V_t)]

    def test_nonlinear_functions(self):

        eqn = "f_1 = k*exp(e_1) - k*exp(e_2)"
        X = sympy.symbols('e_1,f_1, e_2,f_2')
        k = Parameter('k')
        L, M, J = parse_relation(eqn, X, [k])

        assert L == {1:  1}
        assert M == {0: k, 1: -k}
        assert J == [sympy.exp(X[2]), sympy.exp(X[0])]

    def test_constant_function(self):

        eqn = "x_0 - 1"
        X = [Variable(index=0)]
        L, M, J = parse_relation(eqn, X)
        assert L == {0:1}
        assert M == {0:-1}
        assert J ==[1]

    def test_nonlinear_parameter(self):
        eqn = "e_0 - exp(mu)*f_0"
        P = Parameter('mu')
        X = [Effort(index=0), Flow(index=0)]
        L, M, J = parse_relation(eqn, X, {P})

        assert L == {
            0:1,
            1: -sympy.exp(P)
        }
        assert not M
        assert not J

    def test_free_constant(self):
        eqn = " e_0 - mu - R*T*log(x_0/V)"
        x_0 = Variable(index=0)
        e_0 = Effort(index=0)
        mu = Parameter('mu')
        R = Parameter('R')
        T = Parameter('T')
        V = Parameter('V')

        L, M, J = parse_relation(eqn, [x_0, e_0], {mu, R,T, V})

        assert L == {1:1}
        assert M == {0:-mu, 1: -R*T}
        assert J == [sympy.S(1), sympy.log(x_0/V)]


class TestGenerateCoords():
    def test_C(self):

        c = new("C", value=1)
        coords, params, substitutions = _make_coords(c)

        assert isinstance(coords, list)
        assert isinstance(params, set)
        assert isinstance(substitutions, set)

        false_symbols = sympy.symbols("e_0, f_0, x_0, dx_0")
        found_symbols = set()

        assert substitutions == {(sympy.Symbol('C'), 1)}
        assert len(coords) == 4
        for x in coords:
            assert x not in false_symbols
            for y in false_symbols:
                assert x != y
                if x.name == y.name:
                    found_symbols.add(x.name)

        assert len(found_symbols) == 4

    def test_c_control_var(self):
        c = new("C", value=None)
        coords, params, substitutions = _make_coords(c)

        assert not substitutions
        assert not params
        assert len(coords) == 5

        false_symbols = sympy.symbols("e_0, f_0, x_0, dx_0, C")

        found_symbols = set()

        for x in coords:
            assert x not in false_symbols
            for y in false_symbols:
                assert x != y
                if x.name == y.name:
                    found_symbols.add(x.name)

        assert len(found_symbols) == 5

    def test_se_coords_(self):
        se = new('Se')
        coords, params, subs = _make_coords(se)

        assert str(coords) == "[f, e_0, f_0, e]"
        assert not params
        assert not subs


class TestGenerateSystem:

    def test_r(self):
        model = new("R", value=10)

        # X -> local coordinates
        # P -> Parameters
        # L -> Linear Part of the matrix
        # M -> Matrix for nonlinear terms
        # JX - > nonlinear terms
        X, P, L , M , JX = _generate_atomics_system(model)
        assert len(X) == 2
        assert not P
        assert not M
        assert not JX

        assert as_dict(L) == {0: {0: 1, 1: -10}}
        names = [str(x) for x in X]
        assert names == ["e_0", "f_0"]

    def test_c(self):
        C = Parameter('C', value=10)
        model = new("C", value=C)

        X, P, L, M, JX = _generate_atomics_system(model)
        assert len(X) == 4
        assert P == {C}
        assert not M
        assert not JX

        assert L.rows == 2

        for row in as_dict(L).values():
            assert row in ({0: 1, 2: -1}, {1: -C, 3: 1})

        names = [str(x) for x in X]
        assert names == ["dx_0", "e_0", "f_0", "x_0"]

    def test_se(self):
        se = new("Se")
        X, P, L, M, JX = _generate_atomics_system(se)

        assert str(X) == "[f, e_0, f_0, e]"
        assert not P

        for row in as_dict(L).values():
            assert row in [{0: 1, 2: 1}, {1: 1, 3: -1}]

        assert not M
        assert not JX


class TestMerge:
    def test_merge_coords(self):
        c = new("C", value=Parameter('C'))
        c_1, p_1, subs_1 = _make_coords(c)
        r = new("R", value=Parameter('R'))
        c_2, p_2, subs_2 = _make_coords(r)

        assert len(c_1) == 4
        assert len(c_2) == 2

        (c, p), maps = merge_coordinates(
            (c_1,p_1), (c_2,p_2)
        )

        assert len(c) == len(c_1) + len(c_2)
        assert len(p) == len(p_1) + len(p_2)
        assert maps == [{0:0, 1:1, 2:2, 5:3}, {3:0, 4:1}]

    def test_common_param(self):

        p = Parameter('C')
        C = new("C", value=p)
        c_1, p_1, subs_1 = _make_coords(C)
        C2 = new("C", value=p)
        c_2, p_2, subs_1 = _make_coords(C2)

        assert len(c_1) == len(c_2) == 4
        assert len(p_1) == len(p_2) == 1
        (c, p), maps = merge_coordinates(
            (c_1,p_1), (c_2,p_2)
        )

        assert len(p) == 1

    def test_merge_systems(self):
        p1 = Parameter('C')
        p2 = Parameter('R')
        c = new("C", value=p1)
        r = new("R", value=p2)
        system_1 = _generate_atomics_system(c)
        system_2 = _generate_atomics_system(r)

        coords, params, L, M, J, maps = merge_systems(system_1, system_2)

        assert str(coords) == "[dx_0, e_0, f_0, e_1, f_1, x_0]"

        assert len(params) == 2
        assert not M
        assert not J
        assert as_dict(L) == {
            0: {1: -p1, 5: 1},
            1: {0: 1, 2: -1},
            2: {3: 1, 4: -p2}
        }

    def test_merge_nonlinear_system(self):
        P = Parameter('P')
        # K = sympy.Symbol('k')
        K = 1
        Ce = new("Ce", library="BioChem", value={"R": P, "T": 1, "k": K})
        Re = new("Re", library="BioChem", value={"R": P, "T": 1, "r": None})
        system_1 = _generate_atomics_system(Ce)
        system_2 = _generate_atomics_system(Re)

        coords, params, L, M, J, maps = merge_systems(system_1, system_2)

        assert str(coords) == '[dx_0, e_0, f_0, e_1, f_1, e_2, f_2, x_0, u_0]'
        assert params == {P}
        assert as_dict(L) == {
            0: {1: 1},
            1: {0: -1, 2: 1},
            2: {4: 1, 6: 1},
            3: {4: 1}
        }
        assert as_dict(M) == {
            0: {0: -P},
            3: {1: 1, 2: -1}
        }
        u = coords[-1]
        x = coords[-2]
        e_2 = coords[3]
        e_3 = coords[5]
        assert J == [sympy.log(x), u*sympy.exp(e_3 / P), u*sympy.exp(e_2 / P)]


class Test_generate_system_from:
    def test_compound(self):
        p1 = Parameter('C')
        p2 = Parameter('R')
        c = new("C", value=p1)
        r = new("R", value=p2)
        j = new("0")
        model = new()
        model.add(c, r, j)
        # should add 4 extra coordinates
        connect(c, j)
        connect(r, j)

        coords, params, L, M, J = generate_system_from(model)

        assert len(coords) == 10
        assert len(params) == 2
        assert not M
        assert not J
        assert as_dict(L) == {
            0: {1: -p1, 9: 1},  # C_1
            1: {0: 1, 2: -1},   # C_2
            2: {3: 1, 4: -p2},  # R_1
            3: {5: 1, 7: -1},   # 0 Junction
            4: {6: 1, 8: 1},    # 0 Junction
            5: {1: -1, 5: 1},
            6: {2: 1, 6: 1},
            7: {3: -1, 7: 1},
            8: {4: 1, 8: 1}
        }

class Test_getnexteq:
    from BondGraphTools.model_reduction.model_reduction import _get_next_eq

    def test_1(self):
        raise NotImplementedError

class TestSympyToDict:
    from BondGraphTools.model_reduction.model_reduction import _sympy_to_dict

    def test_1(self):
        raise NotImplementedError

class Test_merge_in:
    from BondGraphTools.model_reduction.model_reduction import _merge_in
    def test_1(self):
        raise NotImplementedError

class Test_make_ef_invertible:
    from BondGraphTools.model_reduction.model_reduction import _make_ef_invertible
    def test_1(self):
        raise NotImplementedError

class Test_Reduce:
    pass

class TestSmithNormalForm(object):
    def test_1(self):

        m = sympy.SparseMatrix(2,3,{(0,2):2, (1,1):1})
        mp = smith_normal_form(m)
        print(mp)
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


def test_build_relations():
    c = bgt.new("C")
    eqns = c._build_relations()

    test_eqn = {sympy.sympify("x_0 - C*e_0"),
                sympy.sympify("dx_0 - f_0")}

    assert set(eqns) == test_eqn


def test_zero_junction_relations():
    r = bgt.new("R", value=sympy.symbols('r'))
    l = bgt.new("I", value=sympy.symbols('l'))
    c = bgt.new("C", value=sympy.symbols('c'))
    kvl = bgt.new("0", name="kvl")

    rlc = bgt.new()
    rlc.add([c, l, kvl, r])

    connect(r, kvl)
    connect(l, kvl)
    connect(c, kvl)

    rels = kvl.constitutive_relations

    assert sympy.sympify("e_1 - e_2") in rels
    assert sympy.sympify("e_0 - e_2") in rels
    assert sympy.sympify("f_0 + f_1 + f_2") in rels


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


def test_build_model_fixed_cap():
    c = bgt.new("C", value=0.001)

    eqns = c.constitutive_relations
    assert len(eqns) == 2

    test_eqn1 = sympy.sympify("x_0 - 0.001*e_0")
    test_eqn2 = sympy.sympify("dx_0-f_0")

    assert test_eqn1 in eqns
    assert test_eqn2 in eqns


def test_cv_relations():
    c = bgt.new("C", value=1)
    se = bgt.new("Se")
    r = bgt.new("R", value=1)
    kcl = bgt.new("1")
    bg = bgt.new()
    bg.add([c, se, kcl, r])

    connect(c, (kcl,kcl.non_inverting))
    connect(r, (kcl, kcl.non_inverting))
    connect(se, (kcl, kcl.non_inverting))
    print(bg.constitutive_relations)
    assert bg.constitutive_relations == [sympy.sympify("dx_0 + u_0 + x_0")]


def test_parallel_crv_relations():
    c = bgt.new("C", value=1)
    se = bgt.new("Se")
    r = bgt.new("R", value=1)
    kcl = bgt.new("0")
    bg = bgt.new()
    bg.add([c, se, kcl, r])

    connect(c, kcl)
    connect(se, kcl)
    connect(r, kcl)

    assert bg.constitutive_relations == [sympy.sympify("dx_0 - du_0"),
                                         sympy.sympify("x_0 - u_0")]

#
# def test_generate_subs():
#
#     w, x, y, z = sympy.sympify("w,x,y,z")
#     size_tuple  =(0, 2, 0,4 )
#     coords = [w, x, y, z]
#     #  w + w^2 + x^2
#     #  x + 1 + y^2   < should appear in subs as x = -y^2  - 1
#     #  y + 1         <                          y = - 1
#     #  0 + z^2 + w^2
#
#     L = sympy.SparseMatrix(4, 4, {(0,0): 1,
#                                   (1,1): 1,
#                                   (2,2): 1})
#
#     N = sympy.SparseMatrix(4, 1, {(0, 0): w**2 + x**2,
#                                   (1, 0): 1 + y**2,
#                                   (2, 0): 1})
#
#     constraint = [z**2 + w**2]
#     subs = _generate_substitutions(L, N,constraint, coords, size_tuple)
#     target_subs = [(y,-1), (x, -1-y**2)]
#
#     assert subs == target_subs
#
#
# def test_cv_subs_func():
#     c = bgt.new("C", value=1)
#     se = bgt.new("Se")
#     r = bgt.new("R", value=1)
#     kcl = bgt.new("1")
#     bg = bgt.new()
#     bg.add([c, se, kcl, r])
#
#     connect(c,(kcl,kcl.non_inverting))
#     connect(r, (kcl, kcl.non_inverting))
#     connect(se, (kcl, kcl.non_inverting))
#
#     cv_s = {'u_0': ' -exp(-t)'}
#
#     subs = [(sympy.Symbol('u_0'), sympy.sympify('-exp(-t)'))]
#
#     mappings, coords = inverse_coord_maps(*bg.basis_vectors)
#     assert _generate_cv_substitutions(cv_s, mappings,coords) == subs
#
#
# def test_cv_subs_const():
#     c = bgt.new("C", value=1)
#     se = bgt.new("Se")
#     r = bgt.new("R", value=1)
#     kcl = bgt.new("1")
#     bg = bgt.new()
#     bg.add([c, se, kcl, r])
#
#     connect(c,(kcl,kcl.non_inverting))
#     connect(r, (kcl, kcl.non_inverting))
#     connect(se, (kcl, kcl.non_inverting))
#
#     cv_s = {'u_0': ' 2'}
#
#     subs = [(sympy.Symbol('u_0'), sympy.S(2))]
#
#     mappings, coords = inverse_coord_maps(*bg.basis_vectors)
#     assert _generate_cv_substitutions(cv_s, mappings,coords) == subs

#
# def test_cv_subs_state_func():
#     c = bgt.new("C", value=1)
#     se = bgt.new("Se")
#     r = bgt.new("R", value=1)
#     kcl = bgt.new("1")
#     bg = bgt.new()
#     bg.add([c, se, kcl, r])
#
#     connect(c,(kcl,kcl.non_inverting))
#     connect(r, (kcl, kcl.non_inverting))
#     connect(se, (kcl, kcl.non_inverting))
#
#     cv_s = {'u_0': ' -exp(-x_0)'}
#
#     subs = [(sympy.Symbol('u_0'), sympy.sympify('-exp(-x_0)'))]
#
#     mappings, coords = inverse_coord_maps(*bg.basis_vectors)
#     assert _generate_cv_substitutions(cv_s, mappings,coords) == subs


#
# def test_ported_cr():
#     model = bgt.new()
#     Sf = bgt.new('Sf', name="Sf")
#     R = bgt.new("R", value=2)
#     zero = bgt.new("0")
#     ss = bgt.new("SS")
#
#     model.add(Sf, R, zero, ss)
#     connect(Sf, zero)
#     connect(R, zero)
#     connect(ss, zero)
#
#     bgt.expose(ss, 'A')
#     assert len(model.control_vars) == 1
#
#     ts, ps, cs = model._build_internal_basis_vectors()
#     assert len(cs) == 1
#     assert len(ps) == 7
#     assert len(ts) == 0
#
#     mapping, coords = inverse_coord_maps(ts, ps, cs)
#     assert len(coords) == 15
#
#     coords, mappings, lin_op, nl_op, conttr = model.system_model()
#     assert nl_op.is_zero
#     assert not conttr
#
#     assert model.constitutive_relations == [
#         sympy.sympify('e_0 - 2*f_0 - 2*u_0')
#     ]

def test_ported_series_resistor():

    Se = new("Se")
    r1 = new("R", value=1)
    r2 = new("R", value=2)
    kvl = new('1')
    ss = new("SS")
    model = new()
    model.add(
        Se,r1,r2,kvl, ss
    )
    expose(ss)
    connect(Se, kvl.non_inverting)
    connect(kvl.inverting, r1)
    connect(kvl.inverting, r2)
    connect(kvl.inverting, ss)

    assert len(model.ports) == 1

    assert model.constitutive_relations == [
        sympy.sympify("e_0 - 3*f_0 - u_0")
    ]

def test_ported_cap():
    model = new()
    c = new("C", value=3)
    zero = new("0")
    ss = new("SS")
    model.add(
        c, zero, ss
    )

    connect(c, zero)
    connect(ss, zero)

    expose(ss)
    assert len(model.ports) == 1


    assert model.constitutive_relations == [
        sympy.sympify("dx_0 - f_0"),sympy.sympify("e_0 - x_0/3")
    ]
def test_ported_parallel_rc():

    model = new()
    r = new("R", value=2)
    c = new("C", value=3)
    zero = new("0")
    ss = new("SS")
    model.add(
        r,c,zero, ss
    )

    connect(r,zero)
    connect(c,zero)
    connect(ss, zero)

    expose(ss)
    assert len(model.ports) == 1

    assert model.constitutive_relations == [
        sympy.sympify("dx_0 + x_0/6 - f_0"),
        sympy.sympify("e_0 - x_0/3")
    ]
