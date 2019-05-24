import pytest
from sympy import SparseMatrix, sympify, symbols
import re
from BondGraphTools.exceptions import SymbolicException
from BondGraphTools.model_reduction import parse_relation
from BondGraphTools import new, connect
import sympy
from BondGraphTools.model_reduction.model_reduction import _make_coords
from BondGraphTools.model_reduction.model_reduction import _generate_atomics_system
from BondGraphTools.model_reduction.symbols import *
from BondGraphTools.model_reduction import *


class DummyPort(object):
    def __init__(self, index):
        self.index = int(index)


class DummyModel(object):
    def __init__(self, constitutive_relations, params=None):
        self.constitutive_relations = constitutive_relations

        if params:
            self.params = params
        else:
            self.params = {}

        self.state_vars = []
        self.ports = []
        self.control_vars = []
        self.output_vars = []

        for r in self.constitutive_relations:
            atoms = sympify(r).atoms()
            for atom in atoms:
                if atom.is_number:
                    continue
                a = str(atom)
                try:
                    prefix, index = a.split('_')
                except ValueError:
                    self.params.update({atom: atom})
                    continue

                if prefix == "x" and a not in self.state_vars:
                    self.state_vars.append(a)

                elif prefix == 'dx' and a[1:] not in self.state_vars:
                    self.state_vars.append(a[1:])

                elif prefix in ('e', 'f') and (int(index) not in
                                               {p.index for p in self.ports}):

                    p = DummyPort(index)
                    self.ports.append(p)
                elif prefix == 'u' and a not in self.control_vars:
                    self.control_vars.append(a)
                elif prefix == 'y' and a not in self.outputs_vars:
                    self.outputs_vars.append(a)


class TestDummyModel:
    """ Tests to make sure our dummy model fixture works"""
    def test_coords(self):


        dummy_model = DummyModel(
            ["x_0 - e_0", "f_1 - dx_0", "u_0 - k * e_0"]
        )

        X, P, S = _make_coords(dummy_model)

        assert str(X) == "[dx_0, e_0, f_0, e_1, f_1, x_0, u_0]"
        assert len(P) == 1
        assert len(S) == 1

    def test_coords_nonlinear(self):

        dummy_model = DummyModel(
            ["x_0^2 - e_0", "exp(f_1) - dx_0", "u_0 - k * e_0"]
        )

        X, P, S = _make_coords(dummy_model)

        assert str(X) == "[dx_0, e_0, f_0, e_1, f_1, x_0, u_0]"
        assert len(P) == 1
        assert len(S) == 1

    def test_coords_nonlinear_2(self):

        dummy_model = DummyModel(
            ["dx_0 - e_0", "f_0 - x_0^2"]
        )

        for v in dummy_model.state_vars:
            print(v)

        X, P, S = _make_coords(dummy_model)

        assert str(X) == "[dx_0, e_0, f_0, x_0]"
        assert len(P) == 0
        assert len(S) == 0

    def test_linear(self):



        dummy_model = DummyModel(
            ["x_0 - e_0", "f_1 - dx_0", "u_0 - k * e_0"]
        )
        # Matrix should be
        #    [ 0, -1,  0,  0,  0,  1,  0],
        #    [-1,  0,  0,  0,  1,  0,  0],
        #    [ 0, -k,  0,  0,  0,  0,  1]

        X, P, L, M, J = _generate_atomics_system(dummy_model)
        assert str(X) == "[dx_0, e_0, f_0, e_1, f_1, x_0, u_0]"
        k = next(p for p in P)

        rl = L.row_list()
        assert rl[:4] == [
            (0, 1, -1),
            (0, 5, 1),
            (1, 0, -1),
            (1, 4, 1)
        ]

        r, c, v = rl[4]
        assert r == 2
        assert c == 1


        assert str(k) == str(-v) # TODO: fix parameter comparison
        assert rl[5] == (2, 6, 1)

        assert not M
        assert not J

    def test_nonlinear(self):

        dummy_model = DummyModel(
            ["dx_0 - e_0", "f_0 - x_0^2"]
        )

        # Resulting L should be
        # [1, -1, 0, 0]
        # [0,  0, 1, 0]

        # Resulting M should be
        # [0]
        # [-1]

        # resulting J should be x_0^2

        X, P, L, M, J = _generate_atomics_system(dummy_model)

        assert len(X) == 4

        assert L.row_list() == [
            (0, 0, 1),
            (0, 1, -1),
            (1, 2, 1)
        ]

        assert M.row_list() == [
            (1, 0, -1)
        ]

        assert len(J) == 1

        assert J[0] == X[-1]**2





class TestModelReduction:
    def test_reduciton_without_elimination(self):


        eqns = [
            "dx_0 - x_0 - log(e_0)",
            "f_0 - x_0",
            "k * u_0 - log(e_0)"
        ]

        model = DummyModel(eqns)

        system = generate_system_from(model)
        system = reduce(system)

        X, P, L, M, J = system

        k = next(p for p in P)
        # should directly reduce to
        L_test =[
            [1, 0, 0, -1, -k],
            [0, 1, 0,  0,  0],
            [0, 0, 1,  -1, 0],
            [0, 0, 0,   0, 0],
            [0, 0, 0,   0, 0]
        ]

        from sympy import exp, Mul

        assert J == [exp(Mul(k, X[-1]))]

        M_test = [[0], [1], [0], [0], [0]]
        _cmp(M, M_test)
        _cmp(L, L_test)


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
        X = symbols('dx,e,f,x')

        L, M, J = parse_relation(eqn,X)

        assert L == {0:-1,2:1}
        assert M == {}
        assert J == []

    def test_nonlinear_function(self):
        eqn = "f - I_s*exp(e/V_t) "
        X = symbols('e,f')
        Is = Parameter('I_s')
        V_t = Parameter('V_t')
        P = [Is, V_t]

        L, M, J = parse_relation(eqn, X, P)

        assert L == {1:1}
        assert M == {0:-Is}
        assert J == [exp(X[0]/V_t)]

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


def _cmp(sparse, dense):
    from sympy import Matrix

    M = Matrix(dense)

    diff = M - sparse

    errors = []
    for i in range(diff.rows):
        for j in range(diff.cols):
            v = diff[i, j]
            if v != 0:
                errors.append((i,j,v))
    assert not errors





