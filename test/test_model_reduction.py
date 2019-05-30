import pytest
import logging
logging.basicConfig(level=logging.INFO)
from test.conftest import assert_implicit
from sympy import SparseMatrix, sympify, symbols
import re
from BondGraphTools.exceptions import SymbolicException
from BondGraphTools.model_reduction import parse_relation
from BondGraphTools import new, connect
import sympy
from BondGraphTools.model_reduction.model_reduction import (
    _make_coords, generate_system_from_atomic, _normalise, _reduce_constraints,
    _invert_row, _replace_row, _reduce_row, _simplify_nonlinear_terms,
    _substitute_and_reduce)
from BondGraphTools.model_reduction.symbols import *
from BondGraphTools.model_reduction import *
from sympy import exp

class DummyPort(object):
    def __init__(self, index):
        self.index = int(index)


class DummyModel(object):
    def __init__(self, constitutive_relations, params=None, uri="d1"):
        self.equations = constitutive_relations

        if params:
            self.params = params
        else:
            self.params = {}

        self.state_vars = []
        self.ports = []
        self.control_vars = []
        self.output_vars = []
        self.uri = uri
        self.parent = None
        self.root = self

        for r in self.equations:
            atoms = sympify(r).atoms()
            for atom in atoms:
                if atom.is_number:
                    continue
                a = str(atom)
                try:
                    prefix, index = a.split('_')
                except ValueError:
                    if a not in self.params:
                        self.params.update({a: atom})
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


class TestGenerateCoords:
    """ Tests to make we corredtly identify coordinates"""
    def test_coords(self):
        dummy_model = DummyModel(
            ["x_0 - e_0", "f_1 - dx_0", "u_0 - k * e_0"]
        )

        X, P, S = _make_coords(dummy_model)

        assert str(X) == "[dx_0, e_0, f_0, e_1, f_1, x_0, u_0]"
        assert len(P) == 1
        assert len(X) + len(P) == len(S)

    def test_coords_nonlinear(self):

        dummy_model = DummyModel(
            ["x_0^2 - e_0", "exp(f_1) - dx_0", "u_0 - k * e_0"]
        )

        X, P, S = _make_coords(dummy_model)

        assert str(X) == "[dx_0, e_0, f_0, e_1, f_1, x_0, u_0]"
        assert len(P) == 1
        assert len(S) == 8

    def test_coords_nonlinear_2(self):

        dummy_model = DummyModel(
            ["dx_0 - e_0", "f_0 - x_0^2"]
        )

        X, P, S = _make_coords(dummy_model)

        assert str(X) == "[dx_0, e_0, f_0, x_0]"
        assert len(P) == 0
        assert len(S) == 4


class TestGenerateParams:
    def test_params_1(self):
        "The absence of a value means a unique symbol"
        dummy_model = DummyModel(
            ["dx_0 - k * x_0"]
        )

        X, P, S = _make_coords(dummy_model)
        dx, x = X
        k, = P
        assert isinstance(k, Parameter)
        assert S['k'] is k

    def test_params_2(self):
        "when a value is specified, we should remove the symbol entirely"
        dummy_model = DummyModel(
            ["dx_0 - k * x_0"],
            {'k': 1}
        )

        assert dummy_model.params['k'] == 1

        X, P, S = _make_coords(dummy_model)
        dx, x = X
        assert not P
        assert S['k'] == 1

    def test_params_3(self):
        "when a value is symbolic, the symbol should be stored in the value"
        p = Parameter('k')
        dummy_model = DummyModel(
            ["dx_0 - k * x_0"],
            {'k': Parameter('k')}
        )

        X, P, S = _make_coords(dummy_model)
        assert len(P) == 1
        assert S['k'] is p

    def test_params_4(self):
        "when a value is expression"
        p = Parameter('k')
        dummy_model = DummyModel(
            ["dx_0 - k * x_0"],
            {'k': exp(p)}
        )

        X, P, S = _make_coords(dummy_model)
        assert len(P) == 1
        assert p in P
        assert S['k'] == exp(p)

    def test_parse_relation(self):

        dummy_model = DummyModel(
            ["dx_0 - k * x_0"],
            {'k': 1}
        )

        assert dummy_model.params['k'] == 1

        X, P, S = _make_coords(dummy_model)
        assert not P

        Lp, Mp, Jp = parse_relation(dummy_model.equations[0],
                                    X, P, S)

        assert Lp == {0:1, 1:-1}
        assert not Mp
        assert not Jp

    def test_parse_relation_2(self):
        dummy_model = DummyModel(
            ["dx_0 - k * x_0"]
        )

        X, P, S = _make_coords(dummy_model)
        kp, = P

        Lp, Mp, Jp = parse_relation(dummy_model.equations[0],
                                    X, P, S)
        assert Lp == {0:1, 1: -kp}
        assert not Mp
        assert not Jp

    def test_parse_relation_3(self):
        e = Effort('e_0')
        f = Flow('f_0')
        X = [e, f]
        namespace = {'e_0': e, 'f_0':f, 'r':sympy.Number(10)}
        eqn = "e_0 - r*f_0"
        Lp, Mp, Jp = parse_relation(eqn, X, {}, namespace)
        assert Lp == {
            0:1, 1: -10
        }
        assert not Mp
        assert not Jp

    def test_linear(self):

        dummy_model = DummyModel(
            ["x_0 - e_0", "f_1 - dx_0", "u_0 - k * e_0"]
        )
        # Matrix should be
        #    [ 0, -1,  0,  0,  0,  1,  0],
        #    [-1,  0,  0,  0,  1,  0,  0],
        #    [ 0, -k,  0,  0,  0,  0,  1]

        X, P, L, M, J = generate_system_from_atomic(dummy_model)
        assert str(X) == "[dx_0, e_0, f_0, e_1, f_1, x_0, u_0]"
        k = next(p for p in P)
        assert isinstance(k, Parameter)
        rl = L.row_list()
        assert rl[:4] == [
            (0, 1, -1),
            (0, 5, 1),
            (1, 0, -1),
            (1, 4, 1)
        ]

        r, c, v = rl[4]
        vm = (-v).expand()
        assert vm.__class__ == Parameter
        assert r == 2
        assert c == 1
        assert vm == k  # TODO: fix parameter comparison
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

        X, P, L, M, J = generate_system_from_atomic(dummy_model)

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


class TestNormalise:
    def test_normalise(self):
        dummy_model = DummyModel(
            ["f_0 - x_0^2", "dx_0 - e_0"]
        )
        system = generate_system_from_atomic(dummy_model)

        X, P, L, M, J = system
        assert L.row_list() == [
            (0, 2, 1),
            (1, 0, 1),
            (1, 1, -1)
        ]

        _normalise(system)
        X, P, L, M, J = system
        assert L.row_list() == [
            (0, 0, 1),
            (0, 1, -1),
            (2, 2, 1)
        ]

        assert M.row_list() == [
            (2, 0, -1)
        ]


class TestModelReduction:
    def test_replace_row(self):
        eqns = [
            "dx_0 - x_0 - log(e_0)",
            "f_0 - x_0",
            "k * u_0 - log(e_0)"
        ]

        model = DummyModel(eqns)
        system = generate_system_from_atomic(model)
        _normalise(system)
        # want to replace row 5 with
        k = list(system.P)[0]
        eqn = system.X[1] - exp(k * system.X[4])
        _replace_row(system, 4, eqn)
        term = exp(k * system.X[4])

        _assert_in(term, system.J)
        assert len(system.J) == 2
        assert system.L[4, 1] == 1
        assert system.M[4, 1] == -1, system.M[4,:]
        assert system.M[1, 0] == 0

    def test_invert_row(self):
        eqns = [
            "dx_0 - x_0 - log(e_0)",
            "f_0 - x_0",
            "k * u_0 - log(e_0)"
        ]

        model = DummyModel(eqns)
        system = generate_system_from_atomic(model)
        _normalise(system)
        # X = [dx_0, e_0, f_0, x_0, u_0]
        _, e_0, f_0, _, u_0 = system.X
        k, = system.P
        # row 5 is k*u - log(e_0) = 0
        # invert it with respect to e0

        assert isinstance(k, Parameter)
        assert system.L[4, 4] == 1
        assert system.L[1,1] == 0
        assert system.M[4, 0] == (-1) / k

        atoms = {f_0, e_0}
        atom, eqn = _invert_row(system, 4, atoms)
        _replace_row(system, 4, eqn)
        _reduce_row(system, 4)
        assert system.L[4, 4] == 0
        assert system.L[1, 1] == 1

    def test_reduce_row(self):
        eqns = [
            "dx_0 - x_0 - log(e_0)",
            "f_0 - x_0",
            "k * u_0 - log(e_0)"
        ]

        model = DummyModel(eqns)
        system = generate_system_from_atomic(model)
        _normalise(system)

    def test_reduce_constraints(self):
        eqns = [
            "dx_0 - x_0 - log(e_0)",
            "f_0 - x_0",
            "k * u_0 - log(e_0)"
        ]

        model = DummyModel(eqns)
        system = generate_system_from_atomic(model)
        _normalise(system)
        _reduce_constraints(system)

        for i in range(3):
            assert system.L[i, i] == 1, system.L

        assert system.L[3, 3] == 0, system.L
        assert system.L[4, 4] == 0, system.L
        assert len(system.J) == 2
        assert system.M[1, 0] == 0

    def test_substitute_and_reduce(self):
        eqns = [
            "dx_0 - x_0 - log(e_0)",
            "f_0 - x_0",
            "k * u_0 - log(e_0)"
        ]

        model = DummyModel(eqns)
        system = generate_system_from_atomic(model)
        _normalise(system)
        assert system.L[1, :].is_zero
        _reduce_constraints(system)
        atom = system.X[1]
        p, = system.P
        assert system.L[1, 1] == 1
        for col in range(system.L.cols):
            if col == 1:
                assert system.L[1, col] ==1
            else:
                assert system.L[1,col] == 0

        remainder = exp(p*system.X[-1])

        _substitute_and_reduce(system, atom, remainder)

        assert len(system.J) == 1, system.J
        assert system.M.cols == 1, system.M
        assert system.L[0, 0] == 1
        assert system.L[0, 3] == -1
        assert system.L[0, 4] == -p
        assert system.M[0, 0] == 0
        assert system.M[1, 0] == -1
        assert system.J == [exp(system.X[-1]*p)]
        L = system.L
        row = list(L[1, j] for j in range(L.cols))

        assert row == [0, 1, 0, 0, 0]

    def test_perform_substitutions(self):
        eqns = [
            "dx_0 - x_0 - log(e_0)",
            "f_0 - x_0",
            "k * u_0 - log(e_0)"
        ]

        model = DummyModel(eqns)
        system = generate_system_from_atomic(model)
        _normalise(system)
        _reduce_constraints(system)
        p, = system.P
        _simplify_nonlinear_terms(system)

        assert len(system.J) == 1, system.J
        assert system.M.cols == 1, system.M
        assert system.L[0, 0] == 1
        assert system.L[0, 3] == -1
        assert system.L[0, 4] == -p
        assert system.M[0, 0] == 0

    def test_reduciton_without_elimination(self):

        eqns = [
            "dx_0 - x_0 - log(e_0)",
            "f_0 - x_0",
            "k * u_0 - log(e_0)"
        ]

        model = DummyModel(eqns)

        system = generate_system_from_atomic(model)
        reduce(system)

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

        eqn = exp(k * X[-1])
        _assert_in(eqn,  J)

        M_test = [[0], [-1], [0], [0], [0]]
        _cmp(M, M_test)
        _cmp(L, L_test)


class TestMergeSystems:
    def test_merge_coordinates(self):

        d1 = DummyModel(["dx_0 - c * x_0"], uri="d1")
        d2 = DummyModel(["dx_0 + c * x_0"], uri="d2")

        assert d1.uri != d2.uri
        coords_1, params_1, subs_1 = _make_coords(d1)
        coords_2, params_2, subs_2 = _make_coords(d2)

        assert len(coords_1) == len(coords_2) == 2
        assert len(params_1) == len(params_2) == 1
        p1, = params_1
        p2, = params_2
        assert p1.value == p2.value == sympy.Symbol('c')

        args = (coords_1, params_1), (coords_2, params_2)

        (coords, params), inverses = merge_coordinates(*args)

        (c1_inv, p1_inv), (c2_inv, p2_inv) = inverses
        # new - to - old mappings
        assert len(coords) == 4
        assert len(params) == 1
        assert c1_inv == {0: 0, 2: 1}
        assert c2_inv == {1: 0, 3: 1}
        assert p1_inv == {0: 0}
        assert p2_inv == {0: 0}

    def test_merge_coordinates_2(self):
        d1 = DummyModel(["dx_0 - c * x_0"], uri="d1")
        d2 = DummyModel(["dx_0 + d * x_0"], uri="d2")

        assert d1.uri != d2.uri
        coords_1, params_1, subs_1 = _make_coords(d1)
        coords_2, params_2, subs_2 = _make_coords(d2)

        assert len(coords_1) == len(coords_2) == 2
        assert len(params_1) == len(params_2) == 1
        p1, = params_1
        p2, = params_2

        assert p1.value != p2.value

        args = (coords_1, params_1), (coords_2, params_2)

        (coords, params), inverses = merge_coordinates(*args)

        (c1_inv, p1_inv), (c2_inv, p2_inv) = inverses
        # new - to - old mappings
        assert len(coords) == 4
        assert len(params) == 2
        assert c1_inv == {0: 0, 2: 1}
        assert c2_inv == {1: 0, 3: 1}
        assert p1_inv == {0: 0}
        assert p2_inv == {1: 0}

    def test_merge_coordinates_identical_param(self):
        p = Parameter('mu')
        d1 = DummyModel(["dx_0 - c * x_0"], uri="d1", params={'c': p})
        d2 = DummyModel(["dx_0 + d * x_0"], uri="d2", params={'d': p})
        coords_1, params_1, subs_1 = _make_coords(d1)
        coords_2, params_2, subs_2 = _make_coords(d2)

        assert len(coords_1) == len(coords_2) == 2
        assert len(params_1) == len(params_2) == 1
        p1, = params_1
        p2, = params_2

        assert p1 == p2

        args = (coords_1, params_1), (coords_2, params_2)

        (coords, params), inverses = merge_coordinates(*args)

        (c1_inv, p1_inv), (c2_inv, p2_inv) = inverses
        # new - to - old mappings
        assert len(coords) == 4
        assert len(params) == 1
        assert c1_inv == {0: 0, 2: 1}
        assert c2_inv == {1: 0, 3: 1}
        assert p1_inv == {0: 0}
        assert p2_inv == {0: 0}



    def test_merge_sytem(self):
        d1 = DummyModel(["dx_0 - c * x_0"], uri="d1")
        d2 = DummyModel(["dx_0 + d * x_0"], uri="d2")

        s1 = generate_system_from_atomic(d1)
        s2 = generate_system_from_atomic(d2)
        system, maps = merge_systems(s1, s2)

        assert len(system.X) == 4
        assert len(system.P) == 2


class TestParseRelation:
    def test_basic(self):
        # test 1

        eqn = 'e-R*f'
        X = [Effort('e'), Flow('f')]
        namespace = {str(x):x for x in X}
        with pytest.raises(SymbolicException):
            parse_relation(eqn, X, [], namespace)
        R = Parameter('R')
        P = [R]
        namespace.update({'R':R})
        L, M, J = parse_relation(eqn, X, P, namespace)

        assert L == {0:1, 1:-R}
        assert M == {}
        assert J == []

    def test_extended_array(self):

        eqn = "f = dx"
        X = sympy.symbols('dx,e,f,x')

        L, M, J = parse_relation(eqn, X, [], {str(x): x for x in X})

        assert L == {0:-1,2:1}
        assert M == {}
        assert J == []

    def test_nonlinear_function(self):
        eqn = "f - I_s*exp(e/V_t) "
        X = sympy.symbols('e,f')
        Is = Parameter('I_s')
        V_t = Parameter('V_t')
        P = [Is, V_t]
        namespace = {str(x): x for x in X}
        namespace.update({str(x): x for x in P})

        L, M, J = parse_relation(eqn, X, P, namespace)

        assert L == {1:1}
        assert M == {0:-Is}
        assert J == [exp(X[0]/V_t)]

    def test_nonlinear_functions(self):

        eqn = "f_1 = k*exp(e_1) - k*exp(e_2)"
        X = sympy.symbols('e_1,f_1, e_2,f_2')
        k = Parameter('k')
        P = [k]
        namespace = {str(x): x for x in X}
        namespace.update({str(x): x for x in P})

        L, M, J = parse_relation(eqn, X, [k], namespace)

        assert L == {1:  1}
        assert M == {0: k, 1: -k}
        assert J == [sympy.exp(X[2]), sympy.exp(X[0])]

    def test_constant_function(self):

        eqn = "x_0 - 1"
        X = [Variable(index=0)]
        L, M, J = parse_relation(eqn, X, [], {'x_0': X[0]})
        assert L == {0: 1}
        assert M == {0: -1}
        assert J == [1]

    def test_nonlinear_parameter(self):
        eqn = "e_0 - exp(mu)*f_0"
        p = Parameter('mu')
        P = [p]
        X = [Effort(index=0), Flow(index=0)]
        namespace = {str(x): x for x in X}
        namespace.update({str(x): x for x in P})

        L, M, J = parse_relation(eqn, X, P, namespace)

        assert L == {
            0: 1,
            1: -sympy.exp(p)
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
        X = [x_0, e_0]
        P = [mu, R,T, V]

        namespace = {str(x): x for x in X}
        namespace.update({str(x): x for x in P})
        L, M, J = parse_relation(eqn, X, P, namespace)

        assert L == {1: 1}
        assert M == {0: -mu, 1: -R*T}
        assert J == [sympy.S(1), sympy.log(x_0/V)]


class TestGenerateSystem:
    """These will fail as long as model.constitutive_relation spits out the
     wrong symbols"""

    def test_r(self):
        model = new("R", value=10)

        # X -> local coordinates
        # P -> Parameters
        # L -> Linear Part of the matrix
        # M -> Matrix for nonlinear terms
        # JX - > nonlinear terms

        X, P, L , M , JX = generate_system_from_atomic(model)
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

        X, P, L, M, JX = generate_system_from_atomic(model)
        assert len(X) == 4
        assert len(P) == 1
        assert not M
        assert not JX

        assert L.rows == 2

        for row in as_dict(L).values():
            assert row in ({0: 1, 2: -1}, {1: -C, 3: 1})

        names = [str(x) for x in X]
        assert names == ["dx_0", "e_0", "f_0", "x_0"]

    def test_se(self):
        se = new("Se")
        X, P, L, M, JX = generate_system_from_atomic(se)

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
        assert maps == [
            ({0: 0, 1: 1, 2: 2, 5: 3}, {0: 0}), \
            ({3:0, 4:1}, {1: 0})
        ]

    def test_common_param(self):

        p = Parameter('C')
        C = new("C", value=p)
        c_1, p_1, subs_1 = _make_coords(C)
        C2 = new("C", value=p)
        c_2, p_2, subs_1 = _make_coords(C2)

        assert len(c_1) == len(c_2) == 4
        assert len(p_1) == len(p_2) == 1
        (c, p), maps = merge_coordinates(
            (c_1, p_1), (c_2, p_2)
        )

        assert len(p) == 1

    def test_merge_systems(self):
        p1 = Parameter('C')
        p2 = Parameter('R')
        c = new("C", value=p1)
        r = new("R", value=p2)
        system_1 = generate_system_from_atomic(c)
        system_2 = generate_system_from_atomic(r)

        system, maps = merge_systems(system_1, system_2)

        assert str(system.X) == "[dx_0, e_0, f_0, e_1, f_1, x_0]"

        assert len(system.P) == 2
        assert not system.M
        assert not system.J
        assert as_dict(system.L) == {
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
        system_1 = generate_system_from_atomic(Ce)
        system_2 = generate_system_from_atomic(Re)

        system, maps = merge_systems(system_1, system_2)

        assert str(system.X) == '[dx_0, e_0, f_0, e_1, f_1, e_2, f_2, x_0, u_0]'
        assert system.P == [P]
        assert as_dict(system.L) == {
            0: {1: 1},
            1: {0: -1, 2: 1},
            2: {4: 1, 6: 1},
            3: {4: 1}
        }
        assert as_dict(system.M) == {
            0: {0: -P},
            3: {1: 1, 2: -1}
        }
        u = system.X[-1]
        x = system.X[-2]
        e_2 = system.X[3]
        e_3 = system.X[5]
        assert system.J == [sympy.log(x), u*sympy.exp(e_3 / P), u*sympy.exp(e_2 / P)]


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

        coords, params, L, M, J = model.system_model

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


def _assert_in(eqn, iterable):

    # todo: make this less hacky
    for test_eq in iterable:
        if str(eqn) == str(test_eq):
            return True
    assert False, f"{eqn} not in {iterable}"


# def are_equal(eq1, eq2):
#     if eq1.__class__ != eq2.__class__:
#         return False
#     sympy.Mul
#
#     if len(eq1.args) == len(eq2.args) and len(eq1.args) > 1 and not eq1.is_comutative:
#         for a1, a2 in zip(eq1.args, eq2.args):
#             if not are_equal(a1, a2):
#                 return False
#         return True
#
#     args1 = list(eq1.args)
#     args2 = list(eq2.args)
#
#     while (args1 or args2):
#         a1 = args1.pop()
#         target = None
#         for a in args2:
#             if are_equal(a1, a):
#                 target = a
#         if not target:
#             return False
#         else:
#             args2.remove(target)
#
#     if args1 or args2:
#         return True
#
#     return False
#

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





