from BondGraphTools import new, connect, expose
from BondGraphTools.model_reduction import reduce
from BondGraphTools.model_reduction.symbols import *
import sympy
from .helpers import *


class Test_atomic_relations:
    def test_C(self):
        c = new("C")
        eqns = c.equations
        test_eqn = {"e_0 - x_0/C", "dx_0 - f_0"}
        assert len(eqns) == 2
        assert set(eqns) == {"x_0 - C * e_0", "dx_0 - f_0"}

        assert sym_set_eq(c.constitutive_relations, test_eqn)

    def test_R(self):
        R = new("R")

        eqns = R.equations

        test_eqn = {"f_0 - e_0/r"}

        assert set(eqns) == {"e_0 - f_0*r"}
        assert sym_set_eq(R.constitutive_relations, test_eqn)

    def test_TF(self):

        r = Parameter('r')
        TF = new("TF", value=r)

        test_eqns = {"f_0 + r * f_1", "e_1 - r * e_0"}

        sm = TF.system_model
        reduce(sm)

        assert [str(x) for x in sm.X] == ["e_0", "f_0", "e_1", "f_1"]
        assert sm.L.row_list() == [
            (0, 0, 1),
            (0, 2, -1/r),
            (1, 1, 1),
            (1, 3, r)
        ]

        assert set(TF.equations) == test_eqns
        assert sym_set_eq(TF.constitutive_relations,
                          {"f_0 + f_1*r", "e_0 - e_1/r"})

    def test_Se(self):
        se = new("Se")

        system = se.system_model

        assert system.L.row_list() == [
            (0, 1, 1),
            (0, 3, -1),
            (1, 0, 1),
            (1, 2, 1)
        ]

        reduce(system)
        assert isinstance(system.X[0], Output)
        assert isinstance(system.X[3], Control)
        assert str(system.X) == "[f, e_0, f_0, e]"

        assert system.L.row_list() == [
            (0, 0, 1),
            (0, 2, 1),
            (1, 1, 1),
            (1, 3, -1)
        ]


