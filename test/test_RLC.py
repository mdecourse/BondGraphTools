import pytest
import sympy
import BondGraphTools as bgt
import BondGraphTools.sim_tools as sim
from .helpers import *


@pytest.mark.use_fixture("rlc")
def test_build(rlc):
    assert len(rlc.state_vars) == 2
    assert len(rlc.ports) == 0


def test_build_rlc():
    r = bgt.new("R", value=1)
    l = bgt.new("I", value=1)
    c = bgt.new("C", value=1)
    kvl = bgt.new("0", name="kvl")
    rlc = bgt.new()
    rlc.add([r, l, c, kvl])

    bgt.connect(r, kvl)
    bgt.connect(l, kvl)
    bgt.connect(c, kvl)
    assert len(kvl.ports) == 3


@pytest.mark.use_fixture("rlc")
def test_build_and_drive(rlc):
    se = bgt.new("Se")
    assert len(se.control_vars) == 1
    rlc.add(se)

    for comp in rlc.components:
        if comp.metamodel == "0":
            bgt.connect(se, comp)
            break

    assert len(rlc.bonds) == 4
    assert len(rlc.control_vars) == 1


@pytest.mark.use_fixture("rlc")
def test_rlc_con_rel(rlc):

    eq = rlc.equations

    assert len(eq) == 2
    assert 'dx_0 - x_1' in eq
    assert 'dx_1 + x_0 + x_1' in eq


def test_se():

    Se = bgt.new('Se', value=1)
    c = bgt.new('C', value=1)
    vc = bgt.new()
    vc.add(Se, c)
    bgt.connect(Se, c)

    assert not Se.control_vars
    assert len(Se.output_vars) == 1

    assert vc.output_vars == {"y_0": (Se, 'f')}
    system = vc.system_model
    assert isinstance(system.X[0], Output)
    assert sym_set_eq(vc.constitutive_relations, {"x_0 - 1"})

@pytest.mark.skip
def test_one():
    loop_law = bgt.new('1')
    Se = bgt.new('Se', value=1)
    c = bgt.new('C', value=1)
    r = bgt.new('R', value=1)
    vc = bgt.new()
    vc.add([Se, c, loop_law, r])

    bgt.connect(Se, (loop_law, loop_law.non_inverting))
    bgt.connect(c, (loop_law, loop_law.inverting))
    bgt.connect(r, (loop_law, loop_law.inverting))

    assert sym_set_eq(vc.constitutive_relations, {"dx_0 + x_0 - 1"})


