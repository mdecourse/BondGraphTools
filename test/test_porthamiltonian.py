import BondGraphTools as bgt
from BondGraphTools.port_hamiltonian import PortHamiltonian
import sympy as sp
from .helpers import *


def test_hamiltonian():
    hamiltonian = "x^2/(2*C) + x^4/(4*D)"

    relations, state_vars, params, ports = PortHamiltonian._generate_relations(
        hamiltonian
    )

    assert state_vars == {"x_0": "x"}
    assert params == {"C": None, "D":None}
    assert ports == {0:None}
    assert relations == ["-e_0 + x_0**3/D + x_0/C", "dx_0 - f_0"]


def test_create_PH():
    hamiltonian = "x^2/(2*C) + x^4/(4*D)"
    ph = bgt.new(component="PH", value=hamiltonian)
    assert ph.params == {"C": None, "D":None}

    assert ph.state_vars == {"x_0": "x"}

    assert ph.equations == {"-e_0 + x_0**3/D + x_0/C", "dx_0 - f_0"}

    assert ph.hamiltonian == hamiltonian
    port_list = list(ph.ports)
    assert len(port_list) == 1
    assert port_list[0] == (ph, 0)


def test_create_PH_2():
    hamiltonian = "x^2/(2*C + 2*y)"
    build_args = {
        "hamiltonian": hamiltonian,
    }
    ph = bgt.new(component="PH", value=build_args)

    assert ph.params == {"C": None}

    assert set(ph.state_vars.values()) == {"x", "y"}
    assert sym_set_eq(ph.constitutive_relations,
                {"e_0 - x_0/(C + x_1)",
                 "dx_0 - f_0",
                 "e_1 + x_0**2/(2*(C + x_1)**2)",
                 "dx_1 - f_1"})

    assert ph.hamiltonian == hamiltonian

    port_list = list(ph.ports)
    assert len(port_list) == 2
    assert (ph, 0) in port_list
    assert (ph, 1) in port_list


def test_create_PH_parameters():
    hamiltonian = "x^2/(2*C) + x^4/(4*D)"
    p_1 = 1
    p_2 = sp.S("k")
    build_args = {
        "hamiltonian": hamiltonian,
        "params":{"C":p_1, "D":p_2}
    }

    ph = bgt.new(component="PH", value=build_args)
    assert ph.params == {"C": p_1, "D": p_2}

    assert ph.state_vars == {"x_0": "x"}
    assert sym_set_eq(ph.constitutive_relations,
                      {"e_0 - x_0**3/k - x_0", "dx_0 - f_0"})

    assert ph.hamiltonian == hamiltonian


def test_create_duffing_eqn():

    model = bgt.new(name="Duffing")
    junct = bgt.new("0")
    nlin_cap = bgt.new("PH", value="x^2/2 + x^4/4")
    inductor = bgt.new("I", value=1)
    source = bgt.new("Sf")
    bgt.add(model, nlin_cap, inductor, source, junct)
    bgt.connect(junct, nlin_cap)
    bgt.connect(junct, inductor)
    bgt.connect(source, junct)

    assert sym_set_eq(
        model.constitutive_relations,
        {"dx_0 + x_1 - u_0", "dx_1 - x_0**3 - x_0", "y_0 - x_0**3 - x_0"}
    )
