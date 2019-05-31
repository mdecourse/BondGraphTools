from BondGraphTools.model_reduction.symbols import *
import sympy


def sym_eq(bgt_eqn, string):
    namespace = {
        str(atom): atom for atom in bgt_eqn.atoms()
        if isinstance(atom, BondGraphVariables)
    }

    test_equation = sympy.sympify(string, namespace)

    return test_equation == bgt_eqn


def sym_in(bgt_eqn, stringset):
    namespace = {
        str(atom): atom for atom in bgt_eqn.atoms()
        if isinstance(atom, BondGraphVariables)
    }

    test_set = {sympy.sympify(string, namespace) for string in stringset}

    return bgt_eqn in test_set


def sym_set_eq(bgt_set, stringset):

    namespace = {}
    for bgt_eqn in bgt_set:
        namespace.update({
            str(atom): atom for atom in bgt_eqn.atoms()
            if isinstance(atom, BondGraphVariables)}
        )

    base_set = {eq for eq in bgt_set}

    test_set = {sympy.sympify(string, namespace) for string in stringset}

    return base_set == test_set
