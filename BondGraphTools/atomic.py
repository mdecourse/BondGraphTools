"""This module contains class definitions for atomic models; those which
cannot be decomposed into other components.
"""

import logging
import sympy as sp

from BondGraphTools.base import BondGraphBase
from BondGraphTools.exceptions import InvalidPortException, ModelException
from BondGraphTools.view import Glyph
from BondGraphTools.port_managers import PortManager, PortExpander
from BondGraphTools.model_reduction import generate_system_from_atomic

logger = logging.getLogger(__name__)

_symbolics = sp.Expr, sp.Symbol


def _is_symbolic_const(value):
    return isinstance(value, _symbolics)


class Atomic(BondGraphBase, PortManager):
    """Base class for models not able to be decomposed.

    See Also:
        BondGraphBase, PortManager
    """

    def __init__(self, metamodel, constitutive_relations,
                 state_vars=None, params=None, outputs=None, **kwargs):

        self._metamodel = metamodel
        ports = kwargs.pop("ports")
        super().__init__(**kwargs)
        PortManager.__init__(self, ports)
        self._state_vars = state_vars

        self._params = params
        self._output_vars = outputs

        self.view = Glyph(self)
        self._equations = constitutive_relations

    def __eq__(self, other):
        return self.__dict__ == self.__dict__

    @property
    def template(self):
        return f"{self.__library__}/{self.__component__}"

    @property
    def metamodel(self):
        return self._metamodel

    @property
    def output_vars(self):
        return list(self._output_vars.keys()) if self._output_vars else []

    @property
    def control_vars(self):
        """See `BondGraphBase`"""

        def is_const(value):
            if isinstance(value, (int, float, complex)):
                return True
            elif _is_symbolic_const(value):
                return True
            else:
                return False

        out = []

        for p, v in self.params.items():
            try:
                if is_const(v) or is_const(v["value"]):
                    continue
            except (KeyError, TypeError):
                pass

            out.append(p)
        return out

    @property
    def params(self):
        return self._params if self._params else {}

    def set_param(self, param, value):
        """
        Warning: Scheduled to be deprecated
        """
        if isinstance(self._params[param], dict):
            self._params[param]["value"] = value
        else:
            self._params[param] = value

    @property
    def state_vars(self):
        """See `BondGraphBase`"""
        return self._state_vars if self._state_vars else []

    @property
    def equations(self):
        rels = []
        for string in self._equations:

            iloc = 0
            iloc = string.find("_i", iloc)

            if iloc < 0:
                # just a plain old string; sympy can take care of it
                rels.append(string)

                continue

            sloc = string.rfind("sum(", 0, iloc)

            if sloc < 0:
                # we have a vector equation here.
                for port_id in self.ports:
                    if isinstance(port_id, int):
                        rels.append(string.replace("_i", "_{}".format(port_id)))
            else:

                tiers = 0

                next_open = string.find("(", sloc + 4)
                eloc = string.find(")", sloc + 4)
                while next_open > 0 or tiers > 0:
                    if next_open < eloc:
                        tiers += 1
                        next_open = string.find("(", next_open)
                    else:
                        tiers -= 1
                        eloc = string.find(")", eloc)

                if eloc < 0:
                    raise ValueError("Unbalanced brackets", string)

                substring = string[sloc + 4: eloc]
                terms = [substring.replace("_i", "_{}".format(p))
                         for p in self.ports if isinstance(p, int)]
                symstr = string[0:sloc] + "(" + " + ".join(terms) + string[
                                                                    eloc:]
                rels.append(symstr)

        return [r for r in rels if r != 0]

    def __hash__(self):
        return super().__hash__()

    @property
    def system_model(self):
        return generate_system_from_atomic(self)


class SymmetricAtomic(Atomic):
    """Fixed Multi-port components that are port agnostic.
    See Also:
          Atomic
    """
    def get_port(self, port=None):
        if not port and not isinstance(port, int):
            p = [port for port in self.ports if not port.is_connected]

            if not p:
                raise InvalidPortException("No free ports")
            return super().get_port(p[0])

        else:
            return super().get_port(port)


class EqualEffort(BondGraphBase, PortExpander):
    """Implements 0-junction."""

    def __init__(self, **kwargs):

        PortExpander .__init__(self, {None: None})
        BondGraphBase.__init__(self, **kwargs)
        self.view = Glyph(self)

    @property
    def template(self):
        return "base/0"

    @property
    def params(self):
        return []

    @property
    def basis_vectors(self):
        return {}, self._port_vectors(), {}

    @property
    def equations(self):
        vects = list(self._port_vectors())
        try:
            e_0, f_0 = vects.pop()
        except IndexError:
            raise ModelException("Model %s has no ports", self)
        partial_sum = [f_0]

        relations = []

        while vects:
            e_i, f_i = vects.pop()
            relations.append(f"{e_i} - {e_0}")
            partial_sum.append(f_i)

        relations.append(" + ".join(partial_sum))

        return relations

    @property
    def system_model(self):
        return generate_system_from_atomic(self)


class EqualFlow(BondGraphBase, PortExpander):
    """Implements the Equal Flow 1-junction

    Attributes:
         non_inverting: PortTemplate for the non-inverting port
         inverting: PortTemplate for the inverting port.

    Inverting ports correspond to those with 'outwards pointing' bonds.
    This is required as equal flow junctions are not invariant with respect to
    bond orientation.

    See Also:
        PortExpander, BondGraphBase
    """
    def __init__(self, **kwargs):
        PortExpander.__init__(self, {"non_inverting": {"weight": 1},
                                     "inverting": {"weight": -1}})
        BondGraphBase.__init__(self, **kwargs)
        self.view = Glyph(self)

    @property
    def non_inverting(self):
        t, = (tp for tp in self._templates if tp.index == "non_inverting")
        return t

    @property
    def inverting(self):
        t, = (tp for tp in self._templates if tp.index == "inverting")
        return t

    @property
    def template(self):
        return "base/1"

    @property
    def basis_vectors(self):
        return {},  self._port_vectors(), {}

    @property
    def params(self):
        return []

    def get_port(self, port=None):
        try:
            return super().get_port(port)
        except InvalidPortException as ex:
            if not port:
                raise InvalidPortException("You must specify a port")

    @property
    def equations(self):
        vects = list(self._port_vectors().items())
        try:
            e_0, f_0, port = vects.pop()
        except IndexError:
            raise ModelException("Model %s has no ports", self)

        sigma_0 = port.weight

        partial_sum = [f"{sigma_0} * {f_0}"]

        relations = []

        while vects:
            e_i, f_i, port = vects.pop()
            sigma_i = port.weight
            relations.append(f"{sigma_i} * {e_i} - {sigma_0} * {e_0}")

            partial_sum.append(f"{sigma_i} * {f_i}")

        relations.append(" + ".join(partial_sum))

        return relations

    @property
    def system_model(self):
        return generate_system_from_atomic(self)