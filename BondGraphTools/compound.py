"""Class definition and helper functions for BondGraph model

"""
import logging

from ordered_set import OrderedSet
import sympy as sp

from BondGraphTools.base import BondGraphBase, Bond
from BondGraphTools.port_managers import LabeledPortManager
from BondGraphTools.exceptions import *
from BondGraphTools.view import GraphLayout
from BondGraphTools.model_reduction import (
    merge_systems, merge_bonds, reduce)


logger = logging.getLogger(__name__)

__all__ = [
    "Composite"
]


class Composite(BondGraphBase, LabeledPortManager):
    """Representation of a bond graph model.
    """
    def __init__(self, name, components=None, **kwargs):

        BondGraphBase.__init__(self, name, **kwargs)
        LabeledPortManager.__init__(self)
        self.components = OrderedSet()
        """The components, instances of :obj:`BondGraphBase`, 
        that make up this model"""

        if components:
            for component in components:
                self.add(component)

        self._bonds = BondSet()

        self.view = GraphLayout(self)
        """Graphical Layout of internal components"""

        self._port_map = dict()
        self._model_changed = True

    @property
    def template(self):
        return None

    @property
    def bonds(self):
        """The list of connections between internal components"""
        return list(self._bonds)

    def __truediv__(self, other):
        """See Also: `BondGraph.uri`"""
        try:
            try:
                c_type, name = other.split(":")
            except ValueError:
                c_type = None
                name = other

            name = name.strip(" ")
            test_uri = f"{self.uri}/{name}"
            c, = (c for c in self.components if c.uri == test_uri
                  and (not c_type or c_type == c.metamodel)
                  )
            return c
        except TypeError:
            raise ValueError(f"Cannot find {other}")
        except ValueError:
            raise ValueError(f"Cannot find a unique {other}")

    @property
    def metamodel(self):
        return "BG"

    @bonds.setter
    def bonds(self, arg):
        raise AttributeError("Use add/remove functions.")

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if self.__dict__ != other.__dict__:
            return False

        for c1, c2 in zip(self.components,
                          other.componets):
            if c1 != c2:
                return False

        return True

    @property
    def internal_ports(self):
        """A list of the ports internal to this model"""
        return [p for c in self.components for p in c.ports]

    def map_port(self, label, ef):
        """Exposes a pair of effort and flow variables as an external port
        Args:
            label: The label to assign to this port.
            ef: The internal effort and flow variables.
        """
        try:
            port = self.get_port(label)
        except InvalidPortException:
            port = self.new_port(label)

        self._port_map[port] = ef

    def add(self, *args):
        # Warning: Scheduled to be deprecated
        def validate(component):
            if not isinstance(component, BondGraphBase):
                raise InvalidComponentException("Invalid component class")
            if component is self:
                raise InvalidComponentException("Cannot add a model to itself")
            elif component.root is self.root:
                raise InvalidComponentException(
                    "Component already exists in model")

        work_list = []
        for arg in args:
            if isinstance(arg, BondGraphBase):
                validate(arg)
                work_list.append(arg)
            elif isinstance(arg, list):
                for item in arg:
                    validate(item)
                    work_list.append(item)
            else:
                raise InvalidComponentException(f"Invalid Component: {arg}")

        for item in work_list:
            item.parent = self
            self.components.add(item)

    def remove(self, component):
        # Warning: Scheduled to be deprecated
        if [b for b in self._bonds if b.head.component is component or
                b.tail.component is component]:
            raise InvalidComponentException("Component is still connected")
        if component not in self.components:
            raise InvalidComponentException("Component not found")

        component.parent = None
        self.components.remove(component)

    def set_param(self, param, value):
        # Warning: Scheduled to be deprecated
        c, p = self.params[param]
        c.set_param(p, value)

    @property
    def params(self):
        """
        A dictionary of parameters for this model in the form::
            i: (component, param_name)
        """
        j = 0
        out = dict()

        excluded = {
            v for pair in self._port_map.values() for v in pair
        }

        for v in self.components:
            try:
                params = v.params
            except AttributeError:
                continue
            for p in params:
                param = (v, p)
                if param not in excluded:
                    out.update({j: param})
                    j += 1
        return out

    @property
    def state_vars(self):
        """
        A `dict` of all state variables in the form::

             {
                "x_0": (component, state_var)
             }

        Where `"x_0"` is the model state variable, and `state_var` is the
        corresponding state variable of `component`
        """
        j = 0
        out = dict()
        for v in self.components:
            try:
                x_local = v.state_vars
            except AttributeError:
                continue

            for i in x_local:
                out.update({f"x_{j}": (v, i)})
                j += 1

        return out

    @property
    def system_model(self):
        if not self.components:
            raise ModelException("Model has no components")

        systems = {component: component.system_model
                   for component in self.components}



        system, maps = merge_systems(*systems.values())

        inverse_maps = {component: (subsystem.X, coord_map)
                        for (component, subsystem),(coord_map, _)
                        in zip(systems.items(), maps)}

        merge_bonds(system, self.bonds, inverse_maps)
        reduce(system)

        return system

    @property
    def equations(self):
        return [str(r) for r in self.constitutive_relations]

    @property
    def control_vars(self):
        """
        A `dict` of all control variables in the form::

            {
                "u_0": (component, control_var)
            }

        """
        j = 0
        out = dict()
        excluded = {
            v for pair in self._port_map.values() for v in pair
         }

        for v in self.components:
            try:
                for i in v.control_vars:
                    cv = (v, i)
                    if cv not in excluded:
                        out.update({f"u_{j}": cv})
                        j += 1
            except AttributeError:
                pass
        return out


def _is_label_invalid(label):
    if not isinstance(label, str):
        return True

    for token in [" ", ".", "/"]:
        if len(label.split(token)) > 1:
            return True

    return False


class BondSet(OrderedSet):
    """
    Container class for internal bonds.
    """
    def add(self, bond):
        tail = bond.tail
        head = bond.head
        super().add(bond)
        head.is_connected = True
        tail.is_connected = True

    def remove(self, bond):
        tail = bond.tail
        head = bond.head
        if bond in self:
            super().remove(bond)
        else:
            super().remove(Bond(head, tail))
        head.is_connected = False
        tail.is_connected = False

    def __contains__(self, item):
        if isinstance(item, BondGraphBase):
            return any({item in head or item in tail for tail, head in self})
        else:
            return super().__contains__(item)
