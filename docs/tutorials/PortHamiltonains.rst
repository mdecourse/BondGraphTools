Tutorial: Port Hamiltonians
===============================
+----------------+------------------------------------------------------------+
| Goal:          | Build and simulate a coupled oscillator system based on    |
|                | bond graph modelling techniques. Introduce object oriented |
|                | modelling with BondGraphTools.                             |
+----------------+------------------------------------------------------------+
| Difficulty:    | Intermediate.                                              |
+----------------+------------------------------------------------------------+
| Requirements:  | `BondGraphTools`, `jupyter`.                               |
+----------------+------------------------------------------------------------+
| How to follow: | Enter each block of code in consecutive cells in a jupyter |
|                | notebook.                                                  |
+----------------+------------------------------------------------------------+

Overview
--------
In this tutorial we will demonstrate basic use of port-Hamiltonians via an
example from quantum optomechanics (for more information see [PC2017]_ ).

One can think of the physical systems as consisting of a number of
beams (damped harmonic oscillators), which exchange energy with a
electromagentic field via a energy
conserving interaction hamiltonian:
$$H_\text{int} \propto x_j (x^2_\alpha + y^2_\alpha)$$.
Here $x_j, $ is the displacement of the $j$th oscillator, and
$x_\alpha, y_\alpha$ are the 'position' and 'momentum' of the
electromagnetic field to which the mechanical oscillators are coupled.


Part 1: Building the oscillator array
-------------------------------------

The first step is to build a class representing the linear oscillators.
Classes are a staple of object oriented modelling and, in this case,
represents a kind of prototype for a new model.::

    class Linear_Osc(bgt.BondGraph):
        damping_rate = 0.1

        def __init__(self, freq, index):

            # Create the components
            r = bgt.new("R", name="R", value=self.damping_rate)
            l = bgt.new("I", name="L", value=1/freq)
            c = bgt.new("C", name="C", value=1/freq)
            port = bgt.new("SS")
            conservation_law = bgt.new("1")

            # Create the composite model and add the components
            super().__init__(
                name=f"Osc_{index}",
                components=(r, l, c, port, conservation_law)
            )

            # Define energy bonds
            for component in (r, l, c):
                bgt.connect(conservation_law, component)

            bgt.connect(port, conservation_law)

            # Expose the external port
            bgt.expose(port, label="P_in")

An overview of what's going on:
1. `damping_rate` is a class-level

The `__init__` method is called when a new instance of this class is
created. A

We will also define a set of global parameters::

    # Damping rate relative to natural time scale
    gamma = 0.1

    # Average oscillator frequency
    osc_freq = 2

    # Individual frequency deviations
    osc_spread = [-0.3, -0.1, 0, 0.1, 0.3]

    # The rate of energy exchange between optical and mechanical domains
    coupling_rate = 0.1

    # The bare frequency of the optical cavity
    opt_freq = 10

Part 2: Building the coupling mechanism
---------------------------------------

In contrast to traditional Hamiltonian mechanics, port-Hamiltonian storage
functions (the hamiltonian henceforth) do not come embedded with a symplectic
structure. Instead, the hamiltonian simply describes how the state variables
contribute to energy storage and the modeller must use the interconnection
structure to define the appropriate geometry.

Hence, there are two significant parts to modelling port Hamiltonians in
`BondGraphTools`; firstly, the port Hamiltonian itself `PH` and what is known
as a 'symplectic gyrator'.

Hamiltonian energy can be instantiated with the `new` command for a set of
build arguments.
>>> ph = new("PH", value=build_args)
The build arguments is assumed to be a dictionary with at least the
key 'hamiltonian'. For example, if one wished to build the hamiltonian for a
classical harmonic oscillator with frequency $2 \text{rad/s}$, one would use
the arguments::

    build_args = {
        "hamiltonian": "w*(x^2 + y^2)/2,
        "params": {
            "w":2
        }
    }




Now, for the cavity::

    def coupled_cavity():
        model = new(name="Cavity Model")

        coupling_kwargs = {
            "hamiltonian":"(w + G*x_1)*(x_2^2 + x_3^2)/2",
            "params": {"G":1,"w":cav_freq}
        }

        port_hamiltonian = new("PH", value=coupling_kwargs)
        symplectic_gyrator = new("GY", value=-1)
        emf = new("1") #common position -> common flow
        dissipation = new("R", value=2)
        photon_source = new('SS')
        add(model, port_hamiltonian, symplectic_gyrator, emf, dissipation, photon_source)

        connect(emf, (port_hamiltonian, 1))
        connect(emf, (symplectic_gyrator, 1))
        connect((port_hamiltonian, 2), (symplectic_gyrator, 0))
        connect(emf, dissipation)
        connect(photon_source, emf)
        expose(photon_source)
        mean_field = new("0")
        add(model, mean_field)
        connect(mean_field, (port_hamiltonian, 0))
        osc_array = [linear_osc(f) for f in frequencies]

        for osc in osc_array:
            add(model, osc)
            connect(mean_field, (osc, "V_in"))

        return model


Part 3: Building and running the experimental apparatus
-------------------------------------------------------


.. [PC2017]: https://doi.org/10.14264/uql.2017.462