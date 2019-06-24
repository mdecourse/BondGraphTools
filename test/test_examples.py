import BondGraphTools as bgt


def test_optomechanical_example():
    class Linear_Osc(bgt.BondGraph):
        damping_rate = 0.1

        def __init__(self, freq, index):
            # Create the components
            r = bgt.new("R", name="R", value=self.damping_rate)
            l = bgt.new("I", name="L", value=1 / freq)
            c = bgt.new("C", name="C", value=1 / freq)
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


    def coupled_cavity():
        model = bgt.new(name="Cavity Model")

        # Define the interaction Hamiltonain
        coupling_args = {
            "hamiltonian": "(w + G*x_0)*(x_1^2 + x_2^2)/2",
            "params": {
                "G": 1,  # Coupling constant
                "w": 6  # Cavity resonant freq.
            }
        }
        port_hamiltonian = bgt.new("PH", value=coupling_args)

        # Define the symplectic junction structure
        symplectic_gyrator = bgt.new("GY", value=1)
        em_field = bgt.new("1")
        bgt.add(model, port_hamiltonian,
                symplectic_gyrator,
                em_field)
        bgt.connect(em_field, (port_hamiltonian, 1))
        bgt.connect(em_field, (symplectic_gyrator, 1))

        bgt.connect(
            (port_hamiltonian, 2), (symplectic_gyrator, 0)
        )

        # Construct the open part of the system
        dissipation = bgt.new("R", value=1)
        photon_source = bgt.new('SS')
        bgt.add(model, dissipation, photon_source)
        bgt.connect(em_field, dissipation)
        bgt.connect(photon_source, em_field)
        bgt.expose(photon_source)

        # Build the oscillator array
        frequencies = [2 + f for f in (-0.3, -0.1, 0, 0.1, 0.3)]
        osc_mean_field = bgt.new("0")
        bgt.add(model, osc_mean_field)
        bgt.connect(osc_mean_field, (port_hamiltonian, 0))
        osc_array = [Linear_Osc(freq, index)
                     for index, freq in enumerate(frequencies)]

        for osc in osc_array:
            bgt.add(model, osc)
            bgt.connect(osc_mean_field, (osc, "P_in"))

        return model

    cavity = coupled_cavity()

    results =[
        "dx_0 + 23*x_11/10 + 17*x_3/10 + 19*x_5/10 + 2*x_7 + 21*x_9/10",
        "dx_1 - x_0*x_2 - 6*x_2",
        "dx_2 - e_0 + x_0*x_1 + x_0*x_2 + 6*x_1 + 6*x_2",
        "dx_3 - x_1**2/2 - x_2**2/2 + 17*x_3/100 + 17*x_4/10",
        "dx_4 - 17*x_3/10",
        "dx_5 - x_1**2/2 - x_2**2/2 + 19*x_5/100 + 19*x_6/10",
        "dx_6 - 19*x_5/10",
        "dx_7 - x_1**2/2 - x_2**2/2 + x_7/5 + 2*x_8",
        "dx_8 - 2*x_7",
        "dx_9 - x_1**2/2 + 21*x_10/10 - x_2**2/2 + 21*x_9/100",
        "dx_10 - 21*x_9/10",
        "dx_11 - x_1**2/2 + 23*x_11/100 + 23*x_12/10 - x_2**2/2",
        "dx_12 - 23*x_11/10",
        "f_0 - x_0*x_2 - 6*x_2"]

    for test_res, relation in zip(results, cavity.constitutive_relations):
        assert test_res == str(relation)


def test_readme_example():
    import BondGraphTools as bgt

    # Create a new model
    model = bgt.new(name="RLC")

    # Create components
    # 1 Ohm Resistor
    resistor = bgt.new("R", name="R1", value=1.0)

    # 1 Henry Inductor
    inductor = bgt.new("I", name="L1", value=1.0)

    # 1 Farad Capacitor
    capacitor = bgt.new("C", name="C1", value=1.0)

    # Conservation Law
    law = bgt.new("0")  # Common voltage conservation law

    # Add them to the model
    model.add(resistor, inductor, capacitor, law)

    # Connect the components
    bgt.connect(law, resistor)
    bgt.connect(law, capacitor)
    bgt.connect(law, inductor)

    results = ["dx_0 - x_1", "dx_1 + x_0 + x_1"]

    for truth, test_result in zip(results, model.constitutive_relations):
        assert str(test_result) == truth
