[![PyPI version](https://badge.fury.io/py/BondGraphTools.svg)](https://badge.fury.io/py/BondGraphTools)
[![Build Status](https://travis-ci.com/BondGraphTools/BondGraphTools.svg?branch=master)](https://travis-ci.com/BondGraphTools/BondGraphTools)
[![Test Coverage](https://api.codeclimate.com/v1/badges/4735c13a87b24d3a1899/test_coverage)](https://codeclimate.com/github/BondGraphTools/BondGraphTools/test_coverage)
# BondGraphTools - A Toolkit for modelling multi-physics systems.
## Summary

This toolkit is for rapid modelling and design of networked physical systems.
It is conceptually based upon the Bond Graph modelling methodology.

## Documentation

https://bondgraphtools.readthedocs.io/

## Installation

### Dependencies

BondGraph requires:
- python 3.6 or greater.
- julia 0.6.4

Python dependencies:
- sympy>=1.1.1
- numpy>=1.14
- scipy>=1.0.1
- matplotlib>=2.2.2
- julia==0.1.5
- diffeqpy==0.4
- ordered-set>=3.0.0

Julia dependencies:
 - PyCall
 - DifferentialEquations.jl

### Instructions
1. Install python > 3.6 for your operating system.
2. Install Julia 0.6.4 (https://julialang.org/downloads/) for your operating
 system. _Julia 0.7 and 1.0 are not yet supported_
3. Make sure Julia 0.6.4 is in your os path. (test this by running `julia -v`)
4. Install using PyPI; `pip install BondGraphTools`


## A Simple Example

Build and simulate a RLC driven RLC circuit::

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
    law = bgt.new("0") # Common voltage conservation law
    
    # Add them to the model
    model.add(resistor, inductor, capacitor, law)
    
    # Connect the components
    bgt.connect(law, resistor)
    bgt.connect(law, capacitor)
    bgt.connect(law, inductor)
    
    # produce equations of motion
    print(model.constitutive_relations)

    