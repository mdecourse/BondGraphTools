from BondGraphTools import new, connect, expose
import sympy

def test_build_relations():
    c = new("C")
    eqns = c._build_relations()

    test_eqn = {sympy.sympify("x_0 - C*e_0"),
                sympy.sympify("dx_0 - f_0")}

    assert set(eqns) == test_eqn


def test_zero_junction_relations():
    r = new("R", value=sympy.symbols('r'))
    l = new("I", value=sympy.symbols('l'))
    c = new("C", value=sympy.symbols('c'))
    kvl = new("0", name="kvl")

    rlc = new()
    rlc.add([c, l, kvl, r])

    connect(r, kvl)
    connect(l, kvl)
    connect(c, kvl)

    rels = kvl.constitutive_relations

    assert sympy.sympify("e_1 - e_2") in rels
    assert sympy.sympify("e_0 - e_2") in rels
    assert sympy.sympify("f_0 + f_1 + f_2") in rels




def test_build_model_fixed_cap():
    c = new("C", value=0.001)

    eqns = c.constitutive_relations
    assert len(eqns) == 2

    test_eqn1 = sympy.sympify("x_0 - 0.001*e_0")
    test_eqn2 = sympy.sympify("dx_0-f_0")

    assert test_eqn1 in eqns
    assert test_eqn2 in eqns


def test_cv_relations():
    c = new("C", value=1)
    se = new("Se")
    r = new("R", value=1)
    kcl = new("1")
    bg = new()
    bg.add([c, se, kcl, r])

    connect(c, (kcl,kcl.non_inverting))
    connect(r, (kcl, kcl.non_inverting))
    connect(se, (kcl, kcl.non_inverting))
    assert bg.constitutive_relations == [sympy.sympify("dx_0 + u_0 + x_0")]


def test_parallel_crv_relations():
    c = new("C", value=1)
    se = new("Se")
    r = new("R", value=1)
    kcl = new("0")
    bg = new()
    bg.add([c, se, kcl, r])

    connect(c, kcl)
    connect(se, kcl)
    connect(r, kcl)

    assert bg.constitutive_relations == [sympy.sympify("dx_0 - du_0"),
                                         sympy.sympify("x_0 - u_0")]

def test_ported_series_resistor():

    Se = new("Se")
    r1 = new("R", value=1)
    r2 = new("R", value=2)
    kvl = new('1')
    ss = new("SS")
    model = new()
    model.add(
        Se,r1,r2,kvl, ss
    )
    expose(ss)
    connect(Se, kvl.non_inverting)
    connect(kvl.inverting, r1)
    connect(kvl.inverting, r2)
    connect(kvl.inverting, ss)

    assert len(model.ports) == 1

    assert model.constitutive_relations == [
        sympy.sympify("e_0 - 3*f_0 - u_0")
    ]

def test_ported_cap():
    model = new()
    c = new("C", value=3)
    zero = new("0")
    ss = new("SS")
    model.add(
        c, zero, ss
    )

    connect(c, zero)
    connect(ss, zero)

    expose(ss)
    assert len(model.ports) == 1


    assert model.constitutive_relations == [
        sympy.sympify("dx_0 - f_0"),sympy.sympify("e_0 - x_0/3")
    ]
def test_ported_parallel_rc():

    model = new()
    r = new("R", value=2)
    c = new("C", value=3)
    zero = new("0")
    ss = new("SS")
    model.add(
        r,c,zero, ss
    )

    connect(r,zero)
    connect(c,zero)
    connect(ss, zero)

    expose(ss)
    assert len(model.ports) == 1

    assert model.constitutive_relations == [
        sympy.sympify("dx_0 + x_0/6 - f_0"),
        sympy.sympify("e_0 - x_0/3")
    ]


class TestConstitutiveRelations:
    def test_empty_failstate(self):
        from BondGraphTools import new

        model = new()

        assert model.constitutive_relations == []


# def test_ported_cr():
#     model = bgt.new()
#     Sf = bgt.new('Sf', name="Sf")
#     R = bgt.new("R", value=2)
#     zero = bgt.new("0")
#     ss = bgt.new("SS")
#
#     model.add(Sf, R, zero, ss)
#     connect(Sf, zero)
#     connect(R, zero)
#     connect(ss, zero)
#
#     bgt.expose(ss, 'A')
#     assert len(model.control_vars) == 1
#
#     ts, ps, cs = model._build_internal_basis_vectors()
#     assert len(cs) == 1
#     assert len(ps) == 7
#     assert len(ts) == 0
#
#     mapping, coords = inverse_coord_maps(ts, ps, cs)
#     assert len(coords) == 15
#
#     coords, mappings, lin_op, nl_op, conttr = model.system_model()
#     assert nl_op.is_zero
#     assert not conttr
#
#     assert model.constitutive_relations == [
#         sympy.sympify('e_0 - 2*f_0 - 2*u_0')
#     ]



#
# def test_generate_subs():
#
#     w, x, y, z = sympy.sympify("w,x,y,z")
#     size_tuple  =(0, 2, 0,4 )
#     coords = [w, x, y, z]
#     #  w + w^2 + x^2
#     #  x + 1 + y^2   < should appear in subs as x = -y^2  - 1
#     #  y + 1         <                          y = - 1
#     #  0 + z^2 + w^2
#
#     L = sympy.SparseMatrix(4, 4, {(0,0): 1,
#                                   (1,1): 1,
#                                   (2,2): 1})
#
#     N = sympy.SparseMatrix(4, 1, {(0, 0): w**2 + x**2,
#                                   (1, 0): 1 + y**2,
#                                   (2, 0): 1})
#
#     constraint = [z**2 + w**2]
#     subs = _generate_substitutions(L, N,constraint, coords, size_tuple)
#     target_subs = [(y,-1), (x, -1-y**2)]
#
#     assert subs == target_subs
#
#
# def test_cv_subs_func():
#     c = bgt.new("C", value=1)
#     se = bgt.new("Se")
#     r = bgt.new("R", value=1)
#     kcl = bgt.new("1")
#     bg = bgt.new()
#     bg.add([c, se, kcl, r])
#
#     connect(c,(kcl,kcl.non_inverting))
#     connect(r, (kcl, kcl.non_inverting))
#     connect(se, (kcl, kcl.non_inverting))
#
#     cv_s = {'u_0': ' -exp(-t)'}
#
#     subs = [(sympy.Symbol('u_0'), sympy.sympify('-exp(-t)'))]
#
#     mappings, coords = inverse_coord_maps(*bg.basis_vectors)
#     assert _generate_cv_substitutions(cv_s, mappings,coords) == subs
#
#
# def test_cv_subs_const():
#     c = bgt.new("C", value=1)
#     se = bgt.new("Se")
#     r = bgt.new("R", value=1)
#     kcl = bgt.new("1")
#     bg = bgt.new()
#     bg.add([c, se, kcl, r])
#
#     connect(c,(kcl,kcl.non_inverting))
#     connect(r, (kcl, kcl.non_inverting))
#     connect(se, (kcl, kcl.non_inverting))
#
#     cv_s = {'u_0': ' 2'}
#
#     subs = [(sympy.Symbol('u_0'), sympy.S(2))]
#
#     mappings, coords = inverse_coord_maps(*bg.basis_vectors)
#     assert _generate_cv_substitutions(cv_s, mappings,coords) == subs

#
# def test_cv_subs_state_func():
#     c = bgt.new("C", value=1)
#     se = bgt.new("Se")
#     r = bgt.new("R", value=1)
#     kcl = bgt.new("1")
#     bg = bgt.new()
#     bg.add([c, se, kcl, r])
#
#     connect(c,(kcl,kcl.non_inverting))
#     connect(r, (kcl, kcl.non_inverting))
#     connect(se, (kcl, kcl.non_inverting))
#
#     cv_s = {'u_0': ' -exp(-x_0)'}
#
#     subs = [(sympy.Symbol('u_0'), sympy.sympify('-exp(-x_0)'))]
#
#     mappings, coords = inverse_coord_maps(*bg.basis_vectors)
#     assert _generate_cv_substitutions(cv_s, mappings,coords) == subs
