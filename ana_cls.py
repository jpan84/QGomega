#Joshua Pan Jun 2025
#Classes for diagnostic analytical eddy flux model (EHF only)

import numpy as np

class Bous_plane_consts:
   def __init__(self, p0=1e5, N2=2e-4, T0=280, lat0=np.deg2rad(45), fplane=False):
      self.g, self.a, self.Rd, self.cp, self.OM = 9.81, 6.371e6, 287, 1004, 7.29e-5
      self.p0, self.N2, self.T0 = p0, N2, T0
      self.kap = self.Rd / self.cp
      self.lat0, self.f0 = lat0, 2 * self.OM * np.sin(lat0)
      self.beta = 0. if fplane else 2 * self.OM / self.a * np.cos(lat0)
      self.dens0 = p0 / self.Rd / T0

class Anafld:
   def __init__(self, name, consts, xfunc, yfunc, pfunc):
      self.name = name
      self.consts = consts
      self.funcs = dict(x=xfunc, y=yfunc, p=pfunc)

   def eval(self, xco, yco, pco):
      return self.consts * self.funcs['x'].eval(xco) *\
             self.funcs['y'].eval(yco) * self.funcs['p'].eval(pco)

   def deriv(self, dim):
      newfuncs = self.funcs.copy()
      newfuncs[dim] = self.funcs[dim].deriv
      return Anafld(self.name + '_deriv_' + dim, self.consts,\
             newfuncs['x'], newfuncs['y'], newfuncs['p'])

   @staticmethod
   def product(afld1, afld2):
      newname = 'prod_[' + afld1.name + ']_[' + afld2.name + ']'
      newconsts = afld1.consts * afld2.consts
      newfuncs = {k: Anafunc.product_real(afld1.funcs[k], afld2.funcs[k]) for k in afld1.funcs}
      return Anafld(newname, newconsts, newfuncs['x'], newfuncs['y'], newfuncs['p'])

   def __rmul__(self, sclr):
      newconsts = sclr * self.consts
      newname = f'({sclr} * {self.name})'
      return Anafld(newname, newconsts, self.funcs['x'], self.funcs['y'], self.funcs['p'])
 
   def __mul__(self, sclr):
      return self.__rmul__(sclr)

class Anafunc:
   def __init__(self, func, deriv=None):
      self.func = func
      self.deriv = deriv

   def eval(self, coords):
      return self.func(coords)

   def eval_deriv(self, coords):
      if self.deriv is None:
         raise AttributeError("No derivative function defined for this Anafunc instance.")
      return self.deriv.eval(coords)

   @staticmethod
   def sum(af1, af2):
      sumfunc = lambda coords: af1.eval(coords) + af2.eval(coords)
      sumderiv = None
      if not af1.deriv is None and not af2.deriv is None:
         sumderiv = Anafunc(lambda coords: af1.eval_deriv(coords) + af2.eval_deriv(coords))
      return Anafunc(sumfunc, sumderiv)

   @staticmethod
   def product_real(af1, af2):
      prodfunc = lambda coords: af1.eval(coords).real * af2.eval(coords).real
      prodderiv = None
      if not af1.deriv is None and not af2.deriv is None:
         prodderiv = Anafunc(lambda coords: af1.eval_deriv(coords).real * af2.eval(coords).real\
                          + af1.eval(coords).real * af2.eval_deriv(coords).real)
      return Anafunc(prodfunc, prodderiv)

ysc = 1e6
g1 = lambda y: np.exp(-(y / ysc)**2)
g1sq = lambda y: g1(y)**2
g1_d = lambda y: -2 * y / ysc**2 * g1(y)
g1sq_d = lambda y: 2 * g1_d(y)
g1_dd = lambda y: -2 / ysc**2 * (g1(y) + y * g1_d(y))
g1sq_dd = lambda y: -4 / ysc**2 *\
            (g1sq(y) + 2 * y * g1(y) * g1_d(y))

class my_yfuncs:
   def __init__(self):
      self.g1_ddd = Anafunc(my_yfuncs.diff3_g1)
      self.g1_dd = Anafunc(g1_dd, self.g1_ddd)
      self.g1_d = Anafunc(g1_d, self.g1_dd)
      self.g1 = Anafunc(g1, self.g1_d)

      self.g1sq_ddd = Anafunc(my_yfuncs.diff3_g1sq)
      self.g1sq_dd = Anafunc(g1sq_dd, self.g1sq_ddd)
      self.g1sq_d = Anafunc(g1sq_d, self.g1sq_dd)
      self.g1sq = Anafunc(g1sq, self.g1sq_d)

      self.g2_d = Anafunc(lambda y: 1e4 * g1sq_dd(y),\
                  Anafunc(lambda y: 1e4 * my_yfuncs.diff3_g1sq(y)))
      self.g2 = Anafunc(lambda y: 1 + 1e4 * g1sq_d(y), self.g2_d)

   @staticmethod
   def diff3_g1(y):
      fac = -2 / ysc**2
      t1 = g1_d(y)
      t2a = t1
      t2b = y * g1_dd(y)
      return fac * (t1 + t2a + t2b)

   @staticmethod
   def diff3_g1sq(y):
      fac = -4 / ysc**2
      t1 = g1sq_d(y)
      t2a = g1(y) * g1_d(y)
      t2b = y * g1_d(y)**2
      t2c = y * g1(y) * g1_dd(y)
      return fac * (t1 + 2 * (t2a + t2b + t2c))

xzm = Anafunc(lambda x: 1, Anafunc(lambda x: 0))

#zonal-mean background state
class ZM_bg:
   def __init__(self, **constkwargs):
      self.c = Bous_plane_consts(**constkwargs)
      c = self.c
      pf1 = lambda p: 1 + c.N2 / c.dens0 / c.g**2 * (c.p0 - p)
      pf1d = Anafunc(lambda p: -c.N2 / c.dens0 / c.g**2) #theta lapse rate from N2
      pf1 = Anafunc(pf1, deriv=pf1d)

      yf = my_yfuncs()
      self.th_bg = Anafld('th_bg', c.T0, xzm, yf.g2, pf1)
      self.th_y = self.th_bg.deriv('y')

      #TODO: debug U
      U_consts = -c.Rd * c.T0 * c.N2 / c.f0 / c.p0**c.kap / c.dens0 / c.g**2
      pf2d = Anafunc(lambda p: c.p0 * p**(c.kap - 1) - p**c.kap)
      Ut1 = lambda p: c.p0 / c.kap * (p**c.kap - c.p0**c.kap)
      Ut2 = lambda p: -(p**(c.kap + 1) - c.p0**(c.kap + 1)) / (c.kap + 1)
      pf2 = Anafunc(lambda p: Ut1(p) + Ut2(p), pf2d)
      Upfd = Anafunc(lambda p: c.T0 + pf2d.eval(p))
      Upf = Anafunc(lambda p: c.T0 * (p - c.p0) + Ut1(p) + Ut2(p), Upfd)
      thyy = self.th_y.deriv('y')
      print(thyy.funcs['y'])
      self.U_bg = Anafld('U_bg', U_consts, xzm, thyy.funcs['y'], Upf)

c = Bous_plane_consts()
f1 = lambda x: np.exp(4j * x / c.a / np.cos(c.lat0))
fx_d = Anafunc(lambda x: 4j / c.a / np.cos(c.lat0) * f1(x), Anafunc(lambda x: -4**2 / c.a**2 / np.cos(c.lat0)**2 * f1(x)))
fx = Anafunc(f1, fx_d)

abc = np.array([-1, 1.4e5, -2.875e9]) / 2.3e9
hp = Anafunc(lambda p: np.dot(abc, np.array([p**2, p, 1])), Anafunc(lambda p: np.dot(abc, np.array([2 * p, 1, 0]))))
bt = xzm
bc = Anafunc(lambda p: np.einsum('i,i...->...', abc, np.array([(c.p0**2 - p**2) / 2, c.p0 - p, np.log(c.p0 / p)])))

#eddy fields and fluxes for geostrophic and hydrostatic closed-form eddies
class Eddyflds:
   def __init__(self, Tamp=10, Zamp=200, fx=fx, gy=my_yfuncs().g1, hp=hp, bt=bt, bc=bc):
      self.Tamp, self.Zamp = Tamp, Zamp
      self.Tp = Anafld('Tp', Tamp * 1j, fx, gy, hp)
      self.Zt = Anafld('Z_barotropic', Zamp, fx, gy, bt)
      self.Zc = Anafld('Z_baroclinic', c.Rd * Tamp / c.g * 1j, fx, gy, bc)
      self.Zp = Anafld('Zp', 1, fx, gy, Anafunc(lambda p: Zamp + c.Rd * Tamp / c.g * 1j * bc.eval(p)))

      self.vp, self.up = self.Zp.deriv('x') * (c.g / c.f0), self.Zp.deriv('y') * (-c.g / c.f0)
