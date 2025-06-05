#Joshua Pan Jun 2025
#Classes for diagnostic analytical eddy flux model (EHF only)

class Anafld:
   def __init__(self, name, consts, lonfunc, yfunc, pfunc):
      self.name = name
      self.consts = consts
      self.funcs = dict(lon=lonfunc, y=yfunc, p=pfunc)

   def eval(self, xco, yco, pco):
      return self.consts * self.funcs['lon'].eval(xco) *\
             self.funcs['y'].eval(yco) * self.funcs['p'].eval(pco)

   def difffld(self, dim):
      newfuncs = self.funcs.copy()
      newfuncs[dim] = self.funcs[dim].deriv
      return Anafld(self.name + '_deriv_' + dim, self.consts,\
             newfuncs['lon'], newfuncs['y'], newfuncs['p'])

class Anafunc:
   def __init__(self, func, deriv):
      self.func = func
      self.deriv = deriv

   def eval(self, coords):
      return self.func(coords)

   def eval_deriv(self, coords):
      return self.deriv(coords)
