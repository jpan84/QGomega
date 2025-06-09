#Joshua Pan jvp5930@psu.edu Jun 2025
#Diagnostic analytical baroclinic eddy flux model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#earth consts
g = 9.81
a = 6.371e6
OM = 7.29e-5

#thermo consts
Rd = 287
cp = 1004
kap = Rd / cp

#beta plane
lat0 = np.deg2rad(45)
f0 = 2 * OM * np.sin(lat0)
beta = 2 * OM / a * np.cos(lat0)

#Boussinesq consts
p0 = 1e5
N2 = 2e-4
T0 = 280
dens0 = p0 / Rd / T0
#print('dens0', dens0)

#eddy amplitudes
Zamp = 200
Tamp = 10

#domain coord vectors
lons = np.linspace(0, 2 * np.pi, 100)
yy = np.linspace(-500, 500, 100) * 1e3
pp = np.arange(2.5e4, 1.e5 + 1, 2.5e3)

#separable function params
abc = np.array([-1, 1.4e5, -2.875e9]) / 2.3e9
zwn = 4 #zonal wavenumber
ysc = 1e6 #y-scale
#print(abc)

plt.rc('font', size=16)

Zlevs = np.arange(60, 360, 60)
Zlevs = np.concatenate((-Zlevs[::-1], Zlevs))
Ulevs = np.arange(10, 100, 10)
vlevs = np.arange(5, 30, 5)
vlevs = np.concatenate((-vlevs[::-1], vlevs))
wlevs = 1e-3 * np.arange(1, 10)
wlevs = np.concatenate((-wlevs[::-1], wlevs))
Tlevs = np.arange(2, 11, 2)
Tlevs = np.concatenate((-Tlevs[::-1], Tlevs))
adiablevs = np.arange(1, 5) # K/day
adiablevs = np.concatenate((-adiablevs[::-1], adiablevs))
plabs = np.array([300, 400, 500, 600, 700, 850, 1000])
psilevs = 1e10 * 2.**np.arange(0, 10)
thlevs = np.arange(250, 400, 5)

def main():
   xg, yg, pg = np.meshgrid(lons * a * np.cos(lat0), yy, pp, indexing='ij')

   #eddy state fields
   Tp = Tamp * 1j * f(xg) * g1(yg) * h(pg)
   Z0 = Zamp * f(xg) * g1(yg) #barotropic comp
   Zp = Z0 + f(xg) * g1(yg) * Rd * Tamp / g * 1j * h_int(pg)
   up = -g / f0 * (Zp * g1_d(yg) / g1(yg))
   vp = g / f0 * diffx(Zp)
   #print(vp.max())

   #eddy flux fields
   vpTp = vp.real * Tp.real
   EHFc = vpTp * g1sq_d(yg) / g1sq(yg)
   upvp = up.real * vp.real
   EMFc = upvp * (g1_dd(yg) * g1(yg) + g1_d(yg)**2) / (g1_d(yg) * g1(yg))

   #x-p plane v, T, Z
   plt.rcParams['figure.figsize'] = (12, 8)
   yslc = (slice(None), 50, slice(None))
   plt_xp(xg, pg, yslc, Zp, Tp, 'Contours: Z\' (interval 60 m)\nShading: T\' [K]',\
             'xp_T_Z.png', dict(colors='black', levels=Zlevs), dict(cmap='RdBu_r'))

   #x-y plane Z, EHF
   #print(pg[..., 18])
   pslc = (slice(None), slice(None), 18)
   plt_xy(xg, yg, pslc, Zp, vp, '%d hPa     Contours: Z\' (interval 60 m)\nShading: v\' [m s$^{-1}$]' % (pg[pslc].min() / 100),\
             'xy_Z_v.png', dict(colors='black', levels=Zlevs), dict(cmap='bwr'))

   plt_xy(xg, yg, pslc, Zp, vpTp, '%d hPa     Contours: Z\' (interval 60 m)\nShading: v\'T\' [K m s$^{-1}$]' % (pg[pslc].min() / 100),\
             'xy_Z_EHF.png', dict(colors='black', levels=Zlevs), dict(cmap='bwr', norm=colors.TwoSlopeNorm(0)))

def f(x):
   return np.exp(zwn * 1j * x / a / np.cos(lat0))

def diffx(fn):
   return zwn * 1j / a / np.cos(lat0) * fn

def yargs(y):
   return -zwn * 1j * y / a / np.cos(lat0) - (y / ysc)**2

def yargs_d(y):
   return -zwn * 1j / a / np.cos(lat0) - 2 * y / ysc**2

def yargs_dd(y):
   return -2 / ysc**2

def g1(y):
   return np.exp(yargs(y))

def g1sq(y):
   return g1(y)**2

def g1_d(y):
   return yargs_d(y) * g1(y)

def g1sq_d(y):
   return 2 * g1(y) * g1_d(y)

def g1_dd(y):
   return yargs_dd(y) * g1(y) + yargs_d(y) * g1_d(y)


#######################################

def plt_xy(xg, yg, pslc, contfld, conffld, title, outname, contkw, confkw):
   plt.rcParams['figure.figsize'] = (14, 2)
   lonplt = xg[pslc] / a / np.cos(lat0) * 180 / np.pi
   csf = plt.contourf(lonplt, yg[pslc] / 1e3, conffld[pslc], **confkw)
   plt.contour(lonplt, yg[pslc] / 1e3, contfld[pslc], **contkw)
   plt.title(title)
   plt.xlim(0, 90)
   plt.xlabel('lon')
   plt.ylabel('y [km]')
   plt.colorbar(csf)
   plt.savefig(outname, bbox_inches='tight')
   plt.close()

def plt_xp(xg, pg, yslc, contfld, conffld, title, outname, contkw, confkw):
   plt.rcParams['figure.figsize'] = (12, 8)
   lonplt = xg[yslc] / a / np.cos(lat0) * 180 / np.pi 
   csf = plt.contourf(lonplt, pg[yslc], conffld[yslc], **confkw)
   plt.contour(lonplt, pg[yslc], contfld[yslc], **contkw)
   plt.xlim(0, 90)
   plt.xlabel('lon')
   plt.title(title)
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig(outname, bbox_inches='tight')
   plt.close()

#######################################

def g2(y):
   return 1e4 * EHFd(g1(y)**2, y)

def eigy(y, coef=np.pi / 1e6):
   return np.sin(coef * y), coef

def eigy_pv(y, coef=np.pi / 2e6):
   return np.cos(coef * y), coef

#y derivative of squared perturbations
def EHFd(ehf, y):
   return -2 / 1e6**2 * 2 * y * ehf

def diffy(fn, y):
   return EHFd(fn, y) / 2

def diff2_g1(y):
   return -2 / 1e6**2 * (g1(y) + y * diffy(g1(y), y))

#2nd y derivative of squared Gaussian
def diff2_g1sq(y):
   return -4 / 1e6**2 * (g1(y)**2 + y * 2 * g1(y) * diffy(g1(y), y))

def diff3_g1(y):
   fac = -2 / 1e6**2
   t1 = diffy(g1(y), y)
   t2a = diffy(g1(y), y)
   t2b = y * diff2_g1(y)
   return fac * (t1 + t2a + t2b)

#3rd y derivative of squared Gaussian
def diff3_g1sq(y):
   fac = -4 / 1e6**2
   t1 = EHFd(g1(y)**2, y)
   t2a = g1(y) * diffy(g1(y), y)
   t2b = y * diffy(g1(y), y) * diffy(g1(y), y)
   t2c = y * g1(y) * diff2_g1(y)
   return fac * (t1 + 2 * (t2a + t2b + t2c))

def h(p):
   return np.dot(abc, np.array([p**2, p, 1]))

def diffh(p):
   return 2 * abc[0] * p + abc[1]

def d2_hint(p):
   return -abc[0] + abc[2] / p**2

def eigp(p, coef=np.pi/7.5e4):
   return np.sin(coef * (p - 2.5e4)), coef

def eigp_pv(p, coef=np.pi / 1.4e5):
   return np.sin(coef * (p - 4e4)), coef

def h_int(p):
   #return np.dot(abc, np.array([(p0**2 - p**2) / 2, p0 - p, np.log(p0 / p)]))
   return np.einsum('i,i...->...', abc, np.array([(p0**2 - p**2) / 2, p0 - p, np.log(p0 / p)]))

def h_int_U(p, p_nochg=8.e4):
   return np.einsum('i,i...->...', abc, np.array([(p**2 - p_nochg**2) / 2, p - p_nochg, np.log(p / p_nochg)]))

#not needed
def actual_h_int(p):
   return np.dot(abc, np.array([p**3 / 3, p**2 / 2, p]))

def plt_paxis_adj(ax=None):
   if ax is None:
      ax = plt.gca()
   ax.set_yscale('log')
   ax.set_yticks(100 * plabs)
   ax.set_yticklabels(plabs)
   ax.invert_yaxis()
   ax.set_ylabel('p [hPa]')

if __name__ == '__main__':
   main()
