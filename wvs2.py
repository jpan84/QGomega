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
latcirc = 2 * np.pi * a * np.cos(lat0)

#Boussinesq consts
p0 = 1e5
N2 = 2e-4
T0 = 280
dens0 = p0 / Rd / T0
#print('dens0', dens0)

#eddy amplitudes
Zamp = 100
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

def symclevs(poslevs):
   return np.concatenate((-poslevs[::-1], poslevs))

Zlevs = symclevs(np.arange(60, 360, 60))
Ulevs = symclevs(np.arange(10, 100, 10))
vlevs = symclevs(np.arange(5, 30, 5))
wlevs = symclevs(1e-3 * np.arange(1, 10))
Tlevs = symclevs(np.arange(2, 11, 2))
adiablevs = symclevs(np.arange(1, 5)) # K/day
plabs = np.arange(300, 1001, 100) #np.array([300, 400, 500, 600, 700, 850, 1000])
psilevs = 2.**np.arange(0, 10) #1e10 kg s-1
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
   EHFc = -vpTp * g1sq_d(yg) / g1sq(yg)
   upvp = up.real * vp.real
   EMFc = -upvp * (g1_dd(yg) * g1(yg) + g1_d(yg)**2) / (g1_d(yg) * g1(yg))
   PSI_vT = latcirc * dens0 * g / T0 / N2 * vpTp.mean(axis=0)
   EPFz = g * f0 / T0 / N2 * vpTp
   EPFd_z = (-dens0 * g * EPFz * h_d(pg) / h(pg)).mean(axis=0)

   #background (zonal mean) fields
   th_vert = T0 * (1 + N2 / dens0 / g**2 * (p0 - pg))
   th_bg = th_vert * (1 + 1e4 * g1sq_d(yg))
   th_y = th_vert * 1e4 * g1sq_dd(yg)
   U_consts = 1e4 * Rd * T0 / f0 / p0**kap
   U_bg = U_consts * g1sq_dd(yg) * kap_poly_int(pg)
   print(U_bg.min(), U_bg.max())

   #synoptic fields
   Tadv = -(U_bg * diffx(Tp) + up.real * diffx(Tp).real + vp.real * (Tp * g1_d(yg) / g1(yg)).real + vp * th_y * (pg / p0)**kap) #missing advection of eddy by meridional Eulerian mean wind if not QG

   #x-p plane T, Z
   plt.rcParams['figure.figsize'] = (12, 8)
   yslc = (slice(None), 50, slice(None))
   plt_xp(xg, pg, yslc, Zp, Tp, 'Contours: Z\' (interval 60 m)\nShading: T\' [K]',\
             'xp_T_Z.png', dict(colors='black', levels=Zlevs), dict(cmap='RdBu_r'))

   #x-p plane Z, TA
   plt.rcParams['figure.figsize'] = (12, 8)
   yslc = (slice(None), 2, slice(None))
   plt_xp(xg, pg, yslc, Zp, Tadv * 86400, 'Contours: Z\' (interval 60 m)\nShading: T adv [K day$^{-1}$]',\
             'xp_Z_TA.png', dict(colors='black', levels=Zlevs), dict(cmap='RdBu_r'))

   #x-y plane Z, v
   #print(pg[..., 18])
   pslc = (slice(None), slice(None), 18)
   plt_xy(xg, yg, pslc, Zp, vp, '%d hPa     Contours: Z\' (interval 60 m)\nShading: v\' [m s$^{-1}$]' % (pg[pslc].min() / 100),\
             'xy_Z_v.png', dict(colors='black', levels=Zlevs), dict(cmap='bwr'))

   #x-y plane Z, EHF
   plt_xy(xg, yg, pslc, Zp, vpTp, '%d hPa     Contours: Z\' (interval 60 m)\nShading: v\'T\' [K m s$^{-1}$]' % (pg[pslc].min() / 100),\
             'xy_Z_EHF.png', dict(colors='black', levels=Zlevs), dict(cmap='bwr', norm=colors.TwoSlopeNorm(0)))

   #x-y plane Z, TA
   plt_xy(xg, yg, pslc, Zp, Tadv * 86400, '%d hPa     Contours: Z\' (interval 60 m)\nShading: T adv [K day$^{-1}$]' % (pg[pslc].min() / 100),\
             'xy_Z_TA.png', dict(colors='black', levels=Zlevs), dict(cmap='bwr', norm=colors.TwoSlopeNorm(0)))

   #x-y plane Z, EMF
   pslc = (slice(None), slice(None), 2)
   plt_xy(xg, yg, pslc, Zp, upvp, '%d hPa     Contours: Z\' (interval 60 m)\nShading: u\'v\' [m$^2$ s$^{-2}$]' % (pg[pslc].min() / 100),\
             'xy_Z_EMF.png', dict(colors='black', levels=Zlevs), dict(cmap='bwr', norm=colors.TwoSlopeNorm(0)))

   #y-p plane, PSIvT, EHFc
   plt_yp_zm(yg[0, ...], pg[0, ...], PSI_vT / 1e10, EHFc.mean(axis=0) * 86400,\
            'Contours: Residual streamfunction $\\bar{\Psi}^*$, vT term [10$^{10}$ kg s$^{-1}$]\nShading: EHF warming tendency [K day$^{-1}$]',\
            'yp_PSIvT_EHFc.png', dict(colors='black', levels=psilevs), dict(cmap='bwr'), clabelkw=dict(fmt='%d', inline=1, colors='black'))

   #y-p plane, EMF, EMFc
   plt_yp_zm(yg[0, ...], pg[0, ...], upvp.mean(axis=0), EMFc.mean(axis=0) * 86400,\
            'Contours: u\'v\' [m$^2$ s$^{-2}$]\nShading: EMF U tendency [m s$^{-1}$ day$^{-1}$]',\
            'yp_EMF_EMFc.png', dict(colors='black', levels=np.arange(50, 400, 50)), dict(cmap='bwr'), clabelkw=dict(fmt='%d', inline=1, colors='black'))

   #EPFz
   plt.rcParams['figure.figsize'] = (12, 8)
   csf = plt.contourf(yg[0, ...], pg[0, ...], EPFd_z * 86400, cmap='PuOr_r', norm=colors.TwoSlopeNorm(0))
   #plt.quiver(yg[0, ::8, ::4], pg[0, ::8, ::4], -upvp.mean(axis=0)[::8, ::4], EPFz.mean(axis=0)[::8, ::4] * 500, pivot='mid')
   plt.quiver(yg[0, ::8, ::4], pg[0, ::8, ::4], 0, EPFz.mean(axis=0)[::8, ::4] * 500, pivot='mid')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('yp_EPFz.png')
   plt.close()

   #EPF
   plt.rcParams['figure.figsize'] = (12, 8)
   csf = plt.contourf(yg[0, ...], pg[0, ...], (EMFc.mean(axis=0) + EPFd_z) * 86400, cmap='PuOr_r', norm=colors.TwoSlopeNorm(0))
   plt.quiver(yg[0, ::8, ::4], pg[0, ::8, ::4], -upvp.mean(axis=0)[::8, ::4], EPFz.mean(axis=0)[::8, ::4] * 500, pivot='mid')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('yp_EPF.png')
   plt.close()

   plt.rcParams['figure.figsize'] = (12, 8)
   plt.contour(yg[0, ...] / 1e3, pg[0, ...], th_bg[0, ...], levels=thlevs, colors='red')
   plt.contour(yg[0, ...] / 1e3, pg[0, ...], U_bg[0, ...], levels=Ulevs, colors='black')
   plt_paxis_adj()
   plt.savefig('yp_bg.png')
   plt.close()

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

def yargs_ddd(y):
   return 0

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

def g1_ddd(y):
   return yargs_ddd(y) * g1(y) + yargs_dd(y) * g1_d(y)\
          + yargs_dd(y) * g1_d(y) + yargs_d(y) * g1_dd(y)

def g1sq_dd(y):
   return 2 * (g1_d(y) * g1_d(y) + g1(y) * g1_dd(y))

def g1sq_ddd(y):
   return 2 * (2 * g1_d(y) * g1_dd(y)\
          + g1_d(y) * g1_dd(y) + g1(y) * g1_ddd(y))

def h(p):
   return np.dot(abc, np.array([p**2, p, 1]))

def h_d(p):
   return np.dot(abc[:-1], np.array([2 * p, 1]))

def h_int(p):
   #return np.dot(abc, np.array([(p0**2 - p**2) / 2, p0 - p, np.log(p0 / p)]))
   return np.einsum('i,i...->...', abc, np.array([(p0**2 - p**2) / 2, p0 - p, np.log(p0 / p)]))

#def kap_poly_int(p):
#   return p0 / kap * (p**kap - p0**kap) - (p**(kap + 1) - p0**(kap + 1)) / (kap + 1)

def kap_poly_int(p):
    return (p**kap - p0**kap) / kap * (1 + N2 * p0 / dens0 / g**2)\
           - (p**(kap + 1) - p0**(kap + 1)) / (kap + 1) * N2 / dens0 / g**2

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

def plt_yp_zm(yg, pg, contfld, conffld, title, outname, contkw, confkw, clabelkw=None):
   plt.rcParams['figure.figsize'] = (12, 8)
   csf = plt.contourf(yg / 1e3, pg, conffld, **confkw)
   cs = plt.contour(yg / 1e3, pg, contfld, **contkw)
   if clabelkw:
      plt.clabel(cs, **clabelkw)
   plt.xlabel('y [km]')
   plt.title(title)
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig(outname, bbox_inches='tight')
   plt.close()

def plt_paxis_adj(ax=None):
   if ax is None:
      ax = plt.gca()
   ax.set_yscale('log')
   ax.set_yticks(100 * plabs)
   ax.set_yticklabels(plabs)
   ax.invert_yaxis()
   ax.set_ylabel('p [hPa]')

#######################################

'''
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



def h_int_U(p, p_nochg=8.e4):
   return np.einsum('i,i...->...', abc, np.array([(p**2 - p_nochg**2) / 2, p - p_nochg, np.log(p / p_nochg)]))

#not needed
def actual_h_int(p):
   return np.dot(abc, np.array([p**3 / 3, p**2 / 2, p]))
'''


if __name__ == '__main__':
   main()
