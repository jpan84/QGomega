import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

lat0 = np.deg2rad(45)
g = 9.81
a = 6.371e6
p0 = 1e5
Rd = 287
cp = 1004
kap = Rd / cp
OM = 7.29e-5
f0 = 2 * OM * np.sin(lat0)
N2 = 2e-4
T0 = 280
dens0 = p0 / Rd / T0
#print('dens0', dens0)

lons = np.linspace(0, 2 * np.pi, 100)
yy = np.linspace(-500, 500, 100) * 1e3
pp = np.arange(2.5e4, 1.e5 + 1, 2.5e3)

abc = np.array([-1, 1.4e5, -2.875e9]) / 2.3e9
#print(abc)

Zamp = 200
Tamp = 10
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
   #print(lons, yy, pp)
   xg, yg, pg = np.meshgrid(lons * a * np.cos(lat0), yy, pp, indexing='ij')
   #print(pg[..., -1])

   #print(f(xg))
   #print(h(pg[1,1,:]))

   Tp = Tamp * 1j * f(xg) * g1(yg) * h(pg)
   Z0 = Zamp * f(xg) * g1(yg)
   Zp = Z0 + f(xg) * g1(yg) * Rd * Tamp / g * 1j * h_int(pg)
   up = -g / f0 * diffy(Zp, yg)
   vp = g / f0 * diffx(Zp)
   print(vp.max())

   #x-p plane v, T, Z
   plt.rcParams['figure.figsize'] = (12, 6)
   yslc = (slice(None), 50, slice(None))
   lonplt = xg[yslc] / a / np.cos(lat0) * 180 / np.pi 
   csf = plt.contourf(lonplt, pg[yslc], Tp[yslc], cmap='RdBu_r')
   plt.contour(lonplt, pg[yslc], Zp[yslc], levels=Zlevs, colors='black')
   #plt.contour(lonplt, pg[yslc], vp[yslc], levels=vlevs, colors='green')
   plt.xlim(0, 90)
   plt.xlabel('lon')
   plt.title('Contours: Z anomaly (interval 60 m)\nShading: T anomaly [K]')
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig('xp_v_T_Z.png')
   plt.close()

   #x-p plane EHF
   vpTp = vp.real * Tp.real
   plt.contourf(xg[yslc], pg[yslc], vpTp[yslc], cmap='bwr', norm=colors.TwoSlopeNorm(0))
   plt.xlabel('x [m]')
   plt_paxis_adj()
   plt.colorbar()
   #plt.show()
   plt.savefig('xp_vT.png')
   plt.close()

   #x-y plane Z, EHF
   #print(pg[..., 18])
   pslc = (slice(None), slice(None), 18)
   lonplt = xg[pslc] / a / np.cos(lat0) * 180 / np.pi
   csf = plt.contourf(lonplt, yg[pslc] / 1e3, vpTp[pslc], cmap='bwr', norm=colors.TwoSlopeNorm(0))
   plt.contour(lonplt, yg[pslc] / 1e3, Zp[pslc], levels=Zlevs, colors='black')
   plt.title('%d hPa     Contours: Z anomaly (interval 60 m)\nShading: EHF v\'T\' [K m s$^{-1}$]' % (pg[pslc].min() / 100))
   plt.xlabel('lon')
   plt.ylabel('y [km]')
   plt.colorbar(csf)
   plt.savefig('xy_Z_vT.png')
   plt.close()

   DT_EHF = -EHFd(vpTp, yg)
   PSI_vT = g / T0 / N2 * vpTp.mean(axis=0) * 2 * np.pi * a * np.cos(lat0)
   DT_ADIAB = -N2 * T0 / g * EHFd(PSI_vT[None, ...], yg) / 2 / np.pi / a / np.cos(lat0) #= w N^2 T0/g
   print(PSI_vT.max())
   #y-p plane EHF heating rate, streamfunc response
   csf = plt.contourf(yg[0, ...] / 1e3, pg[0, ...], DT_ADIAB.mean(axis=0) * 86400, cmap='bwr')
   cs = plt.contour(yg[0, ...] / 1e3, pg[0, ...], PSI_vT / 1e10, levels=psilevs / 1e10, colors='black')
   plt.clabel(cs, fmt='%d', inline=1, colors='black')
   plt_paxis_adj()
   plt.xlabel('y [km]')
   plt.title('Contours: Residual streamfunction $\\bar{\Psi}^*$, vT term [10$^{10}$ kg s$^{-1}$]')
   plt.colorbar(csf, label='adiabatic warming associated with $\\bar{\Psi}^*$ [K day$^{-1}$]')
   plt.savefig('yp_DTADIAB.png')
   plt.close()
   csf = plt.contourf(lonplt, yg[pslc] / 1e3, DT_EHF[pslc] * 86400, cmap='bwr', norm=colors.TwoSlopeNorm(0))
   plt.contour(lonplt, yg[pslc] / 1e3, Zp[pslc], levels=Zlevs, colors='black')
   plt.title('%d hPa     Contours: Z anomaly (interval 60 m)\nShading: Convergence of v\'T\' [K day$^{-1}$]' % (pg[pslc].min() / 100))
   plt.xlabel('lon')
   plt.ylabel('y [km]')
   plt.colorbar(csf)
   plt.savefig('xy_DTEHF.png')
   plt.close()

   '''
   #y-p plane streamfunc response, adiab DT
   DT_ADIAB = -N2 * T0 / g * EHFd(PSI_vT[None, ...], yg) / 2 / np.pi / a / np.cos(lat0) #= w N^2 T0/g
   csf = plt.contourf(yg[0, ...], pg[0, ...], DT_ADIAB.mean(axis=0), cmap='bwr')
   plt.contour(yg[0, ...], pg[0, ...], PSI_vT, levels=psilevs, colors='black')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('yp_DTADIAB.png')
   plt.close()
   '''

   #zonal-mean temp advection
   DT_ADV = -(up.real * diffx(Tp).real + vp.real * diffy(Tp.real, yg)).mean(axis=0)
   DT_ADV_EIG = DT_ADV.max() * eigy(yg)[0] * eigp(pg)[0]
   csf = plt.contourf(yg[0, ...], pg[0, ...], DT_ADV, cmap='bwr')
   plt.contour(yg[0, ...], pg[0, ...], DT_ADV_EIG[0, ...], levels=1e-5*np.arange(-10, 11, 2.5), colors='black')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('yp_DTADV.png')
   plt.close()

   #zonal-mean QG eigval solution
   evy, evp = eigy(0)[1], eigp(0)[1]
   W0_zm = g * DT_ADV.max() * evy**2 / N2 / T0 / (evy**2 + (dens0 * g * f0 * evp)**2 / N2)
   wTA_zm = (W0_zm * eigy(yg)[0] * eigp(pg)[0])[0, ...]
   w_vT = g / N2 / T0 * EHFd(vpTp, yg)
   print(wTA_zm.max(), np.nanmean(wTA_zm * N2 * T0 / g / DT_ADV))
   csf = plt.contourf(yg[0, ...], pg[0, ...], DT_ADV, cmap='bwr')
   plt.contour(yg[0, ...], pg[0, ...], wTA_zm, levels=wlevs, colors='black')
   plt.contour(yg[0, ...], pg[0, ...], w_vT.mean(axis=0), levels=wlevs * 3, colors='red')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('yp_WTA.png')
   plt.close()

   #plot zonal-mean eigval solution cleanly
   v_TA_QG = -dens0 * g * W0_zm * evp / evy * (np.cos(evy * yg) * np.cos(evp * (pg - 2.5e4)))[0, ...]
   DTADIAB_TAresp = -wTA_zm * N2 * T0 / g
   print(wTA_zm.max(), np.nanmean(wTA_zm * N2 * T0 / g / DT_ADV))
   csf = plt.contourf(yg[0, ...] / 1e3, pg[0, ...], DT_ADV * 86400, cmap='bwr')
   cs = plt.contour(yg[0, ...] / 1e3, pg[0, ...], DTADIAB_TAresp * 86400, levels=adiablevs, colors='purple')   
   qv = plt.quiver(yg[0, ::8, ::4] / 1e3, pg[0, ::8, ::4], v_TA_QG[::8, ::4], wTA_zm[::8, ::4] * 100, pivot='mid', scale=1e1, width=5e-3)
   plt.quiverkey(qv, X=0.75, Y=-0.1, U=1, label='1 m, cm s$^{-1}$', labelpos='E')
   plt_paxis_adj()
   plt.clabel(cs, fmt='%d', inline=1, colors='purple')
   plt.title('Contours: adiabatic T tendency induced by QG secondary circulation [K day$^{-1}$]', color='purple')
   plt.xlabel('y [km]')
   plt.colorbar(csf, label='T advection [K day$^{-1}$]')
   plt.savefig('yp_QG.png')
   plt.close()

   th_vert = T0 * (1 + N2 / dens0 / g**2 * (p0 - pg))
   th_bg = th_vert * (1 + g2(yg))
   th_y = th_vert * -1e4 * diff2_g1sq(yg) #= th_bg * d/dy(1e4 * (4 / 1e6**2 * yg) * g1(y)**2)
   U_consts = Rd * T0 * N2 / f0 / p0**kap / dens0 / g**2 * -1e4 * diff2_g1sq(yg)
   U_bg = U_consts * (T0 * (pg - p0) + (p0 / kap * (pg**kap - p0**kap) - (pg**(kap + 1) - p0**(kap + 1)) / (kap + 1)))
   print(U_bg.min(), U_bg.max())
   #y-p plane pot temp, its gradient, and U background
   csf = plt.contourf(yg[0, ...], pg[0, ...], th_y[0, ...], cmap='bwr')
   plt.contour(yg[0, ...], pg[0, ...], th_bg[0, ...], levels=thlevs, colors='red')
   plt.contour(yg[0, ...], pg[0, ...], U_bg[0, ...], levels=Ulevs, colors='black')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('yp_bg.png')
   plt.close()

   Tadv_bg = -U_bg * diffx(Tp)
   #x-p plane advection of wave temp by background flow
   csf = plt.contourf(xg[yslc], pg[yslc], Tadv_bg[yslc], cmap='bwr')
   plt.contour(xg[yslc], pg[yslc], Tp[yslc], levels=Tlevs, colors='red')
   plt.xlabel('x [m]')
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig('xp_Tadv_bg.png')
   plt.close()

   #x-y plane T adv
   up = -g / f0 * diffy(Zp, yg)
   #print(up.max(), vp.max())
   pltadv = -(-Tadv_bg + up.real * diffx(Tp).real + vp.real * diffy(Tp.real, yg) + vp * th_y * (pg / p0)**kap) * 86400
   pltadv1 = -(up.real * diffx(Tp).real + vp.real * diffy(Tp.real, yg)).mean(axis=0) * 86400
   print('maxdiff', abs(pltadv.mean(axis=0) - pltadv1).max())
   pslc = (slice(None), slice(None), 18)
   lonplt = xg[pslc] / a / np.cos(lat0) * 180 / np.pi 
   csf = plt.contourf(lonplt, yg[pslc] / 1e3, pltadv[pslc], cmap='bwr')
   plt.contour(lonplt, yg[pslc] / 1e3, Zp[pslc], levels=Zlevs, colors='black')
   qvslc = (slice(None, None, 4), slice(None, None, 4), 18)
   qv = plt.quiver(lonplt[::4, ::4], yg[qvslc] / 1e3, (U_bg + up.real)[qvslc], vp[qvslc], pivot='mid', scale=1e3)
   plt.quiverkey(qv, X=0.75, Y=-0.1, U=20, label='20 m s$^{-1}$', labelpos='E')
   plt.xlabel('lon')
   plt.ylabel('y [km]')
   plt.colorbar(csf)
   plt.title('%d hPa      Contours: Z anomaly [interval 60 m]\nShading: T advection [K day$^{-1}$]' % (pg[pslc].min() / 100))
   plt.savefig('xy_Z_vTA.png')
   plt.close()
   #zonal-mean temp advection
   csf = plt.contourf(yg[0, ...], pg[0, ...], pltadv1, cmap='bwr')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('total_TA_check.png')
   plt.close()

   #x-y plane T total
   csf = plt.contourf(lonplt, yg[pslc] / 1e3, (th_bg * (pg / p0)**kap + Tp)[pslc] - 273.15, cmap='RdYlBu_r')
   plt.contour(lonplt, yg[pslc] / 1e3, Zp[pslc], levels=Zlevs, colors='black')
   qvslc = (slice(None, None, 4), slice(None, None, 4), 18)
   qv = plt.quiver(lonplt[::4, ::4], yg[qvslc] / 1e3, (U_bg + up.real)[qvslc], vp[qvslc], pivot='mid', scale=1e3)
   plt.quiverkey(qv, X=0.75, Y=-0.1, U=20, label='20 m s$^{-1}$', labelpos='E')
   plt.xlabel('lon')
   plt.ylabel('y [km]')
   plt.colorbar(csf)
   plt.title('%d hPa      Contours: Z anomaly [interval 60 m]\nShading: T [°C]' % (pg[pslc].min() / 100))
   plt.savefig('xy_Zp_Ttot_Vtot.png')
   plt.close()

   rv =  g / f0 * (diffx(diffx(Zp)) + f(xg) * diff2_g1(yg) * (Zamp + Rd * Tamp / g * 1j * h_int(pg)))
   #x-p plane relvort, Z
   csf = plt.contourf(xg[yslc], pg[yslc], rv[yslc], cmap='bwr')
   plt.contour(xg[yslc], pg[yslc], Zp[yslc], levels=Zlevs, colors='black')
   plt.xlabel('x [m]')
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig('xp_rv_Z.png')
   plt.close()

   rvadv = -U_bg * diffx(rv)
   #x-p plane Z, relvort advection
   csf = plt.contourf(xg[yslc], pg[yslc], rvadv[yslc], cmap='bwr')
   plt.contour(xg[yslc], pg[yslc], Zp[yslc], levels=Zlevs, colors='black')
   plt.xlabel('x [m]')
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig('xp_Z_rva.png')
   plt.close()

   #x-y plane Z, EHF
   #print(pg[..., 0])
   pslc = (slice(None), slice(None), 0)
   csf = plt.contourf(xg[pslc], yg[pslc], rv[pslc], cmap='bwr')
   plt.contour(xg[pslc], yg[pslc], Zp[pslc], levels=Zlevs, colors='black')
   plt.colorbar(csf)
   plt.savefig('xy_rv_Z.png')
   plt.close()

   ################ PV/vort section below #######################
   psip = g / f0 * Zp
   qp = diffx(diffx(psip)) + diff2_g1(yg) / g1(yg) * psip\
        - f0 * dens0**2 * g**2 / N2 * f(xg) * g1(yg) * Rd * Tamp * 1j * d2_hint(pg)
   #qp = - f0 * dens0**2 * g**2 / N2 * f(xg) * g1(yg) * Rd * Tamp * 1j * d2_hint(pg)
   #x-p plane Z, q'
   lonplt = xg[yslc] / a / np.cos(lat0) * 180 / np.pi
   csf = plt.contourf(lonplt, pg[yslc], qp[yslc], cmap='BrBG')
   plt.contour(lonplt, pg[yslc], Zp[yslc], levels=Zlevs, colors='black')
   plt.xlim(0, 90)
   plt.xlabel('lon')
   plt.title('Contours: Z anomaly (interval 60 m)\nShading: q\' [s$^{-1}$]')
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig('xp_Z_qp.png')
   plt.close()

   #x-p plane Z, v'q'
   vpqp = vp.real * qp.real
   lonplt = xg[yslc] / a / np.cos(lat0) * 180 / np.pi
   csf = plt.contourf(lonplt, pg[yslc], vpqp[yslc], cmap='PuOr_r', norm=colors.TwoSlopeNorm(0))
   plt.contour(lonplt, pg[yslc], Zp[yslc], levels=Zlevs, colors='black')
   plt.xlim(0, 90)
   plt.xlabel('lon')
   plt.title('Contours: Z anomaly (interval 60 m)\nShading: v\'q\' [m s$^{-2}$]')
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig('xp_Z_vpqp.png')
   plt.close()

   #x-y plane Z, v'q'
   #print(pg[..., 2])
   pslc = (slice(None), slice(None), 2)
   lonplt = xg[pslc] / a / np.cos(lat0) * 180 / np.pi
   csf = plt.contourf(lonplt, yg[pslc] / 1e3, vpqp[pslc], cmap='PuOr_r', norm=colors.TwoSlopeNorm(0))
   plt.contour(lonplt, yg[pslc] / 1e3, Zp[pslc], levels=Zlevs, colors='black')
   plt.title('%d hPa     Contours: Z anomaly (interval 60 m)\nShading: v\'q\' [m s$^{-2}$]' % (pg[pslc].min() / 100))
   plt.xlabel('lon')
   plt.ylabel('y [km]')
   plt.colorbar(csf)
   plt.savefig('xy_Z_vpqp.png')
   plt.close()

   #y-p plane vT streamfunc, v'q'
   csf = plt.contourf(yg[0, ...] / 1e3, pg[0, ...], vpqp.mean(axis=0), cmap='PuOr_r', norm=colors.TwoSlopeNorm(0))
   cs = plt.contour(yg[0, ...] / 1e3, pg[0, ...], PSI_vT / 1e10, levels=psilevs / 1e10, colors='black')
   plt.clabel(cs, fmt='%d', inline=1, colors='black')
   plt_paxis_adj()
   plt.xlabel('y [km]')
   plt.title('Contours: Residual streamfunction $\\bar{\Psi}^*$, vT term [10$^{10}$ kg s$^{-1}$]')
   plt.colorbar(csf, label='v\'q\' [m s$^{-2}$]')
   plt.savefig('yp_vpqp_PSIvT.png')
   plt.close()

   #y-p plane v'q', eigfunc approx
   PVF_EIG = 4.2e-3 * eigy_pv(yg)[0] * eigp_pv(pg)[0]
   csf = plt.contourf(yg[0, ...] / 1e3, pg[0, ...], vpqp.mean(axis=0), cmap='PuOr_r', norm=colors.TwoSlopeNorm(0))
   cs = plt.contour(yg[0, ...] / 1e3, pg[0, ...], PVF_EIG[0, ...], levels=np.arange(-5e-3, 5.1e-3, 5e-4), colors='green')
   plt_paxis_adj()
   plt.xlabel('y [km]')
   plt.title('Contours: Eigfunc approx')
   plt.colorbar(csf, label='v\'q\' [m s$^{-2}$]')
   plt.savefig('yp_vpqp_eig.png')
   plt.close()


def f(x):
   return np.exp(4j * x / a / np.cos(lat0))

def diffx(fn):
   return 4j / a / np.cos(lat0) * fn

def g1(y):
   return np.exp(-(y / 1e6)**2)

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
   return 4 / 1e6**2 * (g1(y)**2 + y * 2 * g1(y) * diffy(g1(y), y))

def h(p):
   return np.dot(abc, np.array([p**2, p, 1]))

def d2_hint(p):
   return -abc[0] + abc[2] / p**2

def eigp(p, coef=np.pi/7.5e4):
   return np.sin(coef * (p - 2.5e4)), coef

def eigp_pv(p, coef=np.pi / 1.4e5):
   return np.sin(coef * (p - 4e4)), coef

def h_int(p):
   #return np.dot(abc, np.array([(p0**2 - p**2) / 2, p0 - p, np.log(p0 / p)]))
   return np.einsum('i,i...->...', abc, np.array([(p0**2 - p**2) / 2, p0 - p, np.log(p0 / p)]))

def plt_paxis_adj():
   plt.yscale('log')
   plt.yticks(100 * plabs, labels=plabs)
   plt.gca().invert_yaxis()
   plt.ylabel('p [hPa]')

if __name__ == '__main__':
   main()
