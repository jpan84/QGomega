import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ana_cls import my_yfuncs, Bous_plane_consts, ZM_bg, Eddyflds

yg = np.arange(-5e5, 5e5, 1e4)
a = my_yfuncs()
print(a.g1_dd.eval(0))
plt.plot(yg, a.g1.eval(yg))
ax = plt.twinx()
#ax.plot(yg, a.g1.deriv.eval(yg))
ax.plot(yg, a.g1.deriv.deriv.eval(yg))
#plt.show()
plt.close()

def plt_paxis_adj(ax=None):
   plabs = np.array([300, 400, 500, 600, 700, 850, 1000])
   if ax is None:
      ax = plt.gca()
   ax.set_yscale('log')
   ax.set_yticks(100 * plabs)
   ax.set_yticklabels(plabs)
   ax.invert_yaxis()
   ax.set_ylabel('p [hPa]')

c = Bous_plane_consts()
lons = np.linspace(0, 2 * np.pi, 100)
yy = np.linspace(-500, 500, 100) * 1e3
pp = np.arange(2.5e4, 1.e5 + 1, 2.5e3)
xg, yg, pg = np.meshgrid(lons * c.a * np.cos(c.lat0), yy, pp, indexing='ij')

bg = ZM_bg()
thlevs = np.arange(250, 400, 5)
Ulevs = np.arange(10, 100, 10)

thplt = bg.th_bg.eval(xg, yg, pg)
Uplt = bg.U_bg.eval(xg, yg, pg)
print(Uplt.min(), Uplt.max())
plt.contour(yg[0, ...], pg[0, ...], thplt[0, ...], levels=thlevs, colors='red')
plt.contour(yg[0, ...], pg[0, ...], Uplt[0, ...], levels=Ulevs, colors='black')
plt_paxis_adj()
plt.close()#plt.show()

ef = Eddyflds()
Zp = ef.Zt.eval(xg, yg, pg) + ef.Zc.eval(xg, yg, pg)
Zp = ef.Zp.eval(xg, yg, pg)
Tp = ef.Tp.eval(xg, yg, pg)
vp = ef.vp.eval(xg, yg, pg)
#x-p plane v, T, Z
plt.rcParams['figure.figsize'] = (12, 8)
vlevs = np.arange(5, 30, 5)
vlevs = np.concatenate((-vlevs[::-1], vlevs))
Zlevs = np.arange(60, 360, 60)
Zlevs = np.concatenate((-Zlevs[::-1], Zlevs))
yslc = (slice(None), 50, slice(None))
lonplt = xg[yslc] / c.a / np.cos(c.lat0) * 180 / np.pi 
csf = plt.contourf(lonplt, pg[yslc], Tp[yslc], cmap='RdBu_r')
plt.contour(lonplt, pg[yslc], Zp[yslc], levels=Zlevs, colors='black')
plt.contour(lonplt, pg[yslc], vp[yslc], levels=vlevs, colors='green')
plt.xlim(0, 90)
plt.xlabel('lon')
plt.title('Contours: Z anomaly (interval 60 m)\nShading: T anomaly [K]')
plt_paxis_adj()
plt.colorbar(csf)
plt.close()#plt.show()

psilevs = 1e10 * 2.**np.arange(0, 10)
PSI_vT = ef.PSI_vT.eval(xg, yg, pg)
print(PSI_vT.max())
#y-p plane EHF heating rate, streamfunc response
cs = plt.contour(yg[0, ...] / 1e3, pg[0, ...], PSI_vT[0, ...] / 1e10, levels=psilevs / 1e10, colors='black')
plt.clabel(cs, fmt='%d', inline=1, colors='black')
plt_paxis_adj()
plt.xlabel('y [km]')
plt.title('Contours: Residual streamfunction $\\bar{\Psi}^*$, vT term [10$^{10}$ kg s$^{-1}$]')
plt.close()#plt.show()

vpTp = ef.EHF.eval(xg, yg, pg)
#vpTp = ef.vp.eval(xg, yg, pg).real * ef.Tp.eval(xg, yg, pg).real
plt.contourf(xg[yslc], pg[yslc], vpTp[yslc], cmap='bwr', norm=colors.TwoSlopeNorm(0))
plt.xlabel('x [m]')
plt_paxis_adj()
plt.colorbar()
plt.show()
#plt.savefig('xp_vT.png')
#plt.close()

