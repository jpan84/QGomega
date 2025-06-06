import numpy as np
import matplotlib.pyplot as plt
from ana_cls import my_yfuncs, Bous_plane_consts, ZM_bg

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
plt.show()
