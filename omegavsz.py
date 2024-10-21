import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

PTICKS = [100, 200, 300, 500, 700, 850, 1000]

ds = xr.open_dataset('qgomega.nc')
qgomega = ds.OMEGAQG
omega = ds.OMEGA
slc = qgomega.sel(lat=40, lon=290)
slc2 = omega.sel(lat=slc.lat.values, lon=slc.lon.values)
plt.plot(slc.values, slc.level, label='QG')
plt.plot(slc2.values, slc2.level, label='Reanalysis')
plt.xlabel('$\omega \hspace{0.5} [Pa \hspace{0.5} s^{-1}]$')
plt.ylabel('p [hPa]')
plt.yscale('log')
plt.yticks(PTICKS, labels=PTICKS)
plt.gca().invert_yaxis()
plt.legend()
plt.title('$\omega_{QG}$ %.2f°N, %.2f°E' % (slc.lat.values, slc.lon.values))
plt.savefig('omegavsz.png')
plt.close()

qgforcing = xr.open_dataset('qgomega.nc').FORCING
slc = qgforcing.sel(lat=slc.lat.values, lon=slc.lon.values)
plt.plot(slc.values, slc.level)
plt.xlabel('QG forcings (DVA + TA) $(Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1})$')
plt.ylabel('p (hPa)')
plt.yscale('log')
plt.yticks(PTICKS, labels=PTICKS)
plt.gca().invert_yaxis()
plt.title('%.2f°N, %.2f°E' % (slc.lat.values, slc.lon.values))
plt.savefig('qgforcingvsz.png')
plt.close()

clabelkwargs = {'inline': 1, 'fontsize': 10, 'colors': 'darkgrey', 'fmt': '%.1f'}
contourkwargs = {'colors': clabelkwargs['colors'], 'levels': np.arange(-5, 5.1, 0.2)}
slc = ds.OMEGAQG.sel(lat=37.5, lon = slice(260, 310))
slc2 = ds.OMEGA.sel(lat=slc.lat.values, lon=slc.lon.values)
slc3 = ds.UAG.sel(lat=slc.lat.values, lon=slc.lon.values)
csf = plt.contourf(slc.lon.values, slc.level.values, slc.values, cmap='BrBG_r', levels=15, norm=colors.TwoSlopeNorm(vcenter=0))
cs = plt.contour(slc2.lon.values, slc2.level.values, slc2.values, **contourkwargs)
qv = plt.quiver(slc.lon, slc.level, slc3, -1e1 * slc.values, pivot='mid', scale=1e2)
plt.quiverkey(qv, X=.75, Y=-0.1, U=5, label='5 m s$^{-1}$ 0.5 Pa s$^{-1}$', labelpos='E')
plt.xlabel('Longitude [°]')
plt.ylabel('p [hPa]')
plt.title(' %.1f°N %s\nShading: $\omega_{QG}\hspace{0.5} [Pa \hspace{0.5} s^{-1}]$; Contours: $\omega_{RA}$; Vectors: in-plane winds' % (slc.lat.values, slc.time.values))
plt.yscale('log')
plt.yticks(PTICKS, labels=PTICKS)
plt.gca().invert_yaxis()
plt.clabel(cs, **clabelkwargs)
plt.colorbar(csf)
plt.savefig('xsect.png', bbox_inches='tight')
plt.close()
