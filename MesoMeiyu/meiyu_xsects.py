import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

PTICKS = [150, 200, 300, 400, 500, 600, 700, 850, 925]

ds = xr.open_dataset('qgomega.nc')
qgomega = ds.OMEGAQG
omega = ds.OMEGA
slc = qgomega.sel(lat=37.5, lon=280)
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

figsize = (6, 4)
clabelkwargs = {'inline': 1, 'fontsize': 10, 'colors': 'black', 'fmt': '%.2f'}
contourkwargs = {'colors': clabelkwargs['colors'], 'levels': np.arange(-5, 0, 0.02)}
contourkwargs['levels'] = np.concatenate((contourkwargs['levels'], -contourkwargs['levels'][::-1]))
slc = ds.OMEGAQG.sel(lat=slice(25, 55), lon = slice(117, 124))
slc2 = ds.OMEGA.sel(lat=slc.lat.values, lon=slc.lon.values)
slc3 = ds.VAG.sel(lat=slc.lat.values, lon=slc.lon.values)
slc, slc2, slc3 = slc.mean(dim='lon'), slc2.mean(dim='lon'), slc3.mean(dim='lon')
plt.figure(figsize=figsize)
csf = plt.contourf(slc.lat.values, slc.level.values, slc.values, cmap='BrBG_r', levels=np.arange(-0.05, 0.0501, 0.01))
cs = plt.contour(slc2.lat.values, slc2.level.values, slc2.values, **contourkwargs)
qv = plt.quiver(slc.lat, slc.level, slc3, -1e1 * slc.values, pivot='mid', scale=5, color='red')
plt.quiverkey(qv, X=.75, Y=-0.1, U=0.5, label='0.5 m s$^{-1}$ 0.05 Pa s$^{-1}$', labelpos='E')
plt.xlabel('Latitude [°]')
plt.ylabel('p [hPa]')
plt.title('(b) %s %s\nShading: $\omega_{QG}\hspace{0.5} [Pa \hspace{0.5} s^{-1}]$; Contours: $\omega_{RA}$; Vectors: in-plane $\\vec{v}_{ag}$ (QG approx.)\n' % ('°117-124°E', '28 Jun–12 Jul'))
plt.ylim(150, 925)
#plt.yscale('log')
plt.yticks(PTICKS, labels=PTICKS)
plt.gca().invert_yaxis()
plt.clabel(cs, **clabelkwargs)
plt.colorbar(csf)
plt.savefig('xsect.png', bbox_inches='tight')
plt.close()

figsize = (6, 4)
clabelkwargs = {'inline': 1, 'fontsize': 10, 'colors': 'black', 'fmt': '%d'}
contourkwargs = {'colors': clabelkwargs['colors'], 'levels': np.arange(270, 351, 5)}
#contourkwargs['levels'] = np.concatenate((contourkwargs['levels'], -contourkwargs['levels'][::-1]))
slc = ds.FORCETA.sel(lat=slice(25, 55), lon = slice(117, 124))
slc2 = ds.OMEGAQG.sel(lat=slc.lat.values, lon=slc.lon.values)
slc3 = ds.VAG.sel(lat=slc.lat.values, lon=slc.lon.values)
slc4 = ds.TEMP.sel(lat=slc.lat.values, lon=slc.lon.values)
slc4 = slc4 * (ds.level / 1000) ** (-287 / 1004)
#print(slc4.min())
slc, slc2, slc3, slc4 = slc.mean(dim='lon'), slc2.mean(dim='lon'), slc3.mean(dim='lon'), slc4.mean(dim='lon')
plt.figure(figsize=figsize)
csf = plt.contourf(slc.lat.values, slc.level.values, slc.values, cmap='BrBG', levels=1e-13 * np.arange(-10, 10.1, 2.5), extend='both')
cs = plt.contour(slc4.lat.values, slc4.level.values, slc4.values, **contourkwargs)
qv = plt.quiver(slc.lat, slc.level, slc3, -1e1 * slc2.values, pivot='mid', scale=5, color='red')
plt.quiverkey(qv, X=-.25, Y=-0.1, U=0.5, label='0.5 m s$^{-1}$ 0.05 Pa s$^{-1}$', labelpos='E')
plt.xlabel('Latitude [°]')
plt.ylabel('p [hPa]')
plt.title('(c) %s %s\nShading: QG $T_{adv}$ forcing [Pa m$^{-2}$ $s^{-1}$]; Contours: $\\theta$ [K]; Vectors: in-plane $\\vec{v}_{ag}$ (QG approx.)\n' % ('117–124°E', '28 Jun–12 Jul'))
plt.ylim(150, 925)
#plt.yscale('log')
plt.yticks(PTICKS, labels=PTICKS)
plt.gca().invert_yaxis()
plt.clabel(cs, **clabelkwargs)
plt.colorbar(csf)
plt.savefig('TAxsect.png', bbox_inches='tight')
plt.close()
