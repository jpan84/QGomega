import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

PTICKS = [150, 200, 300, 400, 500, 600, 700, 850, 925]
BNDS = [110, 170, 25, 55]
ULEV = 925 

ds = xr.open_dataset('qgomega.nc')
shum = xr.open_dataset('Reanalysis/shum.2003.nc').shum
print(shum.time[712], shum.time[772])
print(ds.data_vars)

cmap = plt.cm.get_cmap('RdYlGn')
start = 0.4
end = 1.0
colors = cmap(np.linspace(start, end, 256))
new_cmap = mcolors.ListedColormap(colors)


contourkwargs = {'colors': 'red', 'transform': ccrs.PlateCarree(), 'levels': np.arange(240, 331, 2)}
clabelkwargs = {'inline': 1, 'fontsize': 10, 'colors': 'red', 'fmt': '%.1f'}
clabelkwargs['fmt'] = '%d'
fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.set_extent(BNDS)
ax.coastlines(color='black')
cs1 = ax.contour(ds.lon, ds.lat, ds.TEMP.sel(level=ULEV).values, **contourkwargs)
ax.clabel(cs1, **clabelkwargs)
qv = plt.quiver(ds.lon, ds.lat, ds.UAG.sel(level=ULEV).values, ds.VAG.sel(level=ULEV).values, pivot='mid', scale=2e1, color='gray')
plt.quiverkey(qv, X=.75, Y=-0.1, U=1, label='1 m s$^{-1}$', labelpos='E')
plt.contourf(shum.lon, shum.lat, 1000 * shum.isel(time=slice(712, 772)).mean(dim='time').sel(level=ULEV).values, cmap=new_cmap, levels=np.arange(5, 26, 5), extend='both')
plt.colorbar()
ax.set_xticks(np.arange(BNDS[0], BNDS[1] + 1, 10))
ax.set_yticks(np.arange(BNDS[2], BNDS[3] + 1, 5))
plt.title('%d hPa. Contours: $Z$ [m]; Vectors: $\\vec{v}_{ag}$ (QG approx.)' % ULEV)
plt.savefig('planview.png', bbox_inches='tight')
plt.close()


exit()

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
qv = plt.quiver(slc.lat, slc.level, slc3, -1e1 * slc.values, pivot='mid', scale=1e1, color='red')
plt.quiverkey(qv, X=.75, Y=-0.1, U=1, label='1 m s$^{-1}$ 0.1 Pa s$^{-1}$', labelpos='E')
plt.xlabel('Latitude [°]')
plt.ylabel('p [hPa]')
plt.title('%s %s\nShading: $\omega_{QG}\hspace{0.5} [Pa \hspace{0.5} s^{-1}]$; Contours: $\omega_{RA}$; Vectors: in-plane $\\vec{v}_{ag}$ (QG approx.)' % ('°E', '28 Jun–12 Jul'))
plt.ylim(150, 925)
#plt.yscale('log')
plt.yticks(PTICKS, labels=PTICKS)
plt.gca().invert_yaxis()
plt.clabel(cs, **clabelkwargs)
plt.colorbar(csf)
plt.savefig('xsect.png', bbox_inches='tight')
plt.close()

'''
clabelkwargs['fmt'] = '%d'
contourkwargs['levels'] = np.arange(-300, 301, 60)
zslc = ds.HGT.sel(lat=slc.lat.values, lon=slc.lon.values)
tslc = ds.TEMP.sel(lat=slc.lat.values, lon=slc.lon.values)
zslc, tslc = zslc - zslc.mean(dim='lon'), tslc - tslc.mean(dim='lon')
plt.figure(figsize=figsize)
csf = plt.contourf(tslc.lon, tslc.level, tslc.values, cmap='bwr', levels=15, norm=colors.TwoSlopeNorm(vcenter=0))
cs = plt.contour(zslc.lon, zslc.level, zslc.values, **contourkwargs)
qv = plt.quiver(slc.lon, slc.level, slc3, -1e1 * slc.values, pivot='mid', scale=1e2, color='lime')
plt.quiverkey(qv, X=.75, Y=-0.1, U=5, label='5 m s$^{-1}$ 0.5 Pa s$^{-1}$', labelpos='E')
plt.xlabel('Longitude [°]')
plt.ylabel('p [hPa]')
plt.title('%.1f°N %s\nShading: $T\'$[K]; Contours: $Z\'$ [m]; Vectors: in-plane $\\vec{v}_{ag}$ (QG approx.)' % (zslc.lat.values, zslc.time.values))
plt.yscale('log')
plt.yticks(PTICKS, labels=PTICKS)
plt.ylim(200, 1000)
plt.gca().invert_yaxis()
plt.clabel(cs, **clabelkwargs)
plt.colorbar(csf)
plt.savefig('wv_xsect.png', bbox_inches='tight')
plt.close()
'''
