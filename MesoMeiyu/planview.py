import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

BNDS = [110, 150, 25, 55]
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

plt.rc('font', size=18)
contourkwargs = {'colors': 'red', 'transform': ccrs.PlateCarree(), 'levels': np.arange(240, 331, 2)}
clabelkwargs = {'inline': 1, 'fontsize': 16, 'colors': 'red', 'fmt': '%.1f'}
clabelkwargs['fmt'] = '%d'
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.set_extent(BNDS)
ax.coastlines(color='black')
cs1 = ax.contour(ds.lon, ds.lat, ds.TEMP.sel(level=ULEV).values * (1000./ULEV)**(287/1004), **contourkwargs)
ax.clabel(cs1, **clabelkwargs)
csf = plt.contourf(shum.lon, shum.lat, 1000 * shum.isel(time=slice(712, 772)).mean(dim='time').sel(level=ULEV).values, cmap=new_cmap, levels=np.arange(5, 26, 5), extend='both')
cbar = fig.colorbar(csf, shrink=0.8)#, orientation='horizontal')
cbar.set_label('Specific humidity [g / kg]')
qv = plt.quiver(ds.lon, ds.lat, ds.UAG.sel(level=ULEV).values, ds.VAG.sel(level=ULEV).values, pivot='mid', scale=1.5e1, color='gray', width=5e-3)
plt.quiverkey(qv, X=.75, Y=-0.13, U=1, label='1 m s$^{-1}$', labelpos='E')
ax.set_xticks(np.arange(BNDS[0], BNDS[1] + 1, 10))
ax.set_yticks(np.arange(BNDS[2], BNDS[3] + 1, 5))
plt.title('%d hPa. Contours: $\\theta$ [K]\nVectors: $\\vec{v}_{ag}$ (QG approx.)' % ULEV)
plt.savefig('planview.png', bbox_inches='tight')
plt.close()
