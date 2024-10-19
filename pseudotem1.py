#Joshua Pan jp872@cornell.edu 20221028
#Updated 1030 to plot cross sections
#Separate vertical motion forced by Tadv and diabatic

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs

DATADIR = './reanalysis/'
YEAR = '2018'

Rd = 287.06 #J kg-1 K-1
a = 6.371e6 #m
f0 = 1.031e-4 #s-1
cp = 1004 #J kg-1 K-1
OMEGArot = 7.29e-5 #s-1
g = 9.81 #m s-2

dp = 1e4 #Pa
dy = a * 2.5*np.pi/180 #m
dlamb = 2.5*np.pi/180 #rad

def main():
   U = xr.open_dataset('%suwnd.%s.nc' % (DATADIR, YEAR)).uwnd
   V = xr.open_dataset('%svwnd.%s.nc' % (DATADIR, YEAR)).vwnd
   T = xr.open_dataset('%sair.%s.nc' % (DATADIR, YEAR)).air
   Z = xr.open_dataset('%shgt.%s.nc' % (DATADIR, YEAR)).hgt
   OMEGA = xr.open_dataset('%somega.%s.nc' % (DATADIR, YEAR)).omega
   DS = xr.Dataset(data_vars = {'U': U, 'V': V, 'T': T, 'OMEGA': OMEGA, 'Z': Z})

   DS = DS.isel(time = slice(236, 360))
   '''
   tstep = 6 * 3600
   Z = DS.Z
   Z = Z.assign_coords(time=np.arange(0, tstep * DS.time.shape[0], tstep))
   #print(Z.time)

   #print((Z.differentiate('level', edge_order=2).differentiate('time', edge_order=2) / 100).max())

   thicktend = Z.differentiate('level', edge_order=2).differentiate('time', edge_order=2) / 100
   thicktend = thicktend.assign_coords(time=DS.time)
   '''
   sig = sigmastab(DS.T, DS.level).clip(min=1e-6)
   stokes = -Rd * tadv(DS.U, DS.V, DS.T) / DS.level / 100 / sig #- thicktend * g / sig
   #print(stokes)

   omegalev, ulev = 700, 850
   extent = [250, 340, 20, 60]
   pltU = DS.U.mean(dim='time').sel(level=ulev)
   pltstokes = stokes.mean(dim='time').sel(level=omegalev)
   clabelkwargs = {'inline': 1, 'fontsize': 10, 'colors': 'black', 'fmt': '%d'}
   contourkwargs = {'colors': 'black', 'transform': ccrs.PlateCarree(), 'levels': np.arange(-50, 51, 5)}
   contourfkwargs = dict(cmap = 'BrBG_r', levels=np.arange(-0.2, 0.21, .04))
   fig, ax = plt.subplots(figsize=(10,7), subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent(extent)
   ax.coastlines()
   cs = ax.contourf(DS.lon, DS.lat, pltstokes.values, **contourfkwargs)#15, norm=colors.TwoSlopeNorm(vcenter=0, vmin=-0.1, vmax=0.1))
   cbar = fig.colorbar(cs, shrink=0.7, orientation='horizontal')
   cbar.set_label('$Pa \hspace{0.5} s^{-1}$')
   cs1 = ax.contour(DS.lon, DS.lat, pltU.values, **contourkwargs)
   ax.clabel(cs1, **clabelkwargs)
   plt.title('Colors: Pseudo-Stokes $\omega$ (forced by Tadv) %d hPa %s\nContours: U [m s$^{-1}$] %d hPa' % (omegalev, 'March 2018', ulev))
   plt.savefig('psuedostokes.png', bbox_inches='tight')
   plt.close()

   #plt.hist(pltstokes.values)
   #plt.savefig('hist.png')
   #plt.close()

   pseudotem = DS.OMEGA.mean(dim='time') - stokes.mean(dim='time')
   testpt = dict(lat=35, lon=295)
   #print(DS.OMEGA.sel(level=omegalev, **testpt), pltstokes.sel(testpt), pseudotem.sel(level=omegalev, **testpt))

   plttem = pseudotem.sel(level=omegalev)
   fig, ax = plt.subplots(figsize=(10,7), subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent(extent)
   ax.coastlines()
   cs = ax.contourf(DS.lon, DS.lat, plttem.values, **contourfkwargs)
   cbar = fig.colorbar(cs, shrink=0.7, orientation='horizontal')
   cbar.set_label('$Pa \hspace{0.5} s^{-1}$')
   cs1 = ax.contour(DS.lon, DS.lat, pltU.values, **contourkwargs)
   ax.clabel(cs1, **clabelkwargs)
   plt.title('Colors: Pseudo-TEM $\omega$ (diabatically forced) %d hPa %s\nContours: U [m s$^{-1}$] %d hPa' % (omegalev, 'March 2018', ulev))
   plt.savefig('psuedotem.png', bbox_inches='tight')
   plt.close()
 
   lonslc = (280, 320)
   slc = dict(lat=slice(60, 20), lon=slice(*lonslc))
   temslc = pseudotem.sel(slc).mean(dim='lon')
   Uslc = DS.U.sel(slc).mean(dim=['lon', 'time'])
   contourkwargs['transform'] = None
   cs = plt.contourf(temslc.lat, temslc.level, temslc.values, **contourfkwargs)
   cs1 = plt.contour(Uslc.lat, Uslc.level, Uslc.values, **contourkwargs)
   plt.xlabel('Lat [°]')
   plt.ylabel('p [hPa]')
   plt.title('Diabatically forced $\omega$ $[Pa \hspace{0.5} s^{-1}]$ %d-%d°E %s' % (lonslc[0], lonslc[1], 'March 2018'))
   plt.ylim(100, 1000)
   plt.yscale('log')
   plt.gca().invert_yaxis()
   tickkwargs = dict(ticks = [100, 200, 300, 400, 500, 600, 700, 850, 1000], labels = ['100', '200', '300', '400', '500', '600', '700', '850', '1000'])
   plt.yticks(**tickkwargs)
   plt.clabel(cs1, **clabelkwargs)
   plt.colorbar(cs)
   plt.savefig('temxsect.png', bbox_inches='tight')
   plt.close()

   stokesslc = DS.OMEGA.sel(slc).mean(dim=['lon', 'time'])
   cs = plt.contourf(stokesslc.lat, stokesslc.level, stokesslc.values, **contourfkwargs)
   cs1 = plt.contour(Uslc.lat, Uslc.level, Uslc.values, **contourkwargs)
   plt.xlabel('Lat [°]')
   plt.ylabel('p [hPa]')
   plt.title('Mean $\omega$ $[Pa \hspace{0.5} s^{-1}]$ %d-%d°E %s' % (lonslc[0], lonslc[1], 'March 2018'))
   plt.ylim(100, 1000)
   plt.yscale('log')
   plt.gca().invert_yaxis()
   plt.yticks(**tickkwargs)
   plt.clabel(cs1, **clabelkwargs)
   plt.colorbar(cs)
   plt.savefig('stokesxsect.png', bbox_inches='tight')
   plt.close()

def sigmastab(T, p):
   lnth = np.log(thta(T, p))
   dens = Rd * T / p / 100
   dlnth_dp = lnth.differentiate('level', edge_order=2) / 100
   return -dlnth_dp / dens

def thta(T, p):
   #potential temp given T[K] and p[hPa]
   return np.multiply(T, (1000/p)**(Rd/cp))

def tadv(u, v, T):
   Tx = T.differentiate('lon', edge_order=2) * 180/np.pi / a / np.cos(T.lat*np.pi/180)
   Ty = T.differentiate('lat', edge_order=2) * 180/np.pi / a
   return -u * Tx - v * Ty


if __name__ == '__main__':
   main()
