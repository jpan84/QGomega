import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from windspharm.xarray import VectorWind

DIRI = './NCRA/'
PNM = 'level'

OM = 7.29e-5
grav = 9.81
Rd = 287
cp = 1005
ae = 6.371e6

LAT1, LAT2 = 10., 80.

def main():
   ds = xr.open_mfdataset(os.path.join(DIRI, '*.2026.nc')).sel(level=slice(200, None)).chunk(level=-1).isel(time=slice(None, 40))
   ds = ds.assign(f=2 * OM * np.sin(np.deg2rad(ds['lat'])))
   ds = ds.assign(thta=ds['air'] * (1000 / ds['level'])**(Rd / cp))
   ds = ds.assign(dens=ds['level'] * 100 / Rd / ds['air'])
   ds = ds.assign(sig=-1 / ds['dens'] / ds['thta'] * ds['thta'].differentiate('level', edge_order=2) / 100)
   ds = ds.assign(sigzm=ds['sig'].mean(dim='lon'))
   print(ds)

   #plt.contourf(ds['lat'], ds['level'], ds['sig'].mean(dim=['time', 'lon']), levels=np.arange(0, 2.1e-5, 2e-6))
   #plt.gca().invert_yaxis()
   #plt.colorbar()
   #plt.savefig('sigma_stab_test.png')
   #exit()

   vw = VectorWind(ds['uwnd'], ds['vwnd'])
   Z_x, Z_y = vw.gradient(ds['hgt'])
   T_x, T_y = vw.gradient(ds['air'])

   ug = -grav / ds['f'] * Z_y
   vg = grav / ds['f'] * Z_x
   Tp = ds['air'] - ds['air'].mean(dim='lon')
   vpTp = (vg * Tp).mean(dim='lon')
   LTA = (vpTp * np.cos(np.deg2rad(ds['lat']))).differentiate('lat', edge_order=2)\
            .differentiate('lat', edge_order=2).differentiate('lat', edge_order=2)
   LTA = LTA * (180 / np.pi / ae)**3 * Rd / ds['sigzm'] / ds['level'] / 100

   ulevs = np.arange(-10, 55, 5)
   ulevs = ulevs[ulevs != 0]
   plt.contourf(ds['lat'], ds['level'], vpTp.mean(dim=['time']).T, levels=np.arange(-50, 51, 5), cmap='bwr')
   #plt.contourf(ds['lat'], ds['level'], ds['omega'].mean(dim=['time', 'lon']), cmap='BrBG_r', levels=np.arange(-1e-1, 1.1e-1, 2e-2))
   plt.xlim(LAT1, LAT2)
   plt.gca().invert_yaxis()
   plt.colorbar()
   #plt.contour(ds['lat'], ds['level'], ds['omega'].mean(dim=['time', 'lon']), colors='black', levels=np.arange(-4e-2, 4.1e-2, 8e-3))
   plt.contour(ds['lat'], ds['level'], ds['uwnd'].mean(dim=['time', 'lon']), colors='black', levels=ulevs)
   qv = plt.quiver(ds['lat'], ds['level'], ds['vwnd'].mean(dim=['time', 'lon']), -1e2 * ds['omega'].mean(dim=['time', 'lon']), pivot='mid', scale=2e1, color='gray')
   plt.quiverkey(qv, X=.75, Y=-0.1, U=1, label='1 m s$^{-1}$ 0.01 Pa s$^{-1}$', labelpos='E')
   plt.savefig('EHF_test.png')
   plt.close()

   up = ug - ug.mean(dim='lon')
   upvp = (up * vg).mean(dim='lon')
   plt.contourf(ds['lat'], ds['level'], upvp.mean(dim=['time']).T, levels=np.arange(-120, 121, 20), cmap='bwr', extend='both')
   plt.xlim(LAT1, LAT2)
   plt.gca().invert_yaxis()
   plt.colorbar()
   #plt.contour(ds['lat'], ds['level'], ds['omega'].mean(dim=['time', 'lon']), colors='black', levels=np.arange(-4e-2, 4.1e-2, 8e-3))
   plt.contour(ds['lat'], ds['level'], ds['uwnd'].mean(dim=['time', 'lon']), colors='black', levels=ulevs)
   qv = plt.quiver(ds['lat'], ds['level'], ds['vwnd'].mean(dim=['time', 'lon']), -1e2 * ds['omega'].mean(dim=['time', 'lon']), pivot='mid', scale=2e1, color='gray')
   plt.quiverkey(qv, X=.75, Y=-0.1, U=1, label='1 m s$^{-1}$ 0.01 Pa s$^{-1}$', labelpos='E')
   plt.savefig('EMF_test.png')
   plt.close()

   plt.contourf(ds['lat'], ds['level'], LTA.mean(dim=['time']).T, levels=np.arange(-1e-13, 1.01e-13, 1e-14), cmap='BrBG')
   plt.xlim(LAT1, LAT2)
   plt.gca().invert_yaxis()
   plt.colorbar()
   plt.savefig('TA_zmform_test.png')
   plt.close()
   exit()


   vw_geo = VectorWind(ug, vg)

   #plt.contourf(ds['lat'], ds['level'], ug.mean(dim=['time', 'lon']).T)
   #plt.xlim(LAT1, LAT2)
   #plt.gca().invert_yaxis()
   #plt.colorbar()
   #plt.savefig('ug_test.png')
   #plt.close()

   avor = vw.absolutevorticity() #f=0 causes problems with geostrophic vorticity
   print(np.where(np.isnan(avor)))
   avor_x, avor_y = vw_geo.gradient(avor)
   VA = -ug * avor_x - vg * avor_y
   DVA = -ds['f'] / ds['sigzm'] * VA.differentiate('level', edge_order=2) / 100

   plt.contourf(ds['lat'], ds['level'], DVA.mean(dim=['time', 'lon']).T, levels=np.arange(-1e-13, 1.01e-13, 1e-14), cmap='BrBG')
   plt.xlim(LAT1, LAT2)
   plt.gca().invert_yaxis()
   plt.colorbar()
   plt.savefig('DVA_test.png')
   plt.close()

   TA = -ds['uwnd'] * T_x - ds['vwnd'] * T_y
   plt.contourf(ds['lat'], ds['level'], TA.mean(dim=['time', 'lon']), levels=np.arange(-4e-5, 4.1e-5, 1e-5), cmap='BrBG')
   plt.xlim(LAT1, LAT2)
   plt.gca().invert_yaxis()
   plt.colorbar()
   plt.savefig('TA_test.png')
   plt.close()

   TA_x, TA_y = vw_geo.gradient(TA)
   TA_xx, _ = vw_geo.gradient(TA_x)
   TA_yy, _ = vw_geo.gradient(TA_y)
   LTA = -Rd / ds['sigzm'] / ds[PNM] / 100 * (TA_xx + TA_yy)

   plt.contourf(ds['lat'], ds['level'], LTA.mean(dim=['time', 'lon']), levels=np.arange(-1e-13, 1.01e-13, 1e-14), cmap='BrBG')
   plt.xlim(LAT1, LAT2)
   plt.gca().invert_yaxis()
   plt.colorbar()
   plt.savefig('LTA_test.png')
   plt.close()

if __name__ == '__main__':
   main()
