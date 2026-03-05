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

LAT1, LAT2 = 20., 80.

def main():
   ds = xr.open_mfdataset(os.path.join(DIRI, '*.2026.nc')).sel(level=slice(200, None)).chunk(level=-1)
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

if __name__ == '__main__':
   main()
