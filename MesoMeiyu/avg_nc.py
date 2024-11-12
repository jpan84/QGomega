#Joshua Pan 202411
#Average NCEP Reanalysis for METEO 529 Project

import numpy as np
import xarray as xr
import os

DATADIR = './Reanalysis/'
yrstr = '2003'
start_date = '06-28'
end_date = '07-12'
RAvars = ['uwnd', 'vwnd', 'air', 'hgt', 'omega']#, 'shum']
DRV = ['thta']#, 'thta_e']

Rd = 287 #J kg-1 K-1
cp = 1004 #J kg-1 K-1

def main():
   das = [xr.open_mfdataset(os.path.join(DATADIR, '%s*%s.nc' % (vr, yrstr)))[vr] for vr in RAvars]
   tsel = [da.sel(time=da.time.dt.month.isin([6, 7]) &
                       ((da.time.dt.month == 6) & (da.time.dt.day >= 28) | 
                        (da.time.dt.month == 7) & (da.time.dt.day <= 12))) for da in das]
   #tsel.append(thta(tsel[RAvars.index('air')]))

   tmean = [da.mean(dim='time') for da in tsel]
   [da.to_netcdf(path='%s.%s0.nc' % (da.name, yrstr)) for da in tmean]

def thta(da_T, pname='level'):
   #potential temp given T[K] and p[hPa]
   thta_da = np.multiply(da_T, (1000/da_T[pname])**(Rd/cp))
   thta_da.attrs['name'] = 'thta'
   return thta_da

#TODO: compute mean theta-e

if __name__ == '__main__':
   main()
