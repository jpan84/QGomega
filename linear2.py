#Joshua Pan jp872@cornell.edu 202112
#Turn the QG omega equation (forcing from diff vort adv and temp adv) into a linear system by discretizing derivatives. Interpolate NCEP/NCAR Reanalysis 1 to evenly spaced p levels.
#Boundary conditions: homogeneous Dirichlet in pressure, periodic in lon, Dirichlet in lat from actual omega
#For advection terms, assume vg = v.
#Update: corrected sigma^2 to sigma in denominator of LHS pressure derivative.

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

dp = 1e4 #Pa
dy = a * 2.5*np.pi/180 #m
dlamb = 2.5*np.pi/180 #rad

def main():
   U = xr.open_dataset('%suwnd.%s.nc' % (DATADIR, YEAR)).uwnd
   V = xr.open_dataset('%svwnd.%s.nc' % (DATADIR, YEAR)).vwnd
   T = xr.open_dataset('%sair.%s.nc' % (DATADIR, YEAR)).air
   OMEGA = xr.open_dataset('%somega.%s.nc' % (DATADIR, YEAR)).omega
   DS = xr.Dataset(data_vars = {'U': U, 'V': V, 'T': T, 'OMEGA': OMEGA})
   DS = DS.sel(lat=slice(62.5, 27.4))
   DS = DS.reindex(lat = DS.lat.values[::-1]).fillna(0).sel(time = DS.time.values[1200])
   DS = DS.interp(level=np.arange(1000,99,-100.))
   DS = DS.assign(dx=lambda x: a * np.cos(x.lat*np.pi/180) * dlamb)
   DS = DS.assign(sigma=lambda x: sigmastab(x))

   '''
   slc = DS.sigma.sel(lat=47.5, lon=287.5)
   plt.plot(slc.values, slc.level)
   plt.xlabel('$\sigma \hspace{0.5} (J \hspace{0.5} kg^{-1} \hspace{0.5} Pa^{-2})$')
   plt.ylabel('p (hPa)')
   plt.yscale('log')
   plt.gca().invert_yaxis()
   plt.title('%.2f°N, %.2f°E' % (slc.lat.values, slc.lon.values))
   plt.savefig('sigmavsz.png')
   plt.close()
   '''

   forceTA = qgTA(DS)
   fig, ax = plt.subplots(figsize=(10,7), subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent([220, 310, 30, 60])
   ax.coastlines()
   cs = ax.contourf(forceTA.coords['lon'].values, forceTA.coords['lat'].values, forceTA.sel(level=700).values, cmap = 'BrBG', norm=colors.TwoSlopeNorm(vcenter=0))
   cbar = fig.colorbar(cs, shrink=0.7, orientation='horizontal')
   cbar.set_label('$Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1}$')
   plt.title('QG $\omega$ Temp Adv Forcing 700 hPa %s' % str(DS.time.values))
   plt.savefig('TAforcing.png', bbox_inches='tight')
   plt.close()

   forceVA = qgVA(DS)
   fig, ax = plt.subplots(figsize=(10,7), subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent([220, 310, 30, 60])
   ax.coastlines()
   cs = ax.contourf(forceVA.coords['lon'].values, forceVA.coords['lat'].values, forceVA.sel(level=500).values, cmap = 'BrBG', norm=colors.TwoSlopeNorm(vcenter=0))
   cbar = fig.colorbar(cs, shrink=0.7, orientation='horizontal')
   cbar.set_label('$Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1}$')
   plt.title('QG $\omega$ Diff Vort Adv Forcing 500 hPa %s' % str(DS.time.values))
   plt.savefig('DVAforcing.png', bbox_inches='tight')
   plt.close()

   qgforcing = forceTA + forceVA
   fig, ax = plt.subplots(figsize=(10,7), subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent([220, 310, 30, 60])
   ax.coastlines()
   cs = ax.contourf(qgforcing.coords['lon'].values, qgforcing.coords['lat'].values, qgforcing.sel(level=500).values, cmap = 'BrBG', norm=colors.TwoSlopeNorm(vcenter=0))
   cbar = fig.colorbar(cs, shrink=0.7, orientation='horizontal')
   cbar.set_label('$Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1}$')
   plt.title('QG $\omega$ Total Forcing (DVA + TA) 500 hPa %s' % str(DS.time.values))
   plt.savefig('QGforcing.png', bbox_inches='tight')
   plt.close()

   l = qgforcing.shape[0] + 1
   m = qgforcing.shape[1] + 1
   n = qgforcing.shape[2] - 1
   forcevec = np.reshape(qgforcing.values, (l-1)*(m-1)*(n+1), order='C') #vectorize forcing terms (last axis changing fastest in level,lat,lon)
   #handle the north/south boundary conditions
   for r in range(forcevec.shape[0]):
      i = 1 + r // (n+1) // (m-1)
      j = 1 + r // (n+1) % (m-1)
      k = r % (n+1)
      if j == 1:
         forcevec[r] -= DS.OMEGA.values[i, 0, k] / dy**2
      if j == m-1:
         forcevec[r] -= DS.OMEGA.values[i, m, k] / dy**2

   A = matLHS(DS)
   plt.spy(A[:3000,:3000])
   plt.savefig('matrix.png')
   plt.close()
   omegavec = np.linalg.solve(A, forcevec)
   omegavec = np.reshape(omegavec, (l-1, m-1, n+1), order='C')

   qgomega = xr.DataArray(omegavec, coords = qgforcing.coords, dims = qgforcing.dims)
   clabelkwargs = {'inline': 1, 'fontsize': 10, 'colors': 'black'}
   contourkwargs = {'colors': 'black', 'transform': ccrs.PlateCarree(), 'levels': np.arange(-5, 5.1, 0.2)}
   fig, ax = plt.subplots(figsize=(10,7), subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent([220, 310, 30, 60])
   ax.coastlines()
   cs = ax.contourf(qgomega.coords['lon'].values, qgomega.coords['lat'].values, qgomega.sel(level=500).values, cmap = 'BrBG_r', levels=15, norm=colors.TwoSlopeNorm(vcenter=0))
   cbar = fig.colorbar(cs, shrink=0.7, orientation='horizontal')
   cbar.set_label('$Pa \hspace{0.5} s^{-1}$')
   cs1 = ax.contour(DS.lon.values, DS.lat.values, DS.OMEGA.sel(level=500).values, **contourkwargs)
   ax.clabel(cs1, **clabelkwargs)
   plt.title('Colors: $\omega_{QG}$ (forcing by DVA + TA) 500 hPa %s\nContours: Reanalysis $\omega \hspace{0.5} [Pa \hspace{0.5} s^{-1}]$' % str(DS.time.values))
   plt.savefig('QGomega.png', bbox_inches='tight')
   plt.close()

   outds = xr.Dataset(data_vars=dict(OMEGAQG=qgomega, OMEGA=DS.OMEGA, FORCING=qgforcing, FORCETA=forceTA, FORCEVA=forceVA))
   outds = outds.fillna(0)
   outds.to_netcdf(path='QGomega.nc')

   qgerror = qgomega - DS.OMEGA
   fig, ax = plt.subplots(figsize=(10,7), subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent([220, 310, 30, 60])
   ax.coastlines()
   cs = ax.contourf(qgerror.coords['lon'].values, qgerror.coords['lat'].values, qgerror.sel(level=500).values, cmap = 'bwr', norm=colors.TwoSlopeNorm(vcenter=0))
   cbar = fig.colorbar(cs, shrink=0.7, orientation='horizontal')
   cbar.set_label('$Pa \hspace{0.5} s^{-1}$')
   plt.title('QG Error (QG $\omega$ minus Reanalysis) 500 hPa %s' % str(DS.time.values))
   plt.savefig('QGerror.png', bbox_inches='tight')
   plt.close()

def sigmastab(DS):
   #static stability parameter on interior pressure levels
   l = DS.level.shape[0] - 1
   reciprho = Rd * np.multiply(DS.T, 1/100/DS.level)

   slcup = slice(2, l+1)
   slcmid = slice(1, l)
   slcdown = slice(0, l-1)
   thtaup = thta(DS.T.values[slcup,:,:], DS.level.values[slcup])
   thtadown = thta(DS.T.values[slcdown,:,:], DS.level.values[slcdown])

   return reciprho[slcmid,:,:] * (np.log(thtaup) - np.log(thtadown)) / (2*dp)

def thta(T, p):
   #potential temp given T[K] and p[hPa]
   return np.multiply(T, (1000/p[:,None,None])**(Rd/cp))

def qgTA(DS):
   #compute the temp advection forcing term given a Dataset containing U, V, T, and sigma
   l = DS.level.shape[0] - 1
   m = DS.lat.shape[0] - 1
   slcpmid, slclatmid = slice(1, l), slice(1, m)
   return Rd / DS.sigma[slcpmid,slclatmid,:] / DS.level[slcpmid] / 100 * lapl(negTA(DS), DS.dx)[slcpmid,:,:]

def lapl(var, dx):
   #horizonal Laplacian of a DataArray using a centered 2nd derivative (periodic in lon, interior of domain in lat)
   n = var.lon.shape[0] - 1
   m = var.lat.shape[0] - 1
   slceast, slclonmid, slcwest = slice(1, n+1), slice(0, n+1), slice(0, n)
   slcnorth, slclatmid, slcsouth = slice(2, m+1), slice(1, m), slice(0, m-1)

   laply = (var.values[:,slcnorth,:] -2*var.values[:,slclatmid,:] + var.values[:,slcsouth,:]) / (dy**2)

   vareast, varwest = np.concatenate((var.values[:,:,slceast], var.values[:,:,0][:,:,None]), axis=2), np.concatenate((var.values[:,:,-1][:,:,None], var.values[:,:,slcwest]), axis=2)
   laplx = (vareast[:,slclatmid,:] - 2*var.values[:,slclatmid,slclonmid] + varwest[:,slclatmid,:]) / dx.values[None,slclatmid,None]**2

   return laplx + laply

def negTA(DS):
   #negative of temp advection dot(v, delT)
   gradT = grad(DS.T, DS.dx)
   return DS.U * gradT[0] + DS.V * gradT[1]

def grad(var, dx):
   #horizontal gradient of a DataArray using centered difference (periodic in lon, one-sided diff at north/south boundary)
   n = var.lon.shape[0] - 1
   m = var.lat.shape[0] - 1
   slceast, slclonmid, slcwest = slice(1, n+1), slice(0, n+1), slice(0, n)
   slcnorth, slclatmid, slcsouth = slice(2, m+1), slice(1, m), slice(0, m-1)

   grady = np.zeros(var.shape, dtype=np.float32)
   grady[:,slclatmid,:] = (var.values[:,slcnorth,:] - var.values[:,slcsouth,:]) / (2*dy)
   grady[:,0,:] = (var.values[:,1,:] - var.values[:,0,:]) / dy
   grady[:,-1,:] = (var.values[:,-1,:] - var.values[:,-2,:]) / dy

   gradx = np.zeros(var.shape, dtype=np.float32)
   vareast, varwest = np.concatenate((var.values[:,:,slceast], var.values[:,:,0][:,:,None]), axis=2), np.concatenate((var.values[:,:,-1][:,:,None], var.values[:,:,slcwest]), axis=2)
   gradx[:,:,slclonmid] = (vareast - varwest) / (2*dx.values[None,:,None])

   return gradx, grady

def qgVA(DS):
   #compute the differential vort advection forcing term given a Dataset containing U, V, T, and sigma
   l = DS.level.shape[0] - 1
   m = DS.lat.shape[0] - 1
   slcup = slice(2, l+1)
   slcmid = slice(1, l)
   slcdown = slice(0, l-1)
   slclatmid = slice(1, m)

   VA = negVA(DS)[:,slclatmid,:]
   DVA = (VA.values[slcup,:,:] - VA.values[slcdown,:,:]) / (-2*dp)

   return f0 / DS.sigma[slcmid,slclatmid,:] * DVA

def negVA(DS):
   #negative of absolute vort advection dot(v, del{eta})
   eta = absvort(DS)
   eta = xr.DataArray(eta, coords=DS.U.coords, dims = DS.U.dims)
   gradeta = grad(eta, DS.dx)
   return DS.U * gradeta[0] + DS.V * gradeta[1]

def absvort(DS):
   #compute absolute vorticity from a Dataset containing U, V
   relvortvx, _ = grad(DS.V, DS.dx)
   _, relvortuy = grad(DS.U, DS.dx)
   relvort = relvortvx - relvortuy
   planvort = 2 * OMEGArot * np.sin(DS.lat*np.pi/180)
   return relvort + planvort.values[None,:,None]

def matLHS(DS):
   #construct a matrix representation of the differential operators on the left-hand side of the QG omega equation
   l = DS.level.shape[0] - 1
   m = DS.lat.shape[0] - 1
   n = DS.lon.shape[0] - 1
   slcpmid = slice(1,l)
   slclatmid = slice(1,m)
   sz = (l-1)*(m-1)*(n+1)

   sigmavec = np.reshape(DS.sigma.values[slcpmid,slclatmid,:], sz, order='C')
   dxvec = DS.dx.values[None,slclatmid,None]
   dxvec = np.repeat(dxvec, l-1, axis=0)
   dxvec = np.repeat(dxvec, n+1, axis=2)
   dxvec = np.reshape(dxvec, sz, order='C')

   A = np.zeros((sz,sz), dtype=np.float64)
   for r in range(sz):
      ir, jr, kr = idx1Dto3D(r, l, m, n)
      for c in range(sz):
         ic, jc, kc = idx1Dto3D(c, l, m, n)
         if jc == jr and kc == kr and abs(ic-ir) == 1:
            A[r,c] = (f0 / dp)**2 / sigmavec[r]
         if ic == ir and kc == kr and abs(jc-jr) == 1:
            A[r,c] = 1/dy**2
         if ir == ic and jr == jc and abs(kc-kr) == 1:
            A[r,c] = 1/dxvec[r]**2
         if r == c:
            #print(f0, sigmavec[r], dp)
            A[r,c] = -2/dxvec[r]**2 - 2/dy**2 - 2*(f0 / dp)**2/sigmavec[r]
            #print(A[r,c])

         #handle periodic BCs
         if (kr == 0 and kc == n) or (kr == n and kc == 0):
            A[r,c] = 1/dxvec[r]**2

   return A

def idx1Dto3D(rc, l, m, n):
   k = rc % (n+1)
   j = 1 + rc // (n+1) % (m-1)
   i = 1 + rc // (n+1) // (m-1)
   return i, j, k

if __name__ == '__main__':
   main()
