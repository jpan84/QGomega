#Joshua Pan jp872@cornell.edu 202112 updated 202410
#Turn the QG omega equation (forcing from diff vort adv and temp adv) into a linear system by discretizing derivatives. Interpolate NCEP/NCAR Reanalysis 1 to evenly spaced p levels.
#Estimate divergent wind using QG continuity equation
#Boundary conditions: homogeneous at top, Neumann divergence at bottom, periodic in lon, Dirichlet in lat from actual omega
#For advection terms, assume vg = v.
#Account for horizontal variations in static stability.

import xarray as xr
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs

DATADIR = './'
YEAR = '20030'
tstr = '28 Jun–12 Jul'

Rd = 287.06 #J kg-1 K-1
a = 6.371e6 #m
f0 = 1.031e-4 #s-1
cp = 1004 #J kg-1 K-1
OMEGArot = 7.29e-5 #s-1

EPS = np.finfo(float).eps

dp = 7.5e3 #Pa
dy = a * 2.5*np.pi/180 #m
dlamb = 2.5*np.pi/180 #rad

ULEV = 250.
MLEV = 550.
LLEV = 700.
BNDS = [80, 170, 30, 60]

def main():
   U = xr.open_dataset('%suwnd.%s.nc' % (DATADIR, YEAR)).uwnd
   V = xr.open_dataset('%svwnd.%s.nc' % (DATADIR, YEAR)).vwnd
   T = xr.open_dataset('%sair.%s.nc' % (DATADIR, YEAR)).air
   Z = xr.open_dataset('%shgt.%s.nc' % (DATADIR, YEAR)).hgt
   OMEGA = xr.open_dataset('%somega.%s.nc' % (DATADIR, YEAR)).omega
   DS = xr.Dataset(data_vars = {'U': U, 'V': V, 'T': T, 'Z': Z, 'OMEGA': OMEGA})
   DS = DS.sel(lat=slice(70., 25))
   DS = DS.reindex(lat = DS.lat.values[::-1]).fillna(0)#.sel(time = DS.time.values[115])
   DS = DS.interp(level=np.arange(1000,99,-dp/100))
   DS = DS.assign(dx=lambda x: a * np.cos(x.lat*np.pi/180) * dlamb)
   DS = DS.assign(sigma=lambda x: sigmastab(x))

   #acct for horiz variations in stability
   laplogsig = lapl(np.log(np.clip(DS.sigma, a_min=EPS, a_max=None)), DS.dx)
   laplogsig = np.pad(laplogsig, pad_width=((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=np.nan)
   laplogsig = xr.DataArray(laplogsig, dims=DS.sigma.dims, coords=DS.sigma.coords)
   #print(laplogsig.shape)
   #print(DS.sigma.shape)
   DS = DS.assign(op3=laplogsig)

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
   tatitle = 'QG $\omega$ Temp Adv Forcing %d hPa %s' % (int(LLEV), tstr)
   plotmap(forceTA, LLEV, BNDS, tatitle, '$Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1}$', 'TAforcing.png')

   forceVA = qgVA(DS)
   vatitle = 'QG $\omega$ Diff Vort Adv Forcing %d hPa %s' % (int(MLEV), tstr)
   plotmap(forceVA, MLEV, BNDS, vatitle, '$Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1}$', 'DVAforcing.png', norm=colors.TwoSlopeNorm(vcenter=0))

   qgforcing = forceTA + forceVA
   frctitle = 'QG $\omega$ Total Forcing (DVA + TA) %d hPa %s' % (int(MLEV), tstr)
   plotmap(qgforcing, MLEV, BNDS, frctitle, '$Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1}$', 'QGforcing.png', norm=colors.TwoSlopeNorm(vcenter=0))

   udiv, _ = grad(DS.U, DS.dx)
   _, vdiv = grad(DS.V, DS.dx)
   hdiv = udiv + vdiv

   l = qgforcing.shape[0] + 1
   m = qgforcing.shape[1] + 1
   n = qgforcing.shape[2] - 1
   forcevec = np.reshape(qgforcing.values, (l-1)*(m-1)*(n+1), order='C') #vectorize forcing terms (last axis changing fastest in level,lat,lon)
   #print(qgforcing.shape)
   #print(hdiv.shape)
   #print(DS.sigma.shape)
   #handle the Dirichlet meridional and Neumann vertical boundary conditions
   for r in range(forcevec.shape[0]):
      i = 1 + r // (n+1) // (m-1)
      j = 1 + r // (n+1) % (m-1)
      k = r % (n+1)
      if j == 1: #meridional
         forcevec[r] -= DS.OMEGA.values[i, 0, k] / dy**2
      if j == m-1:
         forcevec[r] -= DS.OMEGA.values[i, m, k] / dy**2
      #if i == 0: #vertical
      #   forcevec[r] += hdiv[0, j+1, k] * f0**2 / DS.sigma.values[1, j+1, k] / dp #sigma is only avail on internal levels
      #if i == l-1: !UBC
         #print(DS.sigma.values[l, j, k])
         #forcevec[r] -= hdiv[-1, j+1, k] * f0**2 / DS.sigma.values[l-1, j+1, k] / dp #TODO: make indexing consistent between un/padded fields

   print('Generating A matrix...')
   A = matLHS(DS)
   plt.spy(A[:3000,:3000])
   plt.savefig('matrix.png')
   plt.close()
   print('Solving BVP...')
   omegavec = spsolve(A, forcevec)
   omegavec = np.reshape(omegavec, (l-1, m-1, n+1), order='C')

   #compute vertical BC values from Neumann BC solution
   lbc = (omegavec[0,:,:] - hdiv[0, 1:-1,:] * dp) * 0
   ubc = np.zeros_like(lbc) #omegavec[-1,:,:] + hdiv[-1, 1:-1,:] * dp !UBC
   #print(lbc.shape, ubc.shape, omegavec.shape)
   omegavec = np.concatenate((lbc[None,:,:], omegavec, ubc[None,:,:]), axis=0)
   #print(omegavec.shape)
   #omegavec[0,:,:] = lbc
   #omegavec[-1,:,:] = ubc

   #qgomega = xr.DataArray(omegavec, coords = qgforcing.coords, dims = qgforcing.dims)
   qgomega = xr.DataArray(omegavec, coords=[DS.level, DS.lat[1:-1], DS.lon], dims=['level', 'lat', 'lon'])
   omegatitle = 'Colors: $\omega_{QG}$ (forcing by DVA + TA) %d hPa %s\nContours: Reanalysis $\omega \hspace{0.5} [Pa \hspace{0.5} s^{-1}]$' % (int(MLEV), tstr)
   clabelkwargs = {'inline': 1, 'fontsize': 10, 'colors': 'black', 'fmt': '%.1f'}
   contourkwargs = {'colors': 'black', 'transform': ccrs.PlateCarree(), 'levels': np.arange(-5, 5.1, 0.02)}
   plotmap(qgomega, MLEV, BNDS, omegatitle, '$Pa \hspace{0.5} s^{-1}$', 'QGomega.png', cmap='BrBG_r', norm=colors.TwoSlopeNorm(vcenter=0), contour=True, cntda=DS.OMEGA, clabelkwargs=clabelkwargs, contourkwargs=contourkwargs)

   qgerror = qgomega - DS.OMEGA
   errtitle = 'QG Error (QG $\omega$ minus Reanalysis) %d hPa %s' % (int(MLEV), tstr)
   plotmap(qgerror, MLEV, BNDS, errtitle, '$Pa \hspace{0.5} s^{-1}$', 'QGerror.png', cmap='bwr', norm=colors.TwoSlopeNorm(vcenter=0))

   outds = xr.Dataset(data_vars=dict(OMEGAQG=qgomega, OMEGA=DS.OMEGA, TEMP=DS.T, HGT=DS.Z, FORCING=qgforcing, FORCETA=forceTA, FORCEVA=forceVA))
   outds = outds.fillna(0)
   outds.to_netcdf(path='QGomega.nc')

   print('Computing ageo winds...')
   #ompad = np.pad(qgomega, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.)
   ompad = np.pad(qgomega, ((0, 0), (1, 1), (0, 0)), 'constant', constant_values=0.)
   ompad = xr.DataArray(ompad, coords=[DS.level, DS.lat, DS.lon], dims=['level', 'lat', 'lon'])
   diver = -d_dp(ompad, dp)
   divervec = np.reshape(diver, (l-1)*(m+1)*(n+1), order='C') #vectorize divergence (last axis changing fastest in level,lat,lon)
   Lmat = lapmat(DS, latm=0)
   vpot = spsolve(Lmat, divervec)
   #print(l, m, n, vpot.shape)
   vpot = np.reshape(vpot, (l-1, m+1, n+1), order='C')
   vpot = xr.DataArray(vpot, coords=[DS.level[1:-1], DS.lat, DS.lon], dims=['level', 'lat', 'lon'])
   uag, vag = grad(vpot, DS.dx)
   uag = xr.DataArray(uag, coords=vpot.coords, dims=vpot.dims)
   vag = xr.DataArray(vag, coords=vpot.coords, dims=vpot.dims)

   outds = outds.assign(variables=dict(UAG=uag, VAG=vag))
   outds.to_netcdf(path='QGomega.nc') 

   contourkwargs['levels'] = np.arange(-1000, 20001, 100)
   clabelkwargs['fmt'] = '%d'
   fig, ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent(BNDS)
   ax.coastlines(color='gray')
   cs1 = ax.contour(DS.lon, DS.lat, DS.Z.sel(level=ULEV).values, **contourkwargs)
   ax.clabel(cs1, **clabelkwargs)
   qv = plt.quiver(uag.lon, uag.lat, uag.sel(level=ULEV).values, vag.sel(level=ULEV).values, pivot='mid', scale=1e1, color='black')
   plt.quiverkey(qv, X=.75, Y=-0.1, U=1, label='1 m s$^{-1}$', labelpos='E')
   plt.title('%d hPa. Contours: $Z$ [m]; Vectors: $\\vec{v}_{ag}$ (QG approx.)' % ULEV)
   plt.savefig('hgt_ageo.png', bbox_inches='tight')
   plt.close()

def plotmap(da, plev, extent, title, cbarlabel, outfile, figsize=(10,7), cmap='BrBG', levels=15, norm=colors.TwoSlopeNorm(vcenter=0), contour=False, cntda=None, clabelkwargs=None, contourkwargs=None):
   fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent(extent)
   ax.coastlines()
   cs = ax.contourf(da.lon, da.lat, da.sel(level=plev).values, cmap=cmap, levels=levels, norm=norm)
   cbar = fig.colorbar(cs, shrink=0.7, orientation='horizontal')
   cbar.set_label(cbarlabel)
   if contour:
      cs1 = ax.contour(cntda.lon, cntda.lat, cntda.sel(level=MLEV).values, **contourkwargs)
      ax.clabel(cs1, **clabelkwargs)
   plt.title(title)
   plt.savefig(outfile, bbox_inches='tight')
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

def d_dp(var, dp):
   #simple vertical (pressure) derivative
   l = var.level.shape[0] - 1
   slcup = slice(2, l+1)
   slcdown = slice(0, l-1)
   return (var.values[slcdown,:,:] - var.values[slcup,:,:]) / (2*dp)

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

def lapmat(DS, latm=-1):
   #construct a matrix representation of the horizontal Laplacian
   l = DS.level.shape[0] - 1
   m = DS.lat.shape[0] + latm
   n = DS.lon.shape[0] - 1
   slcpmid = slice(1,l)
   slclatmid = slice(1,m)
   sz = (l-1)*(m+latm)*(n+1)
   rcs = np.arange(sz)
   iis, js, ks = idx1Dto3D(rcs, l, m, n)

   dxvec = DS.dx.values[None,slclatmid,None]
   if latm == 0:
      dxvec = DS.dx.values[None,:,None]
   dxvec = np.repeat(dxvec, l-1, axis=0)
   dxvec = np.repeat(dxvec, n+1, axis=2)
   #print(l, m, n, dxvec.shape)
   dxvec = np.reshape(dxvec, sz, order='C')

   L = np.zeros((sz,sz), dtype=np.float64)
   for r in range(sz):
      ir, jr, kr = iis[r], js[r], ks[r]
      for c in range(sz):
         ic, jc, kc = iis[c], js[c], ks[c]
         if ic == ir and kc == kr and abs(jc-jr) == 1:
            L[r,c] = 1/dy**2
         if ir == ic and jr == jc and abs(kc-kr) == 1:
            L[r,c] = 1/dxvec[r]**2
         if r == c:
            L[r,c] = -2/dxvec[r]**2 - 2/dy**2

         #handle periodic BCs
         if (kr == 0 and kc == n) or (kr == n and kc == 0):
            L[r,c] = 1/dxvec[r]**2

   return L

def matLHS(DS):
   #construct a matrix representation of the differential operators on the left-hand side of the QG omega equation
   l = DS.level.shape[0] - 1
   m = DS.lat.shape[0] - 1
   n = DS.lon.shape[0] - 1
   slcpmid = slice(1,l)
   slclatmid = slice(1,m)
   sz = (l-1)*(m-1)*(n+1)
   rcs = np.arange(sz)
   iis, js, ks = idx1Dto3D(rcs, l, m, n)
   #print(l, m, n)

   sigmavec = np.reshape(DS.sigma.values[slcpmid,slclatmid,:], sz, order='C')
   op3vec = np.reshape(DS.op3.values[slcpmid,slclatmid,:], sz, order='C')
   #dxvec = DS.dx.values[None,slclatmid,None]
   #dxvec = np.repeat(dxvec, l-1, axis=0)
   #dxvec = np.repeat(dxvec, n+1, axis=2)
   #dxvec = np.reshape(dxvec, sz, order='C')

   A = lapmat(DS) #np.zeros((sz,sz), dtype=np.float64)
   for r in range(sz):
      ir, jr, kr = iis[r], js[r], ks[r]
      for c in range(sz):
         ic, jc, kc = iis[c], js[c], ks[c]
         if jc == jr and kc == kr and abs(ic-ir) == 1:
            A[r,c] = (f0 / dp)**2 / sigmavec[r]
         if r == c:
            #print(f0, sigmavec[r], dp)
            #if ir == 0: #or ir == l-1: !UBC
            #   A[r,c] += (f0 / dp)**2/sigmavec[r]
            A[r,c] += -2*(f0 / dp)**2/sigmavec[r] + op3vec[r]

   return A

def idx1Dto3D(rc, l, m, n):
   k = rc % (n+1)
   j = 1 + rc // (n+1) % (m-1)
   i = 1 + rc // (n+1) // (m-1)
   return i, j, k

if __name__ == '__main__':
   main()
