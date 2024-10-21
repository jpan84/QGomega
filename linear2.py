#Joshua Pan jp872@cornell.edu 202112
#Turn the QG omega equation (forcing from diff vort adv and temp adv) into a linear system by discretizing derivatives. Interpolate NCEP/NCAR Reanalysis 1 to evenly spaced p levels.
#Boundary conditions: homogeneous Dirichlet in pressure, periodic in lon, Dirichlet in lat from actual omega
#For advection terms, assume vg = v.
#Update: corrected sigma^2 to sigma in denominator of LHS pressure derivative.

import xarray as xr
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs

DATADIR = './reanalysis/'
YEAR = '2016'

Rd = 287.06 #J kg-1 K-1
a = 6.371e6 #m
f0 = 1.031e-4 #s-1
cp = 1004 #J kg-1 K-1
OMEGArot = 7.29e-5 #s-1

dp = 7.5e3 #Pa
dy = a * 2.5*np.pi/180 #m
dlamb = 2.5*np.pi/180 #rad

ULEV = 550.
LLEV = 700.
BNDS = [220, 310, 30, 60]

def main():
   U = xr.open_dataset('%suwnd.%s.nc' % (DATADIR, YEAR)).uwnd
   V = xr.open_dataset('%svwnd.%s.nc' % (DATADIR, YEAR)).vwnd
   T = xr.open_dataset('%sair.%s.nc' % (DATADIR, YEAR)).air
   OMEGA = xr.open_dataset('%somega.%s.nc' % (DATADIR, YEAR)).omega
   DS = xr.Dataset(data_vars = {'U': U, 'V': V, 'T': T, 'OMEGA': OMEGA})
   DS = DS.sel(lat=slice(75., 27.4))
   DS = DS.reindex(lat = DS.lat.values[::-1]).fillna(0).sel(time = DS.time.values[90])
   DS = DS.interp(level=np.arange(1000,99,-dp/100))
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
   tatitle = 'QG $\omega$ Temp Adv Forcing %d hPa %s' % (int(LLEV), str(DS.time.values))
   plotmap(forceTA, LLEV, BNDS, tatitle, '$Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1}$', 'TAforcing.png')

   forceVA = qgVA(DS)
   vatitle = 'QG $\omega$ Diff Vort Adv Forcing %d hPa %s' % (int(ULEV), str(DS.time.values))
   plotmap(forceVA, ULEV, BNDS, vatitle, '$Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1}$', 'DVAforcing.png', norm=colors.TwoSlopeNorm(vcenter=0))

   qgforcing = forceTA + forceVA
   frctitle = 'QG $\omega$ Total Forcing (DVA + TA) %d hPa %s' % (int(ULEV), str(DS.time.values))
   plotmap(qgforcing, ULEV, BNDS, frctitle, '$Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1}$', 'QGforcing.png', norm=colors.TwoSlopeNorm(vcenter=0))

   l = qgforcing.shape[0] + 1
   m = qgforcing.shape[1] + 1
   n = qgforcing.shape[2] - 1
   forcevec = np.reshape(qgforcing.values, (l-1)*(m-1)*(n+1), order='C') #vectorize forcing terms (last axis changing fastest in level,lat,lon)
   print(qgforcing.shape)
   #handle the north/south boundary conditions
   for r in range(forcevec.shape[0]):
      i = 1 + r // (n+1) // (m-1)
      j = 1 + r // (n+1) % (m-1)
      k = r % (n+1)
      if j == 1:
         forcevec[r] -= DS.OMEGA.values[i, 0, k] / dy**2
      if j == m-1:
         forcevec[r] -= DS.OMEGA.values[i, m, k] / dy**2

   print('Generating A matrix...')
   A = matLHS(DS)
   plt.spy(A[:3000,:3000])
   plt.savefig('matrix.png')
   plt.close()
   print('Solving BVP...')
   omegavec = spsolve(A, forcevec)
   omegavec = np.reshape(omegavec, (l-1, m-1, n+1), order='C')
   print(omegavec.shape)

   qgomega = xr.DataArray(omegavec, coords = qgforcing.coords, dims = qgforcing.dims)
   omegatitle = 'Colors: $\omega_{QG}$ (forcing by DVA + TA) %d hPa %s\nContours: Reanalysis $\omega \hspace{0.5} [Pa \hspace{0.5} s^{-1}]$' % (int(ULEV), str(DS.time.values))
   clabelkwargs = {'inline': 1, 'fontsize': 10, 'colors': 'black', 'fmt': '%.1f'}
   contourkwargs = {'colors': 'black', 'transform': ccrs.PlateCarree(), 'levels': np.arange(-5, 5.1, 0.2)}
   plotmap(qgomega, ULEV, BNDS, omegatitle, '$Pa \hspace{0.5} s^{-1}$', 'QGomega.png', norm=colors.TwoSlopeNorm(vcenter=0), contour=True, cntda=DS.OMEGA, clabelkwargs=clabelkwargs, contourkwargs=contourkwargs)

   qgerror = qgomega - DS.OMEGA
   errtitle = 'QG Error (QG $\omega$ minus Reanalysis) %d hPa %s' % (int(ULEV), str(DS.time.values))
   plotmap(qgerror, ULEV, BNDS, errtitle, '$Pa \hspace{0.5} s^{-1}$', 'QGerror.png', cmap='bwr', norm=colors.TwoSlopeNorm(vcenter=0))

   outds = xr.Dataset(data_vars=dict(OMEGAQG=qgomega, OMEGA=DS.OMEGA, FORCING=qgforcing, FORCETA=forceTA, FORCEVA=forceVA))
   outds = outds.fillna(0)

   print('Computing ageo winds...')
   ompad = np.pad(qgomega, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0.)
   ompad = xr.DataArray(ompad, coords=[DS.level, DS.lat, DS.lon], dims=['level', 'lat', 'lon'])
   print(ompad.shape)
   diver = -d_dp(ompad, dp)
   print(diver.shape)
   divervec = np.reshape(diver, (l-1)*(m+1)*(n+1), order='C') #vectorize divergence (last axis changing fastest in level,lat,lon)
   Lmat = lapmat(DS, latm=1)
   vpot = spsolve(Lmat, divervec)
   vpot = np.reshape(vpot, (l-1, m+1, n+1), order='C')
   uag, vag = grad(vpot, DS.dx)

   outds = outds.assign(variables=dict(UAG=uag, VAG=vag))
   outds.to_netcdf(path='QGomega.nc') 


def plotmap(da, plev, extent, title, cbarlabel, outfile, figsize=(10,7), cmap='BrBG_r', levels=15, norm=colors.TwoSlopeNorm(vcenter=0), contour=False, cntda=None, clabelkwargs=None, contourkwargs=None):
   fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
   ax.set_extent(extent)
   ax.coastlines()
   cs = ax.contourf(da.lon, da.lat, da.sel(level=plev).values, cmap=cmap, levels=levels, norm=norm)
   cbar = fig.colorbar(cs, shrink=0.7, orientation='horizontal')
   cbar.set_label(cbarlabel)
   if contour:
      cs1 = ax.contour(cntda.lon, cntda.lat, cntda.sel(level=ULEV).values, **contourkwargs)
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

   dxvec = DS.dx.values[None,slclatmid,None]
   if latm == 1:
      dxvec = DS.dx.values
   dxvec = np.repeat(dxvec, l-1, axis=0)
   dxvec = np.repeat(dxvec, n+1, axis=2)
   dxvec = np.reshape(dxvec, sz, order='C')

   L = np.zeros((sz,sz), dtype=np.float64)
   for r in range(sz):
      ir, jr, kr = idx1Dto3D(r, l, m, n)
      for c in range(sz):
         ic, jc, kc = idx1Dto3D(c, l, m, n)
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

   sigmavec = np.reshape(DS.sigma.values[slcpmid,slclatmid,:], sz, order='C')
   dxvec = DS.dx.values[None,slclatmid,None]
   dxvec = np.repeat(dxvec, l-1, axis=0)
   dxvec = np.repeat(dxvec, n+1, axis=2)
   dxvec = np.reshape(dxvec, sz, order='C')

   A = lapmat(DS) #np.zeros((sz,sz), dtype=np.float64)
   for r in range(sz):
      ir, jr, kr = idx1Dto3D(r, l, m, n)
      for c in range(sz):
         ic, jc, kc = idx1Dto3D(c, l, m, n)
         if jc == jr and kc == kr and abs(ic-ir) == 1:
            A[r,c] = (f0 / dp)**2 / sigmavec[r]
         if r == c:
            #print(f0, sigmavec[r], dp)
            A[r,c] += -2*(f0 / dp)**2/sigmavec[r]

   return A

def idx1Dto3D(rc, l, m, n):
   k = rc % (n+1)
   j = 1 + rc // (n+1) % (m-1)
   i = 1 + rc // (n+1) // (m-1)
   return i, j, k

if __name__ == '__main__':
   main()
