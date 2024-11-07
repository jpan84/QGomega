#Joshua Pan jp872@cornell.edu 202411
#Generate the time-mean 3D Laplacian matrix for QG omega.
#Approx the time-average static stability using time-averaged temperature
#Boundary conditions: homogeneous at top and bottom, periodic in lon, Dirichlet in lat from actual omega
#Account for horizontal variations in static stability.

import xarray as xr
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs

DATADIR = './'
YEAR = '9999'
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
   DS = DS.sel(lat=slice(75., 27.4))
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

   print('Generating horizontal Laplacian matrix...')
   Lmat = lapmat(DS)
   print('Generating LHS matrix...')
   A = matLHS(DS, Lmat=Lmat)
   plt.spy(A[:200,:200])
   plt.savefig('matrix.png')
   plt.close()

   np.save('Lapmat.%s.npy' % YEAR, Lmat)
   np.save('QGomegaLHS_mat.%s.npy' % YEAR, A)

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

def matLHS(DS, Lmat=None):
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

   A = Lmat
   if A is None:
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
