import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

lat0 = np.deg2rad(45)
g = 9.81
a = 6.371e6
p0 = 1e5
Rd = 287
cp = 1004
kap = Rd / cp
OM = 7.29e-5
f0 = 2 * OM * np.sin(lat0)
N2 = 2e-4
T0 = 280
dens0 = p0 / Rd / T0
#print('dens0', dens0)

lons = np.linspace(0, 2 * np.pi, 100)
yy = np.linspace(-500, 500, 100) * 1e3
pp = np.arange(2.5e4, 1.e5 + 1, 2.5e3)

abc = np.array([-1, 1.4e5, -2.875e9]) / 2.3e9
#print(abc)

Zamp = 200
Tamp = 10
Zlevs = np.arange(60, 360, 60)
Zlevs = np.concatenate((-Zlevs[::-1], Zlevs))
Ulevs = np.arange(10, 100, 10)
vlevs = np.arange(5, 30, 5)
vlevs = np.concatenate((-vlevs[::-1], vlevs))
Tlevs = np.arange(2, 11, 2)
Tlevs = np.concatenate((-Tlevs[::-1], Tlevs))
plabs = np.array([300, 400, 500, 600, 700, 850, 1000])
psilevs = 1e10 * 2.**np.arange(0, 10)
thlevs = np.arange(250, 400, 5)

def main():
   #print(lons, yy, pp)
   xg, yg, pg = np.meshgrid(lons * a * np.cos(lat0), yy, pp, indexing='ij')
   #print(pg[..., -1])

   #print(f(xg))
   #print(h(pg[1,1,:]))

   Tp = Tamp * 1j * f(xg) * g1(yg) * h(pg)
   Z0 = Zamp * f(xg) * g1(yg)
   Zp = Z0 + f(xg) * g1(yg) * Rd * Tamp / g * 1j * h_int(pg)
   vp = g / f0 * diffx(Zp)
   print(vp.max())

   #x-p plane v, T, Z
   plt.rcParams['figure.figsize'] = (12, 6)
   yslc = (slice(None), 50, slice(None))
   csf = plt.contourf(xg[yslc], pg[yslc], Tp[yslc], cmap='bwr')
   plt.contour(xg[yslc], pg[yslc], Zp[yslc], levels=Zlevs, colors='black')
   plt.contour(xg[yslc], pg[yslc], vp[yslc], levels=vlevs, colors='green')
   plt.xlabel('x [m]')
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig('xp_v_T_Z.png')
   plt.close()

   #x-p plane EHF
   vpTp = vp.real * Tp.real
   plt.contourf(xg[yslc], pg[yslc], vpTp[yslc], cmap='bwr', norm=colors.TwoSlopeNorm(0))
   plt.xlabel('x [m]')
   plt_paxis_adj()
   plt.colorbar()
   #plt.show()
   plt.savefig('xp_vT.png')
   plt.close()

   #x-y plane Z, EHF
   #print(pg[..., 18])
   pslc = (slice(None), slice(None), 18)
   csf = plt.contourf(xg[pslc], yg[pslc], vpTp[pslc], cmap='bwr', norm=colors.TwoSlopeNorm(0))
   plt.contour(xg[pslc], yg[pslc], Zp[pslc], levels=Zlevs, colors='black')
   plt.colorbar(csf)
   plt.savefig('xy_Z_vT.png')
   plt.close()

   DT_EHF = -EHFd(vpTp, yg)
   PSI_vT = g / T0 / N2 * vpTp.mean(axis=0) * 2 * np.pi * a * np.cos(lat0)
   print(PSI_vT.max())
   #y-p plane EHF heating rate, streamfunc response
   csf = plt.contourf(yg[0, ...], pg[0, ...], DT_EHF.mean(axis=0), cmap='bwr')
   plt.contour(yg[0, ...], pg[0, ...], PSI_vT, levels=psilevs, colors='black')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('yp_DTEHF.png')
   plt.close()

   #y-p plane streamfunc response, adiab DT
   DT_ADIAB = -N2 * T0 / g * EHFd(PSI_vT[None, ...], yg) / 2 / np.pi / a / np.cos(lat0) #= w N^2 T0/g
   csf = plt.contourf(yg[0, ...], pg[0, ...], DT_ADIAB.mean(axis=0), cmap='bwr')
   plt.contour(yg[0, ...], pg[0, ...], PSI_vT, levels=psilevs, colors='black')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('yp_DTADIAB.png')
   plt.close()

   #zonal-mean temp advection
   DT_ADV = -(vp.real * diffy(Tp.real, yg)).mean(axis=0)
   plt.contourf(yg[0, ...], pg[0, ...], DT_ADV, cmap='bwr')
   plt_paxis_adj()
   plt.colorbar()
   plt.savefig('yp_DTADV.png')
   plt.close()

   th_vert = T0 * (1 + N2 / dens0 / g**2 * (p0 - pg))
   th_bg = th_vert * (1 + g2(yg))
   th_y = th_vert * -1e4 * diff2_g1sq(yg) #= th_bg * d/dy(1e4 * (4 / 1e6**2 * yg) * g1(y)**2)
   U_consts = Rd * T0 * N2 / f0 / p0**kap / dens0 / g**2 * -1e4 * diff2_g1sq(yg)
   U_bg = U_consts * (T0 * (pg - p0) + (p0 / kap * (pg**kap - p0**kap) - (pg**(kap + 1) - p0**(kap + 1)) / (kap + 1)))
   print(U_bg.min(), U_bg.max())
   #y-p plane pot temp, its gradient, and U background
   csf = plt.contourf(yg[0, ...], pg[0, ...], th_y[0, ...], cmap='bwr')
   plt.contour(yg[0, ...], pg[0, ...], th_bg[0, ...], levels=thlevs, colors='red')
   plt.contour(yg[0, ...], pg[0, ...], U_bg[0, ...], levels=Ulevs, colors='black')
   plt_paxis_adj()
   plt.colorbar(csf)
   plt.savefig('yp_bg.png')
   plt.close()

   Tadv_bg = -U_bg * diffx(Tp)
   #x-p plane advection of wave temp by background flow
   csf = plt.contourf(xg[yslc], pg[yslc], Tadv_bg[yslc], cmap='bwr')
   plt.contour(xg[yslc], pg[yslc], Tp[yslc], levels=Tlevs, colors='red')
   plt.xlabel('x [m]')
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig('xp_Tadv_bg.png')
   plt.close()

   rv =  g / f0 * (diffx(diffx(Zp)) + f(xg) * diff2_g1(yg) * (Zamp + Rd * Tamp / g * 1j * h_int(pg)))
   #x-p plane relvort, Z
   csf = plt.contourf(xg[yslc], pg[yslc], rv[yslc], cmap='bwr')
   plt.contour(xg[yslc], pg[yslc], Zp[yslc], levels=Zlevs, colors='black')
   plt.xlabel('x [m]')
   plt_paxis_adj()
   plt.colorbar(csf)
   #plt.show()
   plt.savefig('xp_rv_Z.png')
   plt.close()

   #x-y plane Z, EHF
   #print(pg[..., 0])
   pslc = (slice(None), slice(None), 0)
   csf = plt.contourf(xg[pslc], yg[pslc], rv[pslc], cmap='bwr')
   plt.contour(xg[pslc], yg[pslc], Zp[pslc], levels=Zlevs, colors='black')
   plt.colorbar(csf)
   plt.savefig('xy_rv_Z.png')
   plt.close()

def f(x):
   return np.exp(4j * x / a / np.cos(lat0))

def diffx(fn):
   return 4j / a / np.cos(lat0) * fn

def g1(y):
   return np.exp(-(y / 1e6)**2)

def g2(y):
   return 1e4 * EHFd(g1(y)**2, y)

#y derivative of squared perturbations
def EHFd(ehf, y):
   return -2 / 1e6**2 * 2 * y * ehf

def diffy(fn, y):
   return EHFd(fn, y) / 2

def diff2_g1(y):
   return -2 / 1e6**2 * (g1(y) + y * diffy(g1(y), y))

#2nd y derivative of squared Gaussian
def diff2_g1sq(y):
   return 4 / 1e6**2 * (g1(y)**2 + y * 2 * g1(y) * diffy(g1(y), y))

def h(p):
   return np.dot(abc, np.array([p**2, p, 1]))

def h_int(p):
   #return np.dot(abc, np.array([(p0**2 - p**2) / 2, p0 - p, np.log(p0 / p)]))
   return np.einsum('i,i...->...', abc, np.array([(p0**2 - p**2) / 2, p0 - p, np.log(p0 / p)]))

def plt_paxis_adj():
   plt.yscale('log')
   plt.yticks(100 * plabs, labels=plabs)
   plt.gca().invert_yaxis()
   plt.ylabel('p [hPa]')

if __name__ == '__main__':
   main()
