import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors

qgomega = xr.open_dataset('qgomega.nc').OMEGAQG
slc = qgomega.sel(lat=30, lon=265)
plt.plot(slc.values, slc.level)
plt.xlabel('$\omega_{QG} \hspace{0.5} (Pa \hspace{0.5} s^{-1})$')
plt.ylabel('p (hPa)')
plt.yscale('log')
plt.gca().invert_yaxis()
plt.title('$\omega_{QG}$ %.2f°N, %.2f°E' % (slc.lat.values, slc.lon.values))
plt.savefig('omegaqgvsz.png')
plt.close()

qgforcing = xr.open_dataset('qgomega.nc').FORCING
slc = qgforcing.sel(lat=slc.lat.values, lon=slc.lon.values)
plt.plot(slc.values, slc.level)
plt.xlabel('QG forcings (DVA + TA) $(Pa \hspace{0.5} m^{-2} \hspace{0.5} s^{-1})$')
plt.ylabel('p (hPa)')
plt.yscale('log')
plt.gca().invert_yaxis()
plt.title('%.2f°N, %.2f°E' % (slc.lat.values, slc.lon.values))
plt.savefig('qgforcingvsz.png')
plt.close()

omega = xr.open_dataset('qgomega.nc').OMEGA
slc = omega.sel(lat=slc.lat.values, lon=slc.lon.values)
plt.plot(slc.values, slc.level)
plt.xlabel('$\omega \hspace{0.5} (Pa \hspace{0.5} s^{-1})$')
plt.ylabel('p (hPa)')
plt.yscale('log')
plt.gca().invert_yaxis()
plt.title('$\omega$ %.2f°N, %.2f°E' % (slc.lat.values, slc.lon.values))
plt.savefig('omegavsz.png')
plt.close()

slc = xr.open_dataset('qgomega.nc').OMEGAQG.sel(lat=37.5, lon = slice(260, 310))
plt.contourf(slc.lon.values, slc.level.values, slc.values, cmap='BrBG_r', levels=15, norm=colors.TwoSlopeNorm(vcenter=0))
plt.xlabel('Longitude (°)')
plt.ylabel('p (hPa)')
plt.title('$\omega_{QG} \hspace{0.5} [Pa \hspace{0.5} s^{-1}]$ %.1f°N %s' % (slc.lat.values, slc.time.values))
plt.yscale('log')
plt.gca().invert_yaxis()
plt.colorbar()
plt.savefig('xsect.png', bbox_inches='tight')
plt.close()
