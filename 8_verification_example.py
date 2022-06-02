#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Generate Fig S2 showing harvest, eMapR, and LEMMA maps + timeseries for example project CAR1066

"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scalebar import scale_bar
from matplotlib.ticker import FixedLocator

root = '/Users/scoffiel/california/offsets/'


projects = ShapelyFeature(Reader(root + "processed_data/shapefiles/all_projects/activeCAprojects_vectorized_4326.shp").geometries(), ccrs.PlateCarree())

emapr = xr.open_rasterio(root + 'processed_data/carbon_emapr/emapr_car1066.tiff')
lemma = xr.open_rasterio(root + 'processed_data/carbon_lemma/lemma_car1066.tiff')
harvest = xr.open_rasterio(root + 'processed_data/harvest/harvest_car1066.tiff')

for band in harvest.band:
    harvest.loc[band,:,:] = harvest.loc[band,:,:] * (band+1985)

harvest = harvest.max(dim='band')
#harvest.rio.to_raster(root + 'processed_data/harvest/harvest_car1066_years.tif')

#first panel: map of change plot with offsets overlaid
fig = plt.figure(figsize=(7,10))

grid = gs.GridSpec(3,2, width_ratios=[5,4])

ax = plt.subplot(grid[0], projection = ccrs.Miller())
ax2 = plt.subplot(grid[1])
ax3 = plt.subplot(grid[2], projection = ccrs.Miller())
ax4 = plt.subplot(grid[3])
ax5 = plt.subplot(grid[4], projection=ccrs.Miller())
ax6 = plt.subplot(grid[5])

#harvest['x'] = harvest.x + 360

extent = [-122.07711994295283,-121.61157428865596, 41.25074488861626, 41.486755519637505] #left, right, bottom, top. Match GEE export

#ax1 - harvest map
ax.set_extent([238.09,238.21,43.83,43.96], crs=ccrs.Miller())
ax.add_feature(projects, edgecolor='k', facecolor='none')
plot = ax.imshow(harvest, extent=extent, cmap='pink_r', vmin=1986, vmax=2021, transform=ccrs.PlateCarree())
cbar = plt.colorbar(plot, orientation='vertical', shrink=0.65, pad=0.03, ax=ax) #.65 and .06 for horizontal; 0.8 and .01 for vert 
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Year of last harvest', size=10)
scale_bar(ax, (0.3, 0.03), 2)
ax.text(0,1.03,'(a)',fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.set_title('Harvest history')


#ax2 - harvest timeseries for car1066
harvest_projects = pd.read_csv(root + 'processed_data/harvest/projects_harvest.csv', index_col='project_id') #b2-b37, 1986-2021 (36 yrs)

y =[] #projects

for yr in range(2,38):
    y.append(harvest_projects.loc['CAR1066','b{}'.format(yr)]*100)

ax2.plot(range(1986,2022), y, color='dimgray', linewidth=2)
ax2.set_xlim((1985,2022))
ax2.set_ylim((0,7.1))
ax2.text(-0.25,1.03,'(b)',fontsize=12, fontweight='bold', transform=ax2.transAxes)
ax2.set_ylabel('Fraction of area harvested')
ax2.xaxis.set_minor_locator(FixedLocator(range(1990,2022,5)))
ax2.grid(zorder=0, linewidth=0.4, color='0.9', which='both')
ax2.text(0.7,0.9,'Harvest',fontsize=12, transform=ax2.transAxes)
import matplotlib.ticker as mtick
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))


#ax3 - emapr map
ax3.set_extent([238.09,238.21,43.83,43.96], crs=ccrs.Miller())
ax3.add_feature(projects, edgecolor='k', facecolor='none')
plot = ax3.imshow(emapr[-1,:,:]*0.47, extent=extent, cmap='YlGn', vmin=0, vmax=200, transform=ccrs.PlateCarree())
cbar = plt.colorbar(plot, orientation='vertical', shrink=0.65, pad=0.03, ax=ax3) #.65 and .06 for horizontal; 0.8 and .01 for vert 
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Carbon density (tonC/ha)', size=10)
ax3.set_title('eMapR (2017)')
ax3.text(0,1.03,'(c)',fontsize=12, fontweight='bold', transform=ax3.transAxes)


#ax4 - emapr timeseries for car1066
emapr_projects = pd.read_csv(root + 'processed_data/carbon_emapr/projects_emapr.csv', index_col='project_id')

y =[] #projects

for yr in range(3,35):
    y.append(emapr_projects.loc['CAR1066','b{}'.format(yr)]*0.47)

ax4.plot(range(1986,2018), y, color='dimgray', linewidth=2)
ax4.set_xlim((1985,2022))
ax4.set_ylim((0,115))
ax4.text(-0.25,1.03,'(d)',fontsize=12, fontweight='bold', transform=ax4.transAxes)
ax4.text(0.7,0.9,'eMapR',fontsize=12, transform=ax4.transAxes)
ax4.set_ylabel('Carbon density (tonC/ha)')
ax4.xaxis.set_minor_locator(FixedLocator(range(1990,2022,5)))
ax4.grid(zorder=0, linewidth=0.4, color='0.9', which='both')


#ax5 - lemma map
ax5.set_extent([238.09,238.21,43.83,43.96], crs=ccrs.Miller())
ax5.add_feature(projects, edgecolor='k', facecolor='none')
plot = ax5.imshow(lemma[-1,:,:]*0.47/1000, extent=extent, cmap='YlGn', vmin=0, vmax=200, transform=ccrs.PlateCarree())
cbar = plt.colorbar(plot, orientation='vertical', shrink=0.65, pad=0.03, ax=ax5) #.65 and .06 for horizontal; 0.8 and .01 for vert 
cbar.ax.tick_params(labelsize=8)
cbar.set_label('Carbon density (tonC/ha)', size=10)
ax5.set_title('LEMMA (2017)')
ax5.text(0,1.03,'(e)',fontsize=12, fontweight='bold', transform=ax5.transAxes)


#ax6 - lemma timeseries for car1066
lemma_projects = pd.read_csv(root + 'processed_data/carbon_lemma/projects_lemma.csv', index_col='project_id')

y =[] #projects

for yr in range(1986,2018):
    y.append(lemma_projects.loc['CAR1066','{}_b1'.format(yr)]*0.47/1000)

ax6.plot(range(1986,2018), y, color='dimgray', linewidth=2)
ax6.set_xlim((1985,2022))
ax6.set_ylim((0,115))
ax6.text(-0.25,1.03,'(f)',fontsize=12, fontweight='bold', transform=ax6.transAxes)
ax6.text(0.7,0.9,'LEMMA',fontsize=12, transform=ax6.transAxes)
ax6.set_ylabel('Carbon density (tonC/ha)')
ax6.xaxis.set_minor_locator(FixedLocator(range(1990,2022,5)))
ax6.grid(zorder=0, linewidth=0.4, color='0.9', which='both')

plt.tight_layout()


#second figure looking at recovery for pixels that were harvested in 1989
emapr = emapr.sel(x=slice(-121.91,-121.79), y=slice(41.43,41.34)) #narrow to project
lemma = lemma.sel(x=slice(-121.91,-121.79), y=slice(41.43,41.34))
harvest = harvest.sel(x=slice(-121.91,-121.79), y=slice(41.43,41.34))

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(221)
harvest = xr.open_rasterio(root + 'processed_data/harvest/harvest_car1066.tiff')
h1990 = harvest[4,:,:]
emapr_recovery = emapr.where(h1990==1) * 0.47
emapr_recovery = emapr_recovery.mean(dim=['x','y'])
ax.plot(range(1986,2018), emapr_recovery, c='green', label='eMapR')

lemma_recovery = lemma.where(h1990==1) *0.47/1000
lemma_recovery = lemma_recovery.mean(dim=['x','y'])
ax.plot(range(1986,2018), lemma_recovery, c='blue', label='LEMMA')
ax.set_ylabel('Carbon density (ton C/ha)')
ax.set_title('Pixels harvested in 1989')
ax.text(-0.15,1.03,'(g)',fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.set_ylim((0,110))
ax.legend()

#mean rate of change for all pixels, pixels in recovery after 1986, and undisturbed pixels
undist = harvest.sum(dim='band')==0
emapr_undist = emapr.where(undist) * 0.47
emapr_undist_trend = emapr_undist.mean(dim=['x','y'])
lemma_undist = lemma.where(undist) * 0.47/1000
lemma_undist_trend = lemma_undist.mean(dim=['x','y'])


ax2 = fig.add_subplot(222)
ax2.plot(range(1986,2018), emapr_undist_trend, c='green', label='eMapR')
ax2.plot(range(1986,2018), lemma_undist_trend, c='blue', label='LEMMA')
ax2.set_title('Non-harvested pixels')
ax2.text(-0.15,1.03,'(h)',fontsize=12, fontweight='bold', transform=ax2.transAxes)
ax2.set_ylim((0,110))
ax2.legend()

ax3 = fig.add_subplot(223)
emapr_recovery = emapr.where(h1990==1)[5:,:,:] * 0.47
emapr_recovery = emapr_recovery.values.reshape(27, -1)
emapr_recovery = emapr_recovery[~np.isnan(emapr_recovery)].reshape(27, 4176)
emapr_trend = np.polyfit(range(1991,2018), emapr_recovery, deg=1)
counts, bins = np.histogram(emapr_trend[0,:], bins=np.arange(-6,6,0.5))
ax3.bar(bins[:-1], counts, width=0.5, alpha=0.5, align='edge', color='green', label='eMapR')
ax3.axvline(emapr_trend[0,:].mean(), color='green', linestyle='--')

lemma_recovery = lemma.where(h1990==1)[5:,:,:] * 0.47/1000
lemma_recovery = lemma_recovery.values.reshape(27, -1)
lemma_recovery = lemma_recovery[~np.isnan(lemma_recovery)].reshape(27, 4176)
lemma_trend = np.polyfit(range(1991,2018), lemma_recovery, deg=1)
counts, bins = np.histogram(lemma_trend[0,:], bins=np.arange(-6,6,0.5))
ax3.bar(bins[:-1], counts, width=0.5, alpha=0.5, align='edge', color='blue', label='LEMMA')
ax3.axvline(lemma_trend[0,:].mean(), color='blue', linestyle='--')

ax3.text(-0.15,1.03,'(i)',fontsize=12, fontweight='bold', transform=ax3.transAxes)
ax3.set_ylabel('Frequency')
ax3.set_xlabel('Recovery rate (tonC/ha/y)')
ax3.legend()


ax4 = fig.add_subplot(224)

emapr_undist = emapr_undist.values.reshape(32, -1)
emapr_undist = emapr_undist[~np.isnan(emapr_undist)].reshape(32, 93649)
emapr_trend = np.polyfit(range(1986,2018), emapr_undist, deg=1)
counts, bins = np.histogram(emapr_trend[0,:], bins=np.arange(-6,6,0.5))
ax4.bar(bins[:-1], counts, width=0.5, alpha=0.5, align='edge', color='green', label='eMapR')
ax4.axvline(emapr_trend[0,:].mean(), color='green', linestyle='--')

lemma_undist = lemma_undist.values.reshape(32, -1)
lemma_undist = lemma_undist[~np.isnan(lemma_undist)].reshape(32, 93649)
lemma_trend = np.polyfit(range(1986,2018), lemma_undist, deg=1)
counts, bins = np.histogram(lemma_trend[0,:], bins=np.arange(-6,6,0.5))
ax4.bar(bins[:-1], counts, width=0.5, alpha=0.5, align='edge', color='blue', label='LEMMA')
ax4.axvline(lemma_trend[0,:].mean(), color='blue', linestyle='--')

ax4.text(-0.15,1.03,'(j)',fontsize=12, fontweight='bold', transform=ax4.transAxes)
ax4.set_xlabel('Growth rate (tonC/ha/y)')
ax4.legend()
