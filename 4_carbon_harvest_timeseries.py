#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose:
- Plot Figure 4 with spatial comparisons framework:
- Show coastal vs. interior projects and surroundings
- Show timeseries for carbon and harvest for projects, surroundings, and regions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scipy import stats

root = '/Users/scoffiel/california/offsets/'

carbon_data = 'emapr' #choose emapr or lemma

#read in shapefiles for map panels
interior_projects = ShapelyFeature(Reader(root + "processed_data/shapefiles/interior/interior_projects_4326.shp").geometries(), ccrs.PlateCarree())
coast_projects = ShapelyFeature(Reader(root + "processed_data/shapefiles/coast/coast_projects_4326_vectorized.shp").geometries(), ccrs.PlateCarree())

coast_surround = ShapelyFeature(Reader(root + "processed_data/shapefiles/coast/coast_surround.shp").geometries(), ccrs.PlateCarree())
interior_surround = ShapelyFeature(Reader(root + "processed_data/shapefiles/interior/interior_surround.shp").geometries(), ccrs.PlateCarree())

coast = ShapelyFeature(Reader(root + "processed_data/shapefiles/coast/coast.shp").geometries(), ccrs.PlateCarree())
interior = ShapelyFeature(Reader(root + "processed_data/shapefiles/interior/interior.shp").geometries(), ccrs.PlateCarree())
states = ShapelyFeature(Reader(root + "shapefiles/states/cb_2018_us_state_20m.shp").geometries(), ccrs.PlateCarree())


#first panel: map of change plot with offsets overlaid
fig = plt.figure(figsize=(7,10))

grid = gs.GridSpec(3,2, height_ratios=[3,2,2])

ax = plt.subplot(grid[0], projection = ccrs.Miller())
ax2 = plt.subplot(grid[1], projection = ccrs.Miller())
ax3 = plt.subplot(grid[2])
ax4 = plt.subplot(grid[3])
ax5 = plt.subplot(grid[4])
ax6 = plt.subplot(grid[5])

#ax1 - coast map
ax.set_extent([235.5,240.2,39.5,44.7], crs=ccrs.Miller())
ax.add_feature(coast, edgecolor='none', facecolor='darkgray', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2', facecolor='none')
ax.add_feature(coast_projects, edgecolor='none', facecolor='red')
ax.add_feature(coast_surround, edgecolor='none', facecolor='k', linewidth=0.2)

ax.text(-0.2,1,'(a)',fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.set_xticks([236,238,240], crs=ccrs.PlateCarree())
ax.set_yticks([38,40,42], crs=ccrs.PlateCarree())
ax.set_yticklabels([38,40,42], fontsize=8)
ax.tick_params(top=True, right=True, labelsize=8)
ax.text(-124.86,39.55,'$^\circ$N', size=9)
ax.text(-124.54,39.3,'$^\circ$E', size=9)
ax.set_title('Coastal projects')

# add legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], marker='h', color='w', label='Projects', markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='h', color='w', label='Surroundings', markerfacecolor='k', markersize=10),
                   Patch(facecolor='darkgray', edgecolor='darkgray',label='Coastal region')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)



#ax2 - interior map
ax2.set_extent([235.5,240.2,39.5,44.7], crs=ccrs.Miller())
ax2.add_feature(interior, edgecolor='none', facecolor='darkgray', linewidth=0.2)
ax2.add_feature(states, edgecolor='0.2', facecolor='none')
ax2.add_feature(interior_projects, edgecolor='none', facecolor='red')
ax2.add_feature(interior_surround, edgecolor='none', facecolor='k', linewidth=0.2)

ax2.text(-0.2,1,'(b)',fontsize=12, fontweight='bold', transform=ax2.transAxes)
ax2.set_xticks([236,238,240], crs=ccrs.PlateCarree())
ax2.set_yticks([38,40,42], crs=ccrs.PlateCarree())
ax2.set_yticklabels([38,40,42], fontsize=8)
ax2.tick_params(top=True, right=True, labelsize=8)
ax2.text(-124.86,39.55,'$^\circ$N', size=9)
ax2.text(-124.54,39.3,'$^\circ$E', size=9)
ax2.set_title('Interior projects')

# add legend
legend_elements = [Line2D([0], [0], marker='h', color='w', label='Projects', markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='h', color='w', label='Surroundings', markerfacecolor='k', markersize=10),
                   Patch(facecolor='darkgray', edgecolor='darkgray',label='Interior region')]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=8)


#choose either emapr or lemma above
#ax3 coast: compare year-by-year timeseries for EMAPR for combined polygons surround vs projects
carbon_coast_projects = pd.read_csv(root + 'processed_data/carbon_{}/coast_projects_{}.csv'.format(carbon_data, carbon_data))
carbon_coast_surround = pd.read_csv(root + 'processed_data/carbon_{}/coast_surroundings_{}.csv'.format(carbon_data, carbon_data))
carbon_coast = pd.read_csv(root + 'processed_data/carbon_{}/coast_{}.csv'.format(carbon_data, carbon_data))
projects = pd.read_csv(root + 'all_projects.csv')
projects = projects[~projects.project_id.isin(['CAR1066','CAR1041','CAR1092','CAR1114'])] #exclude 4 interior projects

y1 =[] #projects
y2 =[] #surround
y3 =[] #region
if carbon_data=='emapr':
    for yr in range(3,35):
        y1.append(carbon_coast_projects.loc[0,'b{}'.format(yr)]*0.47)
        y2.append(carbon_coast_surround.loc[0,'b{}'.format(yr)]*0.47)
        y3.append(carbon_coast.loc[0,'b{}'.format(yr)]*0.47)
if carbon_data=='lemma':
    for yr in range(1986,2018):
        y1.append(carbon_coast_projects.loc[0,'{}_b1'.format(yr)]*0.47/1000)
        y2.append(carbon_coast_surround.loc[0,'{}_b1'.format(yr)]*0.47/1000)
        y3.append(carbon_coast.loc[0,'{}_b1'.format(yr)]*0.47/1000)

print('mean and stderr of carbon, projects:', np.mean(y1), np.std(y1)/np.sqrt(len(y1)))
print('mean and stderr of carbon, surroundings:', np.mean(y2), np.std(y2)/np.sqrt(len(y2)))
print('mean and stderr of carbon, region:', np.mean(y3), np.std(y3)/np.sqrt(len(y3)))

print('slope and stderr of carbon, projects:', stats.linregress(range(len(y1)), y1).slope, stats.linregress(range(len(y1)), y1).stderr)
print('slope and stderr of carbon, surroundings:', stats.linregress(range(len(y2)), y2).slope, stats.linregress(range(len(y2)), y2).stderr)
print('slope and stderr of carbon, region:', stats.linregress(range(len(y3)), y3).slope, stats.linregress(range(len(y3)), y3).stderr)

ax3b = ax3.twinx()  #instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax3b.hist(projects.start_year, bins=range(2012,2020))
ax3b.set_yticks([0,2,4,6,8])
ax3b.set_ylim((0,22))
ax3b.tick_params(axis='y', labelcolor=color)

ax3.plot(range(1986,2018), y1, color='red', linewidth=2, label='Projects')
ax3.plot(range(1986,2018), y2, color='black', linewidth=2, label='Surroundings')
ax3.plot(range(1986,2018), y3, color='gray', linewidth=2, label='Coastal region')

if carbon_data=='emapr': label='eMapR'
else: label='LEMMA'
ax3.set_ylabel('Carbon (ton C/ha) ({})'.format(label)) #lemma or emapr
ax3.set_xlim((1985,2022))
ax3.set_ylim((0,149))
ax3.text(-0.12,1,'(c)',fontsize=12, fontweight='bold', transform=ax3.transAxes)
ax3.grid(zorder=0, linewidth=0.4, color='0.9')
ax3.legend(fontsize=8)


#ax4 interior: compare year-by-year timeseries for EMAPR for combined polygons surround vs projects
carbon_interior_projects = pd.read_csv(root + 'processed_data/carbon_{}/interior_projects_{}.csv'.format(carbon_data, carbon_data))
carbon_interior_surround = pd.read_csv(root + 'processed_data/carbon_{}/interior_surroundings_{}.csv'.format(carbon_data, carbon_data))
carbon_interior = pd.read_csv(root + 'processed_data/carbon_{}/interior_{}.csv'.format(carbon_data, carbon_data))
projects = pd.read_csv(root + 'all_projects.csv')
projects = projects[projects.project_id.isin(['CAR1066','CAR1041','CAR1092','CAR1114'])]

y1 =[] #projects
y2 =[] #surround
y3 =[] #region
if carbon_data=='emapr':
    for yr in range(3,35):
        y1.append(carbon_interior_projects.loc[0,'b{}'.format(yr)]*0.47)
        y2.append(carbon_interior_surround.loc[0,'b{}'.format(yr)]*0.47)
        y3.append(carbon_interior.loc[0,'b{}'.format(yr)]*0.47)
if carbon_data=='lemma':
    for yr in range(1986,2018):
        y1.append(carbon_interior_projects.loc[0,'{}_b1'.format(yr)]*0.47/1000)
        y2.append(carbon_interior_surround.loc[0,'{}_b1'.format(yr)]*0.47/1000)
        y3.append(carbon_interior.loc[0,'{}_b1'.format(yr)]*0.47/1000)

print('mean and stderr of carbon, projects:', np.mean(y1), np.std(y1)/np.sqrt(len(y1)))
print('mean and stderr of carbon, surroundings:', np.mean(y2), np.std(y2)/np.sqrt(len(y2)))
print('mean and stderr of carbon, region:', np.mean(y3), np.std(y3)/np.sqrt(len(y3)))

print('slope and stderr of carbon, projects:', stats.linregress(range(len(y1)), y1).slope, stats.linregress(range(len(y1)), y1).stderr)
print('slope and stderr of carbon, surroundings:', stats.linregress(range(len(y2)), y2).slope, stats.linregress(range(len(y2)), y2).stderr)
print('slope and stderr of carbon, region:', stats.linregress(range(len(y3)), y3).slope, stats.linregress(range(len(y3)), y3).stderr)

ax4b = ax4.twinx()
color = 'tab:blue'
ax4b.set_ylabel('New projects', color=color, position=(0,0), ha='left')
ax4b.hist(projects.start_year, bins=range(2012,2020))
ax4b.set_yticks([0,2,4,6,8])
ax4b.set_ylim((0,22))
ax4b.tick_params(axis='y', labelcolor=color)

ax4.plot(range(1986,2018), y1, color='red', linewidth=2, label='Projects')
ax4.plot(range(1986,2018), y2, color='black', linewidth=2, label='Surroundings')
ax4.plot(range(1986,2018), y3, color='gray', linewidth=2, label='Interior region')

ax4.set_xlim((1985,2022))
ax4.set_ylim((0,149))
ax4.text(-0.12,1,'(d)',fontsize=12, fontweight='bold', transform=ax4.transAxes)
ax4.grid(zorder=0, linewidth=0.4, color='0.9')
ax4.legend(fontsize=8)


#ax5 coast: compare year-by-year timeseries for HARVEST for combined polygons surroundings vs projects
harvest_coast_projects = pd.read_csv(root + 'processed_data/harvest/coast_projects_harvest.csv')
harvest_coast_surround = pd.read_csv(root + 'processed_data/harvest/coast_surroundings_harvest.csv')
harvest_coast = pd.read_csv(root + 'processed_data/harvest/coast_harvest.csv')

y1 =[] #projects
y2 =[] #surroundings
y3 =[] #region
for yr in range(2,38): 
    y1.append(harvest_coast_projects.loc[0,'b{}'.format(yr)]*100)
    y2.append(harvest_coast_surround.loc[0,'b{}'.format(yr)]*100)
    y3.append(harvest_coast.loc[0,'b{}'.format(yr)]*100)
      
ax5.plot(range(1986,2022), y1, color='red', linewidth=2, label='Projects')
ax5.plot(range(1986,2022), y2, color='black', linewidth=2, label='Surroundings')
ax5.plot(range(1986,2022), y3, color='gray', linewidth=2, label='Coastal region')
ax5.set_ylabel('Fraction of area harvested')
ax5.set_xlim((1985,2022))
ax5.set_ylim((0,4.6))
ax5.text(-0.12,1,'(e)',fontsize=12, fontweight='bold', transform=ax5.transAxes)
ax5.legend(fontsize=8)
ax5.set_xlabel('Year')
ax5.grid(zorder=0, linewidth=0.4, color='0.9')
import matplotlib.ticker as mtick
ax5.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

y1 = np.array(y1[:-9]) #1986-2012
y2 = np.array(y2[:-9])
y3 = np.array(y3[:-9])

print('harvest, projects vs. surroundings', ((y1 - y2)/ y2).mean())
print(stats.ttest_rel(y1, y2))
print('harvest, projects vs. coast', ((y1 - y3)/ y3).mean())
print(stats.ttest_rel(y1, y3))


#ax6 interior: compare year-by-year timeseries for HARVEST for combined polygons surroundings vs projects
harvest_interior_projects = pd.read_csv(root + 'processed_data/harvest/interior_projects_harvest.csv')
harvest_interior_surround = pd.read_csv(root + 'processed_data/harvest/interior_surroundings_harvest.csv')
harvest_interior = pd.read_csv(root + 'processed_data/harvest/interior_harvest.csv')

y1 =[] #projects
y2 =[] #surroundings
y3 =[] #region
for yr in range(2,38):
    y1.append(harvest_interior_projects.loc[0,'b{}'.format(yr)]*100)
    y2.append(harvest_interior_surround.loc[0,'b{}'.format(yr)]*100)
    y3.append(harvest_interior.loc[0,'b{}'.format(yr)]*100)

ax6.plot(range(1986,2022), y1, color='red', linewidth=2, label='Projects')
ax6.plot(range(1986,2022), y2, color='black', linewidth=2, label='Surroundings')
ax6.plot(range(1986,2022), y3, color='gray', linewidth=2, label='Interior region')
ax6.set_xlim((1985,2022))
ax6.set_ylim((0,4.6))
ax6.text(-0.12,1,'(f)',fontsize=12, fontweight='bold', transform=ax6.transAxes)
ax6.legend(fontsize=8)
ax6.set_xlabel('Year')
ax6.grid(zorder=0, linewidth=0.4, color='0.9')
import matplotlib.ticker as mtick
ax6.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

y1 = np.array(y1[:-9]) #1986-2012
y2 = np.array(y2[:-9])
y3 = np.array(y3[:-9])

print('harvest, projects vs. surroundings', ((y1 - y2)/ y2).mean())
print(stats.ttest_rel(y1, y2))
print('harvest, projects vs. interior', ((y1 - y3)/ y3).mean())
print(stats.ttest_rel(y1, y3))

plt.savefig(root + 'figures/fig4_carbon-harvest-timeseries.eps')
#plt.savefig(root + 'figures/fig4_carbon-harvest-timeseries.jpg', dpi=300)
