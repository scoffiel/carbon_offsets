#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: supplemental analysis matching projects to "similar forests"
- Match each project to most similar 4km pixels in California based on mean annual Temperature, Precip, productivity (site class)

Inputs:
- projects spreadsheet
- CSV files for T, P, site class, harvest, emapr, lemma by project (from GEE)
- rasters of 800m T, P, site class, harvest, emapr, lemma (excluding mostly-project pixels) (from GEE)

Outputs:
- Figures - S4-6 bar charts comparing projects, surroundings, matched pixels
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.spatial.distance import cdist
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scipy import stats
import matplotlib.gridspec as gs

root = '/Users/scoffiel/california/offsets/'

#read in project data
projects_T = pd.read_csv(root + 'processed_data/prism_climate/projects_T.csv', index_col='project_id')['mean']
projects_P = pd.read_csv(root + 'processed_data/prism_climate/projects_P.csv', index_col='project_id')['mean']
projects_siteclass = pd.read_csv(root + 'processed_data/site_class/projects_site_class.csv', index_col='project_id')['mean']

projects_harvest = pd.read_csv(root + 'processed_data/harvest/projects_harvest.csv', index_col='project_id')
projects_emapr = pd.read_csv(root + 'processed_data/carbon_emapr/projects_emapr.csv', index_col='project_id')
projects_lemma = pd.read_csv(root + 'processed_data/carbon_lemma/projects_lemma.csv', index_col='project_id') #need to *0.47 somewhere

#for comparison of similarity
surround_T = pd.read_csv(root + 'processed_data/prism_climate/surround_T.csv', index_col='project_id')['mean']
surround_P = pd.read_csv(root + 'processed_data/prism_climate/surround_P.csv', index_col='project_id')['mean']
surround_siteclass = pd.read_csv(root + 'processed_data/site_class/surround_site_class.csv', index_col='project_id')['mean']


#read in gridded data to match to
gridded_T = xr.open_rasterio(root + 'processed_data/prism_climate/prism_T_nonproject.tiff')[0,:,:]
gridded_P = xr.open_rasterio(root + 'processed_data/prism_climate/prism_P_nonproject.tiff')[0,:,:]
gridded_siteclass = xr.open_rasterio(root + 'processed_data/site_class/site_class_800m_nonproject.tiff')[0,:,:]

gridded_harvest = xr.open_rasterio(root + 'processed_data/harvest/harvest_800m.tiff')
gridded_emapr = xr.open_rasterio(root + 'processed_data/carbon_emapr/emapr_800m.tiff')*0.47
gridded_lemma = xr.open_rasterio(root + 'processed_data/carbon_lemma/lemma_800m.tiff')*0.47/1000

gridded_emapr = gridded_emapr.where(gridded_emapr >= 0)
gridded_lemma = gridded_lemma.where(gridded_lemma >= 0)



#Build data table (projects)
projects = pd.concat([projects_P, projects_T, projects_siteclass], axis=1)
projects.columns = ['p','t','site_class']
projects_area = pd.read_csv(root + 'all_projects.csv', index_col='project_id').area_ha


surround = pd.concat([surround_P, surround_T, surround_siteclass], axis=1)
surround.columns = ['p','t','site_class']


#Build big data table (800m) -------------
#two separate tables - one for coast one for interior to restrict searching
gridded_P = {'coast': xr.open_rasterio(root + 'processed_data/prism_climate/prism_P_nonproject_coast.tiff')[0,:,:], 
             'interior': xr.open_rasterio(root + 'processed_data/prism_climate/prism_P_nonproject_interior.tiff')[0,:,:]}
tables = {}
for region in gridded_P:
    p = gridded_P[region]

    table = p.where(p > 0).to_dataframe('p').dropna().reset_index()
    del table['band']
    
    x = table.x.to_xarray()
    y = table.y.to_xarray()
    
    table['t'] = gridded_T.where(gridded_T > 0).sel(x=x, y=y, method='nearest').data
    
    table['site_class'] = gridded_siteclass.sel(x=x, y=y, method='nearest').data
    for i in range(1,37):
        table['h{}'.format(1985+i)] = gridded_harvest.sel(x=x, y=y, band=i, method='nearest').data
        if i < 33:
            table['e{}'.format(1985+i)] = gridded_emapr.sel(x=x, y=y, band=i, method='nearest').data
            table['l{}'.format(1985+i)] = gridded_lemma.sel(x=x, y=y, band=i, method='nearest').data
    
    tables[region] = table.dropna().reset_index(drop=True)


#Calculate the Mahalanobis distance ------------------------------------------
#For each project, calculate dist to all pixels and find minimum
#sqrt( (b-a)T corr^-1 (b-a))
#calculate a, b, corr separately for each region


projects = {'coast': projects[~projects.index.isin(['CAR1066','CAR1041','CAR1092','CAR1114'])], 
            'interior': projects[projects.index.isin(['CAR1066','CAR1041','CAR1092','CAR1114'])]}
surround = {'coast': surround[~surround.index.isin(['CAR1066','CAR1041','CAR1092','CAR1114'])], 
            'interior': surround[surround.index.isin(['CAR1066','CAR1041','CAR1092','CAR1114'])]}
projects_controls = {'coast': pd.DataFrame(),
                     'interior': pd.DataFrame()}

for region in ['coast','interior']:

    a = tables[region][['p','t','site_class']]    #(n x 3)
    
    projects[region] = (projects[region]-a.mean())/a.std() #standardize units
    surround[region] = (surround[region]-a.mean())/a.std()
    a = (a-a.mean())/a.std() 
    
    corr = np.corrcoef(a.T)
    corrinv = np.linalg.inv(corr)
    
    for p in projects[region].index:
        b = projects[region].loc[p,['p','t','site_class']] #(1 x 3)
        
        d = cdist(a, np.array(b).reshape(1,3), 'mahalanobis', VI=corrinv) #mahalanobis distance in T, P, siteclass space
        #imin = np.nanargmin(d) #locations of minimum values. old way selecting a single minimum
        
        area = projects_area.loc[p]
        n_pixels = int(area * 10000 / 927/927)
        #average project is 4567 ha = 4.6e7 m2 = 53 pixels (927m x 927m)
        order = np.argsort(d, axis=None)[:n_pixels] #indices of 50 most similar pixel
        
        projects[region].loc[p, 'analog_dist'] = d[order].mean()
        #projects.loc[p, 'analog_x'] = table.loc[imin, 'x']
        #projects.loc[p, 'analog_y'] = table.loc[imin, 'y']
        
        #also calculate distance between projects and their surroundings for comparison
        s = surround[region].loc[p,['p','t','site_class']]
        ds = cdist(np.array(s).reshape(1,3), np.array(b).reshape(1,3), 'mahalanobis', VI=corrinv)
        projects[region].loc[p, 'surround_dist'] = float(ds)
        
        projects_controls[region].loc[p, tables[region].columns[-100:]] = tables[region].iloc[order, -100:].mean() #harvest, emapr, lemma data
        projects_controls[region].loc[p, 'area'] = projects_area.loc[p]
    

#set up for plotting
coast_controls = projects_controls['coast']
interior_controls = projects_controls['interior']
projects_controls = pd.concat([projects_controls['coast'], projects_controls['interior']])



#repeat timeseries & barcharts for projects, surroundings, matched controls --------------------------------------------------------

fig = plt.figure(figsize=(7,10))

grid = gs.GridSpec(3,2, height_ratios=[1,1,1])

ax = plt.subplot(grid[0])
ax2 = plt.subplot(grid[1])
ax3 = plt.subplot(grid[2])
ax4 = plt.subplot(grid[3])
ax5 = plt.subplot(grid[4])
ax6 = plt.subplot(grid[5])


#emapr
carbon_data = 'emapr'
#ax1 coast: compare year-by-year timeseries for EMAPR for combined polygons surround vs projects
carbon_coast_projects = pd.read_csv(root + 'processed_data/carbon_{}/coast_projects_{}.csv'.format(carbon_data, carbon_data))
carbon_coast_surround = pd.read_csv(root + 'processed_data/carbon_{}/coast_surroundings_{}.csv'.format(carbon_data, carbon_data))
carbon_coast = pd.read_csv(root + 'processed_data/carbon_{}/coast_{}.csv'.format(carbon_data, carbon_data))
projects = pd.read_csv(root + 'all_projects.csv')
projects = projects[~projects.project_id.isin(['CAR1066','CAR1041','CAR1092','CAR1114'])]

y1 =[] #projects
y2 =[] #surround
y3 =[] #region
y4 =[] #NEW mahalanobis matched controls

for yr in range(3,35):
    y1.append(carbon_coast_projects.loc[0,'b{}'.format(yr)]*0.47)
    y2.append(carbon_coast_surround.loc[0,'b{}'.format(yr)]*0.47)
    y3.append(carbon_coast.loc[0,'b{}'.format(yr)]*0.47)
    y4.append(np.average(coast_controls['e{}'.format(yr+1983)], weights=coast_controls.area))

        
print('mean and stdev of carbon, projects:', np.mean(y1), np.std(y1))
print('mean and stdev of carbon, surrounds:', np.mean(y2), np.std(y2))
print('mean and stdev of carbon, region:', np.mean(y3), np.std(y3))

print('slope and stderr of carbon, projects:', stats.linregress(range(len(y1)), y1/y1[0]).slope, stats.linregress(range(len(y1)), y1/y1[0]).stderr)
print('slope and stderr of carbon, surrounds:', stats.linregress(range(len(y2)), y2/y2[0]).slope, stats.linregress(range(len(y2)), y2/y2[0]).stderr)
print('slope and stderr of carbon, region:', stats.linregress(range(len(y3)), y3/y3[0]).slope, stats.linregress(range(len(y3/y3[0])), y3).stderr)

axb = ax.twinx()
color = 'tab:blue'
axb.hist(projects.start_year, bins=range(2012,2020))
axb.set_yticks([0,2,4,6,8])
axb.set_ylim((0,22))
axb.tick_params(axis='y', labelcolor=color)

ax.plot(range(1986,2018), y1, color='red', linewidth=2, label='Projects')
ax.plot(range(1986,2018), y2, color='black', linewidth=2, label='Surroundings')
ax.plot(range(1986,2018), y3, color='gray', linewidth=2, label='Coastal region')
ax.plot(range(1986,2018), y4, color='purple', linewidth=2, label='Matched controls')

ax.set_ylabel('eMapR carbon (ton C/ha)') #lemma or emapr
ax.set_xlim((1985,2022))
ax.set_ylim((0,149))
ax.set_title('Coastal projects')
ax.text(-0.12,1,'(a)',fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.grid(zorder=0, linewidth=0.4, color='0.9')
ax.legend(fontsize=8)



#ax2 interior: compare year-by-year timeseries for EMAPR for combined polygons surround vs projects
carbon_interior_projects = pd.read_csv(root + 'processed_data/carbon_{}/interior_projects_{}.csv'.format(carbon_data, carbon_data))
carbon_interior_surround = pd.read_csv(root + 'processed_data/carbon_{}/interior_surroundings_{}.csv'.format(carbon_data, carbon_data))
carbon_interior = pd.read_csv(root + 'processed_data/carbon_{}/interior_{}.csv'.format(carbon_data, carbon_data))
projects = pd.read_csv(root + 'all_projects.csv')
projects = projects[projects.project_id.isin(['CAR1066','CAR1041','CAR1092','CAR1114'])]

y1 =[] #projects
y2 =[] #surround
y3 =[] #region
y4 =[] #controls

for yr in range(3,35):
    y1.append(carbon_interior_projects.loc[0,'b{}'.format(yr)]*0.47)
    y2.append(carbon_interior_surround.loc[0,'b{}'.format(yr)]*0.47)
    y3.append(carbon_interior.loc[0,'b{}'.format(yr)]*0.47)
    y4.append(np.average(interior_controls['e{}'.format(yr+1983)], weights=interior_controls.area))


print('mean and stdev of carbon, projects:', np.mean(y1), np.std(y1))
print('mean and stdev of carbon, surrounds:', np.mean(y2), np.std(y2))
print('mean and stdev of carbon, region:', np.mean(y3), np.std(y3))

print('slope and stderr of carbon, projects:', stats.linregress(range(len(y1)), y1/y1[0]).slope, stats.linregress(range(len(y1)), y1/y1[0]).stderr)
print('slope and stderr of carbon, surrounds:', stats.linregress(range(len(y2)), y2/y2[0]).slope, stats.linregress(range(len(y2)), y2/y2[0]).stderr)
print('slope and stderr of carbon, region:', stats.linregress(range(len(y3)), y3/y3[0]).slope, stats.linregress(range(len(y3/y3[0])), y3).stderr)


ax2b = ax2.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2b.set_ylabel('New projects', color=color, position=(0,0), ha='left') 
ax2b.hist(projects.start_year, bins=range(2012,2020))
ax2b.set_yticks([0,2,4,6,8])
ax2b.set_ylim((0,22))
ax2b.tick_params(axis='y', labelcolor=color)

ax2.plot(range(1986,2018), y1, color='red', linewidth=2, label='Projects')
ax2.plot(range(1986,2018), y2, color='black', linewidth=2, label='Surroundings')
ax2.plot(range(1986,2018), y3, color='gray', linewidth=2, label='Interior region')
ax2.plot(range(1986,2018), y4, color='purple', linewidth=2, label='Matched controls')

ax2.set_xlim((1985,2022))
ax2.set_ylim((0,149))
ax2.set_title('Interior projects')
ax2.text(-0.12,1,'(b)',fontsize=12, fontweight='bold', transform=ax2.transAxes)
ax2.grid(zorder=0, linewidth=0.4, color='0.9')
ax2.legend(fontsize=8)
ax2.set_xticks(range(1990,2021,10)) #not showing up?



#ax3 coast: compare year-by-year timeseries for LEMMA for combined polygons surround vs projects
carbon_data = 'lemma'
carbon_coast_projects = pd.read_csv(root + 'processed_data/carbon_{}/coast_projects_{}.csv'.format(carbon_data, carbon_data))
carbon_coast_surround = pd.read_csv(root + 'processed_data/carbon_{}/coast_surroundings_{}.csv'.format(carbon_data, carbon_data))
carbon_coast = pd.read_csv(root + 'processed_data/carbon_{}/coast_{}.csv'.format(carbon_data, carbon_data))
projects = pd.read_csv(root + 'all_projects.csv')
projects = projects[~projects.project_id.isin(['CAR1066','CAR1041','CAR1092','CAR1114'])]

y1 =[] #projects
y2 =[] #surround
y3 =[] #region
y4 =[] #NEW mahalanobis matched controls

for yr in range(1986,2018):
    y1.append(carbon_coast_projects.loc[0,'{}_b1'.format(yr)]*0.47/1000)
    y2.append(carbon_coast_surround.loc[0,'{}_b1'.format(yr)]*0.47/1000)
    y3.append(carbon_coast.loc[0,'{}_b1'.format(yr)]*0.47/1000)
    y4.append(np.average(coast_controls['l{}'.format(yr)], weights=coast_controls.area))
        
print('mean and stdev of carbon, projects:', np.mean(y1), np.std(y1))
print('mean and stdev of carbon, surrounds:', np.mean(y2), np.std(y2))
print('mean and stdev of carbon, region:', np.mean(y3), np.std(y3))

print('slope and stderr of carbon, projects:', stats.linregress(range(len(y1)), y1/y1[0]).slope, stats.linregress(range(len(y1)), y1/y1[0]).stderr)
print('slope and stderr of carbon, surrounds:', stats.linregress(range(len(y2)), y2/y2[0]).slope, stats.linregress(range(len(y2)), y2/y2[0]).stderr)
print('slope and stderr of carbon, region:', stats.linregress(range(len(y3)), y3/y3[0]).slope, stats.linregress(range(len(y3/y3[0])), y3).stderr)

ax3.plot(range(1986,2018), y1, color='red', linewidth=2, label='Projects')
ax3.plot(range(1986,2018), y2, color='black', linewidth=2, label='Surroundings')
ax3.plot(range(1986,2018), y3, color='gray', linewidth=2, label='Coastal region')
ax3.plot(range(1986,2018), y4, color='purple', linewidth=2, label='Matched controls')

ax3.set_ylabel('LEMMA carbon (ton C/ha)') #lemma or emapr
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
y4 =[] #controls

for yr in range(1986,2018):
    y1.append(carbon_interior_projects.loc[0,'{}_b1'.format(yr)]*0.47/1000)
    y2.append(carbon_interior_surround.loc[0,'{}_b1'.format(yr)]*0.47/1000)
    y3.append(carbon_interior.loc[0,'{}_b1'.format(yr)]*0.47/1000)
    y4.append(np.average(interior_controls['l{}'.format(yr)], weights=interior_controls.area))

print('mean and stdev of carbon, projects:', np.mean(y1), np.std(y1))
print('mean and stdev of carbon, surrounds:', np.mean(y2), np.std(y2))
print('mean and stdev of carbon, region:', np.mean(y3), np.std(y3))

print('slope and stderr of carbon, projects:', stats.linregress(range(len(y1)), y1/y1[0]).slope, stats.linregress(range(len(y1)), y1/y1[0]).stderr)
print('slope and stderr of carbon, surrounds:', stats.linregress(range(len(y2)), y2/y2[0]).slope, stats.linregress(range(len(y2)), y2/y2[0]).stderr)
print('slope and stderr of carbon, region:', stats.linregress(range(len(y3)), y3/y3[0]).slope, stats.linregress(range(len(y3/y3[0])), y3).stderr)

ax4.plot(range(1986,2018), y1, color='red', linewidth=2, label='Projects')
ax4.plot(range(1986,2018), y2, color='black', linewidth=2, label='Surroundings')
ax4.plot(range(1986,2018), y3, color='gray', linewidth=2, label='Interior region')
ax4.plot(range(1986,2018), y4, color='purple', linewidth=2, label='Matched controls')

#ax4.set_ylabel('Carbon (ton C/ha) (eMapR)')
ax4.set_xlim((1985,2022))
ax4.set_ylim((0,149))
ax4.text(-0.12,1,'(d)',fontsize=12, fontweight='bold', transform=ax4.transAxes)
ax4.grid(zorder=0, linewidth=0.4, color='0.9')
ax4.legend(fontsize=8)


#ax5 coast: compare year-by-year timeseries for HARVEST for combined polygons surround vs projects
harvest_coast_projects = pd.read_csv(root + 'processed_data/harvest/coast_projects_harvest.csv')
harvest_coast_surround = pd.read_csv(root + 'processed_data/harvest/coast_surroundings_harvest.csv')
harvest_coast = pd.read_csv(root + 'processed_data/harvest/coast_harvest.csv')

y1 =[] #projects
y2 =[] #surrounds
y3 =[] #region
y4 =[] #controls
for yr in range(2,38): #file has bands b2-b37, 36 years [1986-2021]
    y1.append(harvest_coast_projects.loc[0,'b{}'.format(yr)]*100)
    y2.append(harvest_coast_surround.loc[0,'b{}'.format(yr)]*100)
    y3.append(harvest_coast.loc[0,'b{}'.format(yr)]*100)
    y4.append(np.average(coast_controls['h{}'.format(yr+1984)], weights=coast_controls.area)*100)

ax5.plot(range(1986,2022), y1, color='red', linewidth=2, label='Projects')
ax5.plot(range(1986,2022), y2, color='black', linewidth=2, label='Surroundings')
ax5.plot(range(1986,2022), y3, color='gray', linewidth=2, label='Coastal region')
ax5.plot(range(1986,2022), y4, color='purple', linewidth=2, label='Matched controls')

ax5.set_ylabel('Fraction of area harvested')
ax5.set_xlim((1985,2022))
ax5.set_ylim((0,4.6))
ax5.text(-0.12,1,'(e)',fontsize=12, fontweight='bold', transform=ax5.transAxes)
ax5.legend(fontsize=8)
ax5.set_xlabel('Year')
ax5.grid(zorder=0, linewidth=0.4, color='0.9')
import matplotlib.ticker as mtick
ax5.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))


#ax6 interior: compare year-by-year timeseries for HARVEST for combined polygons surround vs projects
harvest_interior_projects = pd.read_csv(root + 'processed_data/harvest/interior_projects_harvest.csv')
harvest_interior_surround = pd.read_csv(root + 'processed_data/harvest/interior_surroundings_harvest.csv')
harvest_interior = pd.read_csv(root + 'processed_data/harvest/interior_harvest.csv')

y1 =[] #projects
y2 =[] #surrounds
y3 =[] #region
y4 =[] #controls
for yr in range(2,38):
    y1.append(harvest_interior_projects.loc[0,'b{}'.format(yr)]*100)
    y2.append(harvest_interior_surround.loc[0,'b{}'.format(yr)]*100)
    y3.append(harvest_interior.loc[0,'b{}'.format(yr)]*100)
    y4.append(np.average(interior_controls['h{}'.format(yr+1984)], weights=interior_controls.area)*100)

ax6.plot(range(1986,2022), y1, color='red', linewidth=2, label='Projects')
ax6.plot(range(1986,2022), y2, color='black', linewidth=2, label='Surroundings')
ax6.plot(range(1986,2022), y3, color='gray', linewidth=2, label='Interior region')
ax6.plot(range(1986,2022), y4, color='purple', linewidth=2, label='Matched controls')

ax6.set_xlim((1985,2022))
ax6.set_ylim((0,4.6))
ax6.text(-0.12,1,'(f)',fontsize=12, fontweight='bold', transform=ax6.transAxes)
ax6.set_xlabel('Year')
ax6.grid(zorder=0, linewidth=0.4, color='0.9')
import matplotlib.ticker as mtick
ax6.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)

(y1[:-9] - y2[:-9]).mean() 

print(stats.ttest_rel(y1[:-9], y2[:-9]))
print(stats.ttest_rel(y1[:-9], y3[:-9]))



# barcharts figures ----------------------------------------------------------------------

from matplotlib.patches import Patch
import chowtest

#read in projects table with projects indices
projects = pd.read_csv(root + 'all_projects.csv', index_col=0)
projects = projects.sort_values('group')

projects = projects[projects.start_year < 2015]
projects = projects.drop(index='CAR1046') #drop the one that burned
n = len(projects)

harvest_projects = pd.read_csv(root + 'processed_data/harvest/projects_harvest.csv', index_col='project_id').loc[projects.index,:]
harvest_surround = pd.read_csv(root + 'processed_data/harvest/surroundings_harvest.csv', index_col='project_id').loc[projects.index,:]
emapr_projects = pd.read_csv(root + 'processed_data/carbon_emapr/projects_emapr.csv', index_col='project_id').loc[projects.index,:]
emapr_surround = pd.read_csv(root + 'processed_data/carbon_emapr/surroundings_emapr.csv', index_col='project_id').loc[projects.index,:]
lemma_projects = pd.read_csv(root + 'processed_data/carbon_lemma/projects_lemma.csv', index_col='project_id').loc[projects.index,:]
lemma_surround = pd.read_csv(root + 'processed_data/carbon_lemma/surroundings_lemma.csv', index_col='project_id').loc[projects.index,:]


for p in projects.index:
    #harvest goes from 1986-2021, b2-b37, 36 years
    #emapr goes from 1986-2017, b3-b34, 32 years
    #lemma goes from 1986-2017, 1986_b1 to 2017_b1, 32 years

    start_year = projects.loc[p, 'start_year']
    length = 2021-start_year
    
    pre_range_harvest = range(start_year - length +1, start_year+1)
    post_range_harvest = range(start_year + 1, 2022)
    
    pre_range_emapr = pre_range_lemma = pre_range_harvest[4:] #4 years shorter on each end
    post_range_emapr = post_range_lemma = post_range_harvest[:-4]
    
    #new part for controls
    pre_range_controls_harvest = ['h' + str(y) for y in pre_range_harvest]
    post_range_controls_harvest = ['h' + str(y) for y in post_range_harvest]
    pre_range_controls_emapr = ['e' + str(y) for y in pre_range_emapr]
    post_range_controls_emapr = ['e' + str(y) for y in post_range_emapr]
    pre_range_controls_lemma = ['l' + str(y) for y in pre_range_lemma]
    post_range_controls_lemma = ['l' + str(y) for y in post_range_lemma]
    #
    
    pre_range_harvest = ['b' + str(y-1984) for y in pre_range_harvest]
    post_range_harvest = ['b' + str(y-1984) for y in post_range_harvest]
    
    pre_range_emapr = ['b' + str(y-1983) for y in pre_range_emapr]
    post_range_emapr = ['b' + str(y-1983) for y in post_range_emapr]
    initial_emapr = 'b' + str(start_year - 1983)
    
    pre_range_lemma = [str(y) + '_b1' for y in pre_range_lemma]
    post_range_lemma = [str(y) + '_b1' for y in post_range_lemma]
    initial_lemma = str(start_year) + '_b1'

    project_pre_harvest = harvest_projects.loc[p, pre_range_harvest]
    project_post_harvest = harvest_projects.loc[p, post_range_harvest]
    surround_pre_harvest = harvest_surround.loc[p, pre_range_harvest]
    surround_post_harvest = harvest_surround.loc[p, post_range_harvest]
    
    projects.loc[p, 'project_pre_harvest'] = project_pre_harvest.mean()
    projects.loc[p, 'project_post_harvest'] = project_post_harvest.mean()
    projects.loc[p, 'surround_pre_harvest'] = surround_pre_harvest.mean()
    projects.loc[p, 'surround_post_harvest'] = surround_post_harvest.mean()
    
    project_pre_emapr = list(emapr_projects.loc[p, pre_range_emapr]*0.47)
    project_post_emapr = list(emapr_projects.loc[p, post_range_emapr].dropna()*0.47)
    surround_pre_emapr = list(emapr_surround.loc[p, pre_range_emapr]*0.47)
    surround_post_emapr = list(emapr_surround.loc[p, post_range_emapr].dropna()*0.47)
    
    projects.loc[p, 'project_pre_emapr'] = stats.linregress(range(len(project_pre_emapr)),project_pre_emapr ).slope
    projects.loc[p, 'project_post_emapr'] = stats.linregress(range(len(project_post_emapr)),project_post_emapr ).slope
    projects.loc[p, 'surround_pre_emapr'] = stats.linregress(range(len(surround_pre_emapr)),surround_pre_emapr ).slope
    projects.loc[p, 'surround_post_emapr'] = stats.linregress(range(len(surround_post_emapr)),surround_post_emapr ).slope
    projects.loc[p, 'project_initial_emapr'] = emapr_projects.loc[p, initial_emapr] * 0.47 #new
    projects.loc[p, 'surround_initial_emapr'] = emapr_surround.loc[p, initial_emapr] * 0.47 #new
    
    project_pre_lemma = list(lemma_projects.loc[p, pre_range_lemma]*0.47/1000)
    project_post_lemma = list(lemma_projects.loc[p, post_range_lemma].dropna()*0.47/1000)
    surround_pre_lemma = list(lemma_surround.loc[p, pre_range_lemma]*0.47/1000)
    surround_post_lemma = list(lemma_surround.loc[p, post_range_lemma].dropna()*0.47/1000)
    
    projects.loc[p, 'project_pre_lemma'] = stats.linregress(range(len(project_pre_lemma)),project_pre_lemma ).slope
    projects.loc[p, 'project_post_lemma'] = stats.linregress(range(len(project_post_lemma)),project_post_lemma ).slope
    projects.loc[p, 'surround_pre_lemma'] = stats.linregress(range(len(surround_pre_lemma)),surround_pre_lemma ).slope
    projects.loc[p, 'surround_post_lemma'] = stats.linregress(range(len(surround_post_lemma)),surround_post_lemma ).slope
    projects.loc[p, 'project_initial_lemma'] = lemma_projects.loc[p, initial_lemma] * 0.47/1000 #new
    projects.loc[p, 'surround_initial_lemma'] = lemma_surround.loc[p, initial_lemma] * 0.47/1000 #new
    

    controls_pre_harvest = projects_controls.loc[p, pre_range_controls_harvest]
    controls_post_harvest = projects_controls.loc[p, post_range_controls_harvest]
    projects.loc[p, 'control_pre_harvest'] = controls_pre_harvest.mean()
    projects.loc[p, 'control_post_harvest'] = controls_post_harvest.mean()
    
    controls_pre_emapr = list(projects_controls.loc[p, pre_range_controls_emapr])
    controls_post_emapr = list(projects_controls.loc[p, post_range_controls_emapr].dropna())
    projects.loc[p, 'control_pre_emapr'] = stats.linregress(range(len(controls_pre_emapr)),controls_pre_emapr ).slope
    projects.loc[p, 'control_post_emapr'] = stats.linregress(range(len(controls_post_emapr)),controls_post_emapr ).slope
    
    controls_pre_lemma = list(projects_controls.loc[p, pre_range_controls_lemma])
    controls_post_lemma = list(projects_controls.loc[p, post_range_controls_lemma].dropna())
    projects.loc[p, 'control_pre_lemma'] = stats.linregress(range(len(controls_pre_lemma)),controls_pre_lemma).slope
    projects.loc[p, 'control_post_lemma'] = stats.linregress(range(len(controls_post_lemma)),controls_post_lemma ).slope
    
    
    
    #chow tests across projects --------    
    #project emapr
    x1 = np.array(range(len(project_pre_emapr)), dtype='float')
    x2 = x1+len(x1)
    y1 = np.array(project_pre_emapr, dtype='float')
    y2 = np.array(project_post_emapr, dtype='float')
    p_val = chowtest.p_value(y1, x1, y2, x2)
    m1,b1,_,_,_ = stats.linregress(x1,y1)
    m2,b2,_,_,_ = stats.linregress(x2,y2)
    projects.loc[p, 'p_value_project_emapr'] = p_val
    projects.loc[p, 'increased'] = m2>m1
    
    #surround emapr
    y1 = np.array(surround_pre_emapr, dtype='float')
    y2 = np.array(surround_post_emapr, dtype='float')
    p_val = chowtest.p_value(y1, x1, y2, x2)
    m1,b1,_,_,_ = stats.linregress(x1,y1)
    m2,b2,_,_,_ = stats.linregress(x2,y2)
    projects.loc[p, 'p_value_surround_emapr'] = p_val
    
    #control emapr
    y1 = np.array(controls_pre_emapr, dtype='float')
    y2 = np.array(controls_post_emapr, dtype='float')
    p_val = chowtest.p_value(y1, x1, y2, x2)
    m1,b1,_,_,_ = stats.linregress(x1,y1)
    m2,b2,_,_,_ = stats.linregress(x2,y2)
    projects.loc[p, 'p_value_control_emapr'] = p_val
    
    
    #project lemma
    x1 = np.array(range(len(project_pre_lemma)), dtype='float')
    x2 = x1+len(x1)
    y1 = np.array(project_pre_lemma, dtype='float')
    y2 = np.array(project_post_lemma, dtype='float')
    p_val = chowtest.p_value(y1, x1, y2, x2)
    m1,b1,_,_,_ = stats.linregress(x1,y1)
    m2,b2,_,_,_ = stats.linregress(x2,y2)
    projects.loc[p, 'p_value_project_lemma'] = p_val
    projects.loc[p, 'increased'] = m2>m1
    
    #surround lemma
    y1 = np.array(surround_pre_lemma, dtype='float')
    y2 = np.array(surround_post_lemma, dtype='float')
    p_val = chowtest.p_value(y1, x1, y2, x2)
    m1,b1,_,_,_ = stats.linregress(x1,y1)
    m2,b2,_,_,_ = stats.linregress(x2,y2)
    projects.loc[p, 'p_value_surround_lemma'] = p_val
    
    #control lemma
    y1 = np.array(controls_pre_lemma, dtype='float')
    y2 = np.array(controls_post_lemma, dtype='float')
    p_val = chowtest.p_value(y1, x1, y2, x2)
    m1,b1,_,_,_ = stats.linregress(x1,y1)
    m2,b2,_,_,_ = stats.linregress(x2,y2)
    projects.loc[p, 'p_value_control_lemma'] = p_val

    #project harvest
    y1 = np.array(project_pre_harvest, dtype='float')
    y2 = np.array(project_post_harvest, dtype='float')
    p_val = stats.ttest_ind(y1, y2).pvalue
    projects.loc[p, 'p_value_project_harvest'] = p_val
    
    #surround harvest 
    y1 = np.array(surround_pre_harvest, dtype='float')
    y2 = np.array(surround_post_harvest, dtype='float')
    p_val = stats.ttest_ind(y1, y2).pvalue
    projects.loc[p, 'p_value_surround_harvest'] = p_val
    
    #control harvest 
    y1 = np.array(controls_pre_harvest, dtype='float')
    y2 = np.array(controls_post_harvest, dtype='float')
    p_val = stats.ttest_ind(y1, y2).pvalue
    projects.loc[p, 'p_value_control_harvest'] = p_val


#first figure for all projects - emapr carbon, lemma carbon, and harvest proj vs surrounds (fig s3)
fig, (ax1,ax2, ax3) = plt.subplots(3,1, figsize=(10,9), tight_layout=True)

#ax1 emapr
i = 0
for p in projects.index:
    ax1.bar(i-0.375, projects.loc[p,'project_pre_emapr'], color='salmon', width=0.15)
    ax1.bar(i-0.225, projects.loc[p,'project_post_emapr'], color='red', width=0.15)
    if projects.loc[p,'p_value_project_emapr'] < 0.05:
        ax1.text(i-0.38, max(projects.loc[p,'project_pre_emapr'], projects.loc[p,'project_post_emapr']),'*', fontsize=12, color='red')
    
    ax1.bar(i-0.075, projects.loc[p,'surround_pre_emapr'], color='0.55', width=0.15)
    ax1.bar(i+0.075, projects.loc[p,'surround_post_emapr'], color='0.4', width=0.15)
    if projects.loc[p,'p_value_surround_emapr'] < 0.05:
        ax1.text(i-0.1, max(projects.loc[p,'surround_pre_emapr'], projects.loc[p,'surround_post_emapr']),'*', fontsize=12, color='0.4')
    
    ax1.bar(i+0.225, projects.loc[p,'control_pre_emapr'], color='plum', width=0.15)
    ax1.bar(i+0.375, projects.loc[p,'control_post_emapr'], color='purple', width=0.15)
    if projects.loc[p,'p_value_control_emapr'] < 0.05:
        ax1.text(i+0.2, max(projects.loc[p,'control_pre_emapr'], projects.loc[p,'control_post_emapr']),'*', fontsize=12, color='purple')
    
    i = i+1
    
ax1.set_xticks(range(len(projects)))
ax1.set_xticklabels(['']*len(projects))
ax1.set_ylabel('eMapR carbon accumulation rate\n(tonC/ha/y)')
ax1.set_ylim((-3,3.5))
ax1.text(0.01,0.9,'(a)',fontsize=12, fontweight='bold', transform=ax1.transAxes)

legend_elements = [Patch(facecolor='salmon', edgecolor='none',label='Projects, before'),
                   Patch(facecolor='red', edgecolor='none',label='Projects, after'),
                   Patch(facecolor='0.55', edgecolor='none',label='Surroundings, before'),
                   Patch(facecolor='0.4', edgecolor='none',label='Surroundings, after'),
                   Patch(facecolor='plum', edgecolor='none',label='Matched controls, before'),
                   Patch(facecolor='purple', edgecolor='none',label='Matched controls, after')]
ax1.legend(handles=legend_elements, ncol=3)


#ax2 lemma
i = 0
for p in projects.index:
    ax2.bar(i-0.375, projects.loc[p,'project_pre_lemma'], color='salmon', width=0.15)
    ax2.bar(i-0.225, projects.loc[p,'project_post_lemma'], color='red', width=0.15)
    if projects.loc[p,'p_value_project_lemma'] < 0.05:
        ax2.text(i-0.38, max(projects.loc[p,'project_pre_lemma'], projects.loc[p,'project_post_lemma']),'*', fontsize=12, color='red')
    
    ax2.bar(i-0.075, projects.loc[p,'surround_pre_lemma'], color='0.55', width=0.15)
    ax2.bar(i+0.075, projects.loc[p,'surround_post_lemma'], color='0.4', width=0.15)
    if projects.loc[p,'p_value_surround_lemma'] < 0.05:
        ax2.text(i-0.1, max(projects.loc[p,'surround_pre_lemma'], projects.loc[p,'surround_post_lemma']),'*', fontsize=12, color='0.4')
    
    ax2.bar(i+0.225, projects.loc[p,'control_pre_lemma'], color='plum', width=0.15)
    ax2.bar(i+0.375, projects.loc[p,'control_post_lemma'], color='purple', width=0.15)
    if projects.loc[p,'p_value_control_lemma'] < 0.05:
        ax2.text(i+0.2, max(projects.loc[p,'control_pre_lemma'], projects.loc[p,'control_post_lemma']),'*', fontsize=12, color='purple')
    
    i = i+1
    
ax2.set_xticks(range(len(projects)))
ax2.set_xticklabels(['']*len(projects))
ax2.set_ylabel('LEMMA carbon accumulation rate\n(tonC/ha/y)')
ax2.set_ylim((-3,3.5))
ax2.text(0.01,0.9,'(b)',fontsize=12, fontweight='bold', transform=ax2.transAxes)



#ax3 harvest
i = 0
for p in projects.index:
    ax3.bar(i-0.375, projects.loc[p,'project_pre_harvest']*100, color='salmon', width=0.15)
    ax3.bar(i-0.225, projects.loc[p,'project_post_harvest']*100, color='red', width=0.15)
    if projects.loc[p,'p_value_project_harvest'] < 0.05:
        ax3.text(i-0.38, max(projects.loc[p,'project_pre_harvest'], projects.loc[p,'project_post_harvest'])*100,'*', fontsize=12, color='red')
    
    ax3.bar(i-0.075, projects.loc[p,'surround_pre_harvest']*100, color='0.55', width=0.15)
    ax3.bar(i+0.075, projects.loc[p,'surround_post_harvest']*100, color='0.4', width=0.15)
    if projects.loc[p,'p_value_surround_harvest'] < 0.05:
        ax3.text(i-0.1, max(projects.loc[p,'surround_pre_harvest'], projects.loc[p,'surround_post_harvest'])*100,'*', fontsize=12, color='0.4')
    
    ax3.bar(i+0.225, projects.loc[p,'control_pre_harvest']*100, color='plum', width=0.15)
    ax3.bar(i+0.375, projects.loc[p,'control_post_harvest']*100, color='purple', width=0.15)
    if projects.loc[p,'p_value_control_harvest'] < 0.05:
        ax3.text(i+0.2, max(projects.loc[p,'control_pre_harvest'], projects.loc[p,'control_post_harvest'])*100,'*', fontsize=12, color='purple')
        
    i = i+1
    
ax3.set_xticks(range(len(projects)))
ax3.set_xticklabels(projects.index, rotation=35, ha='right', rotation_mode='anchor')
ax3.set_ylabel('Harvest (fractional area per year)')
ax3.set_ylim((0,2.2))
ax3.text(0.01,0.9,'(c)',fontsize=12, fontweight='bold', transform=ax3.transAxes)
import matplotlib.ticker as mtick
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))


for i in range(len(projects)-1):
    ax1.axvline(i + 0.5, linewidth=0.4, color='0.6')
    ax2.axvline(i + 0.5, linewidth=0.4, color='0.6')
    ax3.axvline(i + 0.5, linewidth=0.4, color='0.6')





#second figure bars by landowner type -------------------------------------------------
fig = plt.figure(figsize=(8,10), tight_layout=True)
categories = ['other','timber']

#emapr
ax1 = fig.add_subplot(311)
i = 0
for c in categories:
    #see if there are any significant
    subset = projects.loc[projects.group==c,:]
    n = len(subset)
    sqrtn = np.sqrt(n)
    print('emapr projects',c,stats.ttest_rel(subset.project_pre_emapr, subset.project_post_emapr))
    print('emapr surround',c,stats.ttest_rel(subset.surround_pre_emapr, subset.surround_post_emapr))
    print('emapr control',c,stats.ttest_rel(subset.control_pre_emapr, subset.control_post_emapr))
    
    pre_p = np.average(projects[projects.group==c]['project_pre_emapr'], weights = projects[projects.group==c]['area_ha'])
    pre_p_err = np.sqrt(np.average((projects[projects.group==c]['project_pre_emapr']-pre_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_p = np.average(projects[projects.group==c]['project_post_emapr'], weights = projects[projects.group==c]['area_ha'])
    post_p_err = np.sqrt(np.average((projects[projects.group==c]['project_post_emapr']-post_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    pre_s = np.average(projects[projects.group==c]['surround_pre_emapr'], weights = projects[projects.group==c]['area_ha'])
    pre_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_pre_emapr']-pre_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_s = np.average(projects[projects.group==c]['surround_post_emapr'], weights = projects[projects.group==c]['area_ha'])
    post_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_post_emapr']-post_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    #new part
    pre_c = np.average(projects[projects.group==c]['control_pre_emapr'], weights = projects[projects.group==c]['area_ha'])
    pre_c_err = np.sqrt(np.average((projects[projects.group==c]['control_pre_emapr']-pre_c)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_c = np.average(projects[projects.group==c]['control_post_emapr'], weights = projects[projects.group==c]['area_ha'])
    post_c_err = np.sqrt(np.average((projects[projects.group==c]['control_post_emapr']-post_c)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    
    ax1.bar(i-0.375, pre_p, color='salmon', width=0.15, yerr=pre_p_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i-0.225, post_p, color='red', width=0.15, yerr=post_p_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i-0.075, pre_s, color='0.55', width=0.15, yerr=pre_s_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i+0.075, post_s, color='0.4', width=0.15, yerr=post_s_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i+0.225, pre_c, color='plum', width=0.15, yerr=pre_c_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i+0.375, post_c, color='purple', width=0.15, yerr=post_c_err/sqrtn, ecolor='gray', capsize=5)
    
    ax1.annotate('', xy = (i-0.225, post_p ), xycoords='data',
                 xytext = (i-0.375, pre_p ), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax1.annotate('', xy = (i+0.075, post_s ), xycoords='data',
             xytext = (i-0.075, pre_s ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax1.annotate('', xy = (i+0.375, post_c ), xycoords='data',
             xytext = (i+0.225, pre_c ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    i = i+1

emapr_all_norcal = pd.read_csv(root + 'processed_data/carbon_emapr/norcal_emapr.csv')
pre_n = emapr_all_norcal.loc[:,['b27','b28','b29','b30']].mean() * 0.47
pre_n = stats.linregress(range(len(pre_n)),pre_n ).slope 
post_n = emapr_all_norcal.loc[:,'b31':'b34'].mean() * 0.47
post_n = stats.linregress(range(len(post_n)),post_n ).slope 

ax1.bar(i-0.08, pre_n, color='0.85', width=0.15)
ax1.bar(i+0.08, post_n, color='0.7', width=0.15)    
ax1.annotate('', xy = (i+0.08, post_n ), xycoords='data',
             xytext = (i-0.08, pre_n ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
ax1.text(-0.32,1.5,'*', fontsize=18, fontweight='bold')

legend_elements = [Patch(facecolor='salmon', edgecolor='none',label='Projects, before'),
                   Patch(facecolor='red', edgecolor='none',label='Projects, after'),
                   Patch(facecolor='0.55', edgecolor='none',label='Surroundings, before'),
                   Patch(facecolor='0.4', edgecolor='none',label='Surroundings, after'),
                   Patch(facecolor='plum', edgecolor='none',label='Matched controls, before'),
                   Patch(facecolor='purple', edgecolor='none',label='Matched controls, after'),
                   Patch(facecolor='0.85', edgecolor='none',label='Northern CA, before'),
                   Patch(facecolor='0.7', edgecolor='none',label='Northern CA, after')]
ax1.legend(handles=legend_elements, ncol=2, loc='upper right')
    
ax1.set_xticks(range(3))
labels = ['other\n(n=12)', 'timber\n(n=4)','Northern CA']
ax1.set_xticklabels(labels)
ax1.set_ylim((-0.7,2.1))
#ax1.set_ylabel('Carbon accumulation rate (eMapR) (ton C/ha/y)')
ax1.set_ylabel('eMapR carbon accumulation rate\n(tonC/ha/y)', fontsize=12)
ax1.text(0.01,0.9,'(a)',fontsize=12, fontweight='bold', transform=ax1.transAxes)


#lemma
ax2 = fig.add_subplot(312)
i = 0
for c in categories:
    #see if there are any significant
    subset = projects.loc[projects.group==c,:]
    n = len(subset)
    sqrtn = np.sqrt(n)
    print('lemma projects',c,stats.ttest_rel(subset.project_pre_lemma, subset.project_post_lemma))
    print('lemma surround',c,stats.ttest_rel(subset.surround_pre_lemma, subset.surround_post_lemma))
    print('lemma control',c,stats.ttest_rel(subset.control_pre_lemma, subset.control_post_lemma))
    
    pre_p = np.average(projects[projects.group==c]['project_pre_lemma'], weights = projects[projects.group==c]['area_ha'])
    pre_p_err = np.sqrt(np.average((projects[projects.group==c]['project_pre_lemma']-pre_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_p = np.average(projects[projects.group==c]['project_post_lemma'], weights = projects[projects.group==c]['area_ha'])
    post_p_err = np.sqrt(np.average((projects[projects.group==c]['project_post_lemma']-post_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    pre_s = np.average(projects[projects.group==c]['surround_pre_lemma'], weights = projects[projects.group==c]['area_ha'])
    pre_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_pre_lemma']-pre_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_s = np.average(projects[projects.group==c]['surround_post_lemma'], weights = projects[projects.group==c]['area_ha'])
    post_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_post_lemma']-post_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    #new part
    pre_c = np.average(projects[projects.group==c]['control_pre_lemma'], weights = projects[projects.group==c]['area_ha'])
    pre_c_err = np.sqrt(np.average((projects[projects.group==c]['control_pre_lemma']-pre_c)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_c = np.average(projects[projects.group==c]['control_post_lemma'], weights = projects[projects.group==c]['area_ha'])
    post_c_err = np.sqrt(np.average((projects[projects.group==c]['control_post_lemma']-post_c)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    
    ax2.bar(i-0.375, pre_p, color='salmon', width=0.15, yerr=pre_p_err/sqrtn, ecolor='gray', capsize=5)
    ax2.bar(i-0.225, post_p, color='red', width=0.15, yerr=post_p_err/sqrtn, ecolor='gray', capsize=5)
    ax2.bar(i-0.075, pre_s, color='0.55', width=0.15, yerr=pre_s_err/sqrtn, ecolor='gray', capsize=5)
    ax2.bar(i+0.075, post_s, color='0.4', width=0.15, yerr=post_s_err/sqrtn, ecolor='gray', capsize=5)
    ax2.bar(i+0.225, pre_c, color='plum', width=0.15, yerr=pre_c_err/sqrtn, ecolor='gray', capsize=5)
    ax2.bar(i+0.375, post_c, color='purple', width=0.15, yerr=post_c_err/sqrtn, ecolor='gray', capsize=5)
    
    ax2.annotate('', xy = (i-0.225, post_p ), xycoords='data',
                 xytext = (i-0.375, pre_p ), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax2.annotate('', xy = (i+0.075, post_s ), xycoords='data',
             xytext = (i-0.075, pre_s ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax2.annotate('', xy = (i+0.375, post_c ), xycoords='data',
             xytext = (i+0.225, pre_c ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    i = i+1

lemma_all_norcal = pd.read_csv(root + 'processed_data/carbon_lemma/norcal_lemma.csv')
pre_n = lemma_all_norcal.loc[:,'2010_b1':'2013_b1'].mean() * 0.47/1000
pre_n = stats.linregress(range(len(pre_n)),pre_n ).slope 
post_n = lemma_all_norcal.loc[:,'2014_b1':'2017_b1'].mean() * 0.47/1000
post_n = stats.linregress(range(len(post_n)),post_n ).slope 

ax2.bar(i-0.08, pre_n, color='0.85', width=0.15)
ax2.bar(i+0.08, post_n, color='0.7', width=0.15)    
ax2.annotate('', xy = (i+0.08, post_n ), xycoords='data',
             xytext = (i-0.08, pre_n ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    
ax2.set_xticks(range(3))
labels = ['other\n(n=12)', 'timber\n(n=4)','Northern CA']
ax2.set_xticklabels(labels)
ax2.set_ylim((-0.8,2.1))
ax2.set_ylabel('LEMMA carbon accumulation rate\n(tonC/ha/y)', fontsize=12)
ax2.text(0.01,0.9,'(b)',fontsize=12, fontweight='bold', transform=ax2.transAxes)


#harvest
ax3 = fig.add_subplot(313)
i = 0
for c in categories:
    subset = projects.loc[projects.group==c,:]
    print('harvest projects',c,stats.ttest_rel(subset.project_pre_harvest, subset.project_post_harvest))
    print('harvest surround',c,stats.ttest_rel(subset.surround_pre_harvest, subset.surround_post_harvest))
    print('harvest surround',c,stats.ttest_rel(subset.control_pre_harvest, subset.control_post_harvest))
    n = len(subset)
    sqrtn = np.sqrt(n)
    
    pre_p = np.average(projects[projects.group==c]['project_pre_harvest'], weights = projects[projects.group==c]['area_ha'])
    pre_p_err = np.sqrt(np.average((projects[projects.group==c]['project_pre_harvest']-pre_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)

    post_p = np.average(projects[projects.group==c]['project_post_harvest'], weights = projects[projects.group==c]['area_ha'])
    post_p_err = np.sqrt(np.average((projects[projects.group==c]['project_post_harvest']-post_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)

    pre_s = np.average(projects[projects.group==c]['surround_pre_harvest'], weights = projects[projects.group==c]['area_ha'])
    pre_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_pre_harvest']-pre_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)

    post_s = np.average(projects[projects.group==c]['surround_post_harvest'], weights = projects[projects.group==c]['area_ha'])
    post_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_post_harvest']-post_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)

    pre_c = np.average(projects[projects.group==c]['control_pre_harvest'], weights = projects[projects.group==c]['area_ha'])
    pre_c_err = np.sqrt(np.average((projects[projects.group==c]['control_pre_harvest']-pre_c)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)

    post_c = np.average(projects[projects.group==c]['control_post_harvest'], weights = projects[projects.group==c]['area_ha'])
    post_c_err = np.sqrt(np.average((projects[projects.group==c]['control_post_harvest']-post_c)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)


    ax3.bar(i-0.375, pre_p*100, color='salmon', width=0.15, yerr=pre_p_err/sqrtn*100, ecolor='gray', capsize=5)
    ax3.bar(i-0.225, post_p*100, color='red', width=0.15, yerr=post_p_err/sqrtn*100, ecolor='gray', capsize=5)
    ax3.bar(i-0.075, pre_s*100, color='0.55', width=0.15, yerr=pre_s_err/sqrtn*100, ecolor='gray', capsize=5)
    ax3.bar(i+0.075, post_s*100, color='0.4', width=0.15, yerr=post_s_err/sqrtn*100, ecolor='gray', capsize=5)
    ax3.bar(i+0.225, pre_c*100, color='plum', width=0.15, yerr=pre_c_err/sqrtn*100, ecolor='gray', capsize=5)
    ax3.bar(i+0.375, post_c*100, color='purple', width=0.15, yerr=post_c_err/sqrtn*100, ecolor='gray', capsize=5)
    
    ax3.annotate('', xy = (i-0.225, post_p*100 ), xycoords='data',
                 xytext = (i-0.375, pre_p*100 ), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax3.annotate('', xy = (i+0.075, post_s*100 ), xycoords='data',
             xytext = (i-0.075, pre_s*100 ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax3.annotate('', xy = (i+0.375, post_c*100 ), xycoords='data',
             xytext = (i+0.225, pre_c*100 ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    i = i+1
     
harvest_all_norcal = pd.read_csv(root + 'processed_data/harvest/norcal_harvest.csv')
pre_n = harvest_all_norcal.loc[:,'b25':'b31'].mean().mean()
post_n = harvest_all_norcal.loc[:,'b31':'b37'].mean().mean()
ax3.bar(i-0.08, pre_n*100, color='0.85', width=0.15)
ax3.bar(i+0.08, post_n*100, color='0.7', width=0.15)    
ax3.annotate('', xy = (i+0.08, post_n*100 ), xycoords='data',
             xytext = (i-0.08, pre_n*100 ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))


ax3.set_xticks(range(3))
ax3.set_xticklabels(labels)
ax3.set_ylabel('Harvest (fractional area per year)', fontsize=12)
ax3.text(0.01,0.9,'(c)',fontsize=12, fontweight='bold', transform=ax3.transAxes)

import matplotlib.ticker as mtick
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

