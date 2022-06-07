#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: generate Fig 6 with harvest and species for Green Diamond and Sierra Pacific properties vs. other landholdings

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scipy import stats
import matplotlib.ticker as mtick

root = '/Users/scoffiel/california/offsets/'

spi_projects = ShapelyFeature(Reader(root + "processed_data/shapefiles/spi_projects/spi_projects.shp").geometries(), ccrs.PlateCarree())
spi = ShapelyFeature(Reader(root + "processed_data/shapefiles/spi/spi.shp").geometries(), ccrs.PlateCarree())
gd_projects = ShapelyFeature(Reader(root + "processed_data/shapefiles/gd_projects/gd_projects.shp").geometries(), ccrs.PlateCarree())
gd = ShapelyFeature(Reader(root + "processed_data/shapefiles/gd/gd.shp").geometries(), ccrs.PlateCarree())

supersections = ShapelyFeature(Reader(root + "shapefiles/supersections/Supersections_4326.shp").geometries(), ccrs.PlateCarree())
states = ShapelyFeature(Reader(root + "shapefiles/states/cb_2018_us_state_20m.shp").geometries(), ccrs.PlateCarree())

fig = plt.figure(figsize=(9,12))

grid = gs.GridSpec(3,2, height_ratios=[3,2,2])
ax1 = plt.subplot(grid[0], projection = ccrs.Miller())
ax2 = plt.subplot(grid[1], projection = ccrs.Miller())
ax3 = plt.subplot(grid[2])
ax4 = plt.subplot(grid[3])
ax5 = plt.subplot(grid[4])
ax6 = plt.subplot(grid[5])


#first panel: GD green diamond
ax1.set_extent([235.5,237.5,42.8,44.7], crs=ccrs.Miller())
ax1.add_feature(supersections, edgecolor='darkgray', facecolor='none', linewidth=0.2)
ax1.add_feature(states, edgecolor='0.2', facecolor='none')
ax1.add_feature(gd, edgecolor='none', facecolor='darkblue', linewidth=0.2)
ax1.add_feature(gd_projects, edgecolor='none', facecolor='red')

ax1.text(-0.15,1,'(a)',fontsize=12, fontweight='bold', transform=ax1.transAxes)
ax1.set_xticks([236,237], crs=ccrs.PlateCarree())
ax1.set_yticks([40,41,42], crs=ccrs.PlateCarree())
ax1.set_yticklabels([40,41,42], fontsize=8)
ax1.tick_params(top=True, right=True, labelsize=8)
ax1.text(-124.65,42.35,'$^\circ$N', size=9)
ax1.text(-124.45,42.17,'$^\circ$E', size=9)
ax1.set_title('Green Diamond Resource Company')

# add legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='h', color='w', label='GD project', markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='h', color='w', label='GD non-project', markerfacecolor='darkblue', markersize=10)]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)


#second panel: SPI sierra pacific
ax2.set_extent([235.5,240.5,38.7,44.7], crs=ccrs.Miller())
ax2.add_feature(supersections, edgecolor='darkgray', facecolor='none', linewidth=0.2)
ax2.add_feature(states, edgecolor='0.2', facecolor='none')
ax2.add_feature(spi, edgecolor='none', facecolor='darkblue', linewidth=0.2)
ax2.add_feature(spi_projects, edgecolor='none', facecolor='red')

ax2.text(-0.17,1,'(b)',fontsize=12, fontweight='bold', transform=ax2.transAxes)
ax2.set_xticks([236,238,240], crs=ccrs.PlateCarree())
ax2.set_yticks([38,40,42], crs=ccrs.PlateCarree())
ax2.set_yticklabels([38,40,42], fontsize=8)
ax2.tick_params(top=True, right=True, labelsize=8)
ax2.text(-124.85,38.85,'$^\circ$N', size=9)
ax2.text(-124.54,38.50,'$^\circ$E', size=9)
ax2.set_title('Sierra Pacific Industries')

legend_elements = [Line2D([0], [0], marker='h', color='w', label='SPI projects', markerfacecolor='red', markersize=10),
                   Line2D([0], [0], marker='h', color='w', label='SPI non-project', markerfacecolor='darkblue', markersize=10)]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)



#harvest GD
harvest_gd_projects = pd.read_csv(root + 'processed_data/harvest/gd_projects_harvest.csv')
harvest_gd = pd.read_csv(root + 'processed_data/harvest/gd_harvest.csv')

y1 =[] #gd projects
y2 =[] #gd

for yr in range(2,38):
    y1.append(harvest_gd_projects.loc[0,'b{}'.format(yr)]*100)
    y2.append(harvest_gd.loc[0,'b{}'.format(yr)]*100)

ax3.plot(range(1986,2022), y1, color='red', linewidth=2, label='GD projects')
ax3.plot(range(1986,2022), y2, color='darkblue', linewidth=2, label='GD non-project')
ax3.set_ylabel('Fraction of area harvested')
ax3.set_xlim((1985,2022))
ax3.text(-0.2,1,'(c)',fontsize=12, fontweight='bold', transform=ax3.transAxes)
ax3.legend(fontsize=10)
ax3.set_xlabel('Year')
ax3.set_ylim((0,4))
ax3.grid(zorder=0, linewidth=0.4, color='0.9')
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

y1 = np.array(y1[:-9]) #1986-2012
y2 = np.array(y2[:-9])

print('harvest, GD projects vs. GD', ((y1 - y2)/ y2).mean())
print(stats.ttest_rel(y1, y2))



#harvest SPI
harvest_spi_projects = pd.read_csv(root + 'processed_data/harvest/spi_projects_harvest.csv')
harvest_spi = pd.read_csv(root + 'processed_data/harvest/spi_harvest.csv')

y1 =[] #spi projects
y2 =[] #spi

for yr in range(2,38):
    y1.append(harvest_spi_projects.loc[0,'b{}'.format(yr)]*100)
    y2.append(harvest_spi.loc[0,'b{}'.format(yr)]*100)

ax4.plot(range(1986,2022), y1, color='red', linewidth=2, label='SPI projects')
ax4.plot(range(1986,2022), y2, color='darkblue', linewidth=2, label='SPI non-project')
ax4.set_ylabel('Fraction of area harvested')
ax4.set_xlim((1985,2022))
ax4.text(-0.2,1,'(d)',fontsize=12, fontweight='bold', transform=ax4.transAxes)
ax4.legend(fontsize=10)
ax4.set_xlabel('Year')
ax4.set_ylim((0,4))
ax4.grid(zorder=0, linewidth=0.4, color='0.9')

ax4.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

y1 = np.array(y1[:-9]) #1986-2012
y2 = np.array(y2[:-9])

print('harvest, SPI projects vs. SPI', ((y1 - y2)/ y2).mean())
print(stats.ttest_rel(y1, y2))



#all species pie charts
spp_spi_projects = pd.read_csv(root + 'processed_data/species/spi_projects_species.csv')
spp_spi_projects = spp_spi_projects.loc[:,'BigconeDouglasFir':'WhiteAlder']
spp_spi = pd.read_csv(root + 'processed_data/species/spi_species.csv')
spp_spi = spp_spi.loc[:,'BigconeDouglasFir':'WhiteAlder']
spp_gd_projects = pd.read_csv(root + 'processed_data/species/gd_projects_species.csv')
spp_gd_projects = spp_gd_projects.loc[:,'BigconeDouglasFir':'WhiteAlder']
spp_gd = pd.read_csv(root + 'processed_data/species/gd_species.csv')
spp_gd = spp_gd.loc[:,'BigconeDouglasFir':'WhiteAlder']

table = pd.DataFrame()
table['spi_projects'] = spp_spi_projects.sum()
table['spi'] = spp_spi.sum()
table['gd_projects'] = spp_gd_projects.sum()
table['gd'] = spp_gd.sum()
table.to_csv(root + 'processed_data/species/pie_charts_raw_properties.csv')

table = pd.read_csv(root + 'pie_charts/pie_charts_properties.csv') #then add column for categories
table_spi = table.groupby('label_spi').sum()
table_gd = table.groupby('label_gd').sum()


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
size = 0.4
cmap = plt.get_cmap("Reds")
p_colors = cmap(np.arange(20,230,30))
p_colors = p_colors[[0,3,1,4,2,5,6]]

cmap = plt.get_cmap("Blues")
b_colors = cmap(np.arange(20,230,30))
b_colors = b_colors[[0,3,1,4,2,5,6]]

ax5.pie(table_gd.gd_projects, labels=table_gd.index, autopct='%1.f%%', radius=1, colors=p_colors, wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.85, textprops={'fontsize':10})
ax5.pie(table_gd.gd, autopct='%1.f%%', radius=1-size, colors=b_colors, wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.75, textprops={'fontsize':10})
ax5.text(-2.1,1,'(e)',fontweight='bold', fontsize=12)

ax6.pie(table_spi.spi_projects[[1,4,3,6,8,5,2,0,7]], labels=table_spi.index[[1,4,3,6,8,5,2,0,7]], autopct='%1.f%%', radius=1, colors=p_colors, wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.85, textprops={'fontsize':10})
ax6.pie(table_spi.spi[[1,4,3,6,8,5,2,0,7]], autopct='%1.f%%', radius=1-size, colors=b_colors, wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.75, textprops={'fontsize':10})
ax6.text(-2,1,'(f)',fontweight='bold', fontsize=12)

plt.tight_layout()

plt.savefig(root + 'figures/fig6_GD-SPI-projects.eps')