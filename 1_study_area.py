#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose:
- Plot study area with offset projects and supersections in California
- Figure 2
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scalebar import scale_bar

root = '/Users/scoffiel/california/offsets/'

supersections = Reader(root + "shapefiles/supersections/Supersections_4326.shp")
northcoast = [ss for ss in supersections.records() if ss.attributes['SSection'] == "Northern California Coast"][0]
northcoast = ShapelyFeature([northcoast.geometry], ccrs.PlateCarree())
scascades = [ss for ss in supersections.records() if ss.attributes['SSection'] == "Southern Cascades"][0]
scascades = ShapelyFeature([scascades.geometry], ccrs.PlateCarree())
modoc = [ss for ss in supersections.records() if ss.attributes['SSection'] == "Modoc Plateau"][0]
modoc = ShapelyFeature([modoc.geometry], ccrs.PlateCarree())
sierra = [ss for ss in supersections.records() if ss.attributes['SSection'] == "Sierra Nevada"][0]
sierra = ShapelyFeature([sierra.geometry], ccrs.PlateCarree())

supersections = ShapelyFeature(Reader(root + "shapefiles/supersections/Supersections_4326.shp").geometries(), ccrs.PlateCarree())
projects = ShapelyFeature(Reader(root + "processed_data/shapefiles/all_projects/activeCAprojects_vectorized_4326.shp").geometries(), ccrs.PlateCarree())
states = Reader(root + "shapefiles/states/cb_2018_us_state_20m.shp")
cali = [s for s in states.records() if s.attributes['STATEFP'] == "06"][0]
cali = ShapelyFeature([cali.geometry], ccrs.PlateCarree())
states = ShapelyFeature(Reader(root + "shapefiles/states/cb_2018_us_state_20m.shp").geometries(), ccrs.PlateCarree())


#first panel: map of change plot with offsets overlaid
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection=ccrs.Miller())
ax.set_extent([235.5,241,39.5,44.8], crs=ccrs.Miller())
ax.patch.set_facecolor('#d1ebff')
ax.add_feature(states, edgecolor='0.2', facecolor='0.95')
ax.add_feature(cali, edgecolor='0.2', facecolor='white')
ax.add_feature(supersections, edgecolor='0.7', facecolor='none', linewidth=0.2)
ax.add_feature(northcoast, edgecolor='0.3', facecolor='#b3e2cd', linewidth=0.2)
ax.add_feature(scascades, edgecolor='0.3', facecolor='#fdcdac', linewidth=0.2)
ax.add_feature(modoc, edgecolor='0.3', facecolor='#cbd5e8', linewidth=0.2)
ax.add_feature(sierra, edgecolor='0.3', facecolor='#f4cae4', linewidth=0.2)
ax.add_feature(states, edgecolor='0.2', facecolor='none')
scale_bar(ax, (0.05, 0.07), 50)
ax.add_feature(projects, edgecolor='none', facecolor='0')

#labels for top 9 projects
ax.text(-121.9,44,'CAR1066',size=8)
ax.text(-122.5,43.65,'CAR1041',size=8)
ax.text(-123.43,42.9,'CAR1046',size=8)
ax.text(-123.28,42.14,'ACR173',size=8)
ax.text(-123.65,42.25,'ACR262',size=8)
ax.text(-123.2,41.82,'CAR1095',size=8)
ax.text(-123.22,40.77,'CAR1013',size=8)
ax.text(-123.4,41.05,'ACR182',size=8)
ax.text(-123.22,40.77,'CAR1013',size=8)
ax.text(-124,41.06,'ACR200',size=8)


ax.set_xticks([236,238,240], crs=ccrs.PlateCarree())
ax.set_yticks([38,40,42], crs=ccrs.PlateCarree())
ax.set_yticklabels([38,40,42], fontsize=8)
ax.tick_params(top=True, right=True, labelsize=8)
ax.text(-124.75,39.6,'$^\circ$N', size=9)
ax.text(-124.5,39.35,'$^\circ$E', size=9)

# add legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [Patch(facecolor='#b3e2cd', edgecolor='gray',label='Northern Coast'),
                   Patch(facecolor='#fdcdac', edgecolor='gray',label='Southern Cascades'),
                   Patch(facecolor='#cbd5e8', edgecolor='gray',label='Modoc Plateau'),
                   Patch(facecolor='#f4cae4', edgecolor='gray',label='Sierra Nevada'),
                   Line2D([0], [0], marker='h', color='w', label='Offset projects', markerfacecolor='k', markersize=10)]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.28), framealpha=0.9)

plt.savefig(root + 'figures/fig2_studyarea.eps')
#plt.savefig(root + 'figures/fig2_studyarea.jpg', dpi=300)
