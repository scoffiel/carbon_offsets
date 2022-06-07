#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: generate Fig 5 with tanoak/redwood species comparison for Northern Coast projects

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

root = '/Users/scoffiel/california/offsets/'

projects = pd.read_csv(root + 'all_projects.csv', index_col=0) #37 projects
projects = projects.sort_values('area_ha', ascending=False) #sort columns alphabetically so emapr, lemma years together
projects = projects[projects.dominant_ssection=='Northern California Coast']
norcoast_projects = projects.index

#all species pie charts
spps_projects = pd.read_csv(root + 'processed_data/species/projects_species.csv', index_col='project_id')#.sort_values('area', ascending=False)
spps_projects = spps_projects.loc[norcoast_projects,'BigconeDouglasFir':'WhiteAlder']
del spps_projects['OPO']

spps_surround = pd.read_csv(root + 'processed_data/species/surroundings_species.csv', index_col='project_id')
spps_surround = spps_surround.loc[norcoast_projects,'BigconeDouglasFir':'WhiteAlder']
del spps_surround['OPO']

spps_norcoast = pd.read_csv(root + 'processed_data/species/norcoast_species.csv')
spps_norcoast = spps_norcoast.loc[:,'BigconeDouglasFir':'WhiteAlder']

ids = spps_projects.index


fig, (ax1,ax2,ax3) = plt.subplots(3,1, gridspec_kw={'height_ratios':[2,2,3]}, figsize=(8,11))
i = 0
for p in ids:
    ax1.bar(i-0.15, spps_projects.loc[p,'Tanoak'] / spps_projects.sum(axis=1)[p], color='red', width=0.3)
    ax1.bar(i+0.15, spps_surround.loc[p,'Tanoak'] / spps_surround.sum(axis=1)[p], color='0.3', width=0.3)
    i = i+1
    
ax1.bar(i, spps_norcoast.loc[0,'Tanoak'] / spps_norcoast.sum().sum(), color='#B3E3CD', width=0.4)
ax1.axhline(spps_norcoast.loc[0,'Tanoak'] / spps_norcoast.sum().sum(), color='gray', linewidth=0.4, linestyle='--')
ax1.set_xticks(range(len(spps_projects)+1))
ax1.set_xticklabels(list(spps_projects.index)+['North.Coast'], rotation=45, ha='right', rotation_mode='anchor')
ax1.set_ylabel('Fraction of carbon in tanoak', fontsize=12)
ax1.set_ylim((0,0.50))

ax1.text(-1,0.46,'(a)',fontweight='bold', fontsize=12)

print(stats.ttest_rel(spps_projects.loc[ids,'Tanoak'], spps_surround.loc[ids,'Tanoak']))



i = 0
for p in ids:
    ax2.bar(i-0.15, spps_projects.loc[p,'Redwood'] / spps_projects.sum(axis=1)[p], color='red', width=0.3)
    ax2.bar(i+0.15, spps_surround.loc[p,'Redwood'] / spps_surround.sum(axis=1)[p], color='0.3', width=0.3)
    i = i+1
    
ax2.bar(i, spps_norcoast.loc[0,'Redwood'] / spps_norcoast.sum().sum(), color='#B3E3CD', width=0.4)
ax2.axhline(spps_norcoast.loc[0,'Redwood'] / spps_norcoast.sum().sum(), color='gray', linewidth=0.4, linestyle='--')
ax2.set_xticks(range(len(spps_projects)+1))
ax2.set_xticklabels(list(spps_projects.index)+['North.Coast'], rotation=45, ha='right', rotation_mode='anchor')
ax2.set_ylabel('Fraction of carbon in redwood', fontsize=12)
ax2.set_ylim((0,0.50))

ax2.text(-1,0.46,'(b)',fontweight='bold', fontsize=12)

print(stats.ttest_rel(spps_projects.loc[ids,'Redwood'], spps_surround.loc[ids,'Redwood']))



#all species pie charts
area = projects.area_ha
area = area / area.sum() #for weighted average
spps_projects = spps_projects.multiply(area, axis='index')
spps_surround = spps_surround.multiply(area, axis='index')

table = pd.DataFrame()
table['projects'] = spps_projects.sum()
table['surround'] = spps_surround.sum()
table['norcoast'] = spps_norcoast.sum()

#table.to_csv(root + 'processed_data/species/pie_charts_raw.csv') #add group labels manually

table = pd.read_csv(root + 'processed_data/species/pie_charts.csv')
table = table.groupby('label').sum()


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = list(table.index)
labels[labels.index('Douglas fir')] = 'Douglas-fir'

size = 0.3
cmap = plt.get_cmap("Reds")
p_colors = cmap(np.arange(20,230,30))
cmap = plt.get_cmap("Greys")
b_colors = cmap(np.arange(20,230,30))
cmap = plt.get_cmap("BuGn")
n_colors = cmap(np.arange(1,7)*25)

ax3.pie(table.projects, labels=labels, autopct='%1.f%%', radius=1, colors=p_colors, wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.85, textprops={'fontsize':10})
ax3.pie(table.surround, autopct='%1.f%%', radius=1-size, colors=b_colors, wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.75, textprops={'fontsize':10})
ax3.pie(table.norcoast, autopct='%1.f%%', radius=0.7-size, colors=n_colors, wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.75, textprops={'fontsize':10})

# add legend
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor='red', edgecolor='None',label='Projects'),
                   Patch(facecolor='0.4', edgecolor='None',label='Surroundings'),
                   Patch(facecolor='#B3E3CD', edgecolor='None',label='Northern Coast')]
ax3.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4,0.8), framealpha=0.9, fontsize=11)

ax3.text(-2.35,1,'(c)',fontweight='bold', fontsize=12)

#ax2.set(aspect="equal")
plt.tight_layout()

plt.savefig(root + 'figures/fig5_spps_comparison.eps')
