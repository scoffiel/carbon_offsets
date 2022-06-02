#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Purpose: Compare 30 projects' reported accumulation rates, baselines, and credits issued.
Generate stats and Fig S3 illustrating trade-off between high accumulation rate and high initial stocking above baseline 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

root = '/Users/scoffiel/california/offsets/'

projects = pd.read_csv(root + 'all_projects.csv', index_col=0) #should have 37 projects

#exclude projects that transferred from early action
exclude = ['CAR1067','CAR1098','CAR1099','CAR1100','CAR1139','CAR1140','CAR1141']
projects = projects[~projects.index.isin(exclude)]


#calculate mean stock and trend for reported
for p in projects.index:
    start_year= int(projects.loc[p, 'start_year'])
    end_year = int(projects.loc[p, 'final_stocks_year'])

    if end_year - start_year > 0:
        reported = projects.loc[p, ['reported'+str(y) for y in range(start_year, end_year+1)]]
        
        projects.loc[p, 'reported_mean'] = reported.mean() #over period of both satellite and reported record
        projects.loc[p, 'reported_trend'] = stats.linregress(np.where(reported>0), list(reported[reported>0]))[0]
        projects.loc[p, 'reported_change'] = reported[-1] - reported[0]

table = pd.DataFrame()
table['area'] = projects.area_ha
table['trend'] = projects.reported_trend
table['baseline'] = projects.baseline_tonCha
table['baseline_gap'] = projects.initial_stock - projects.baseline_tonCha
table['payoff_time'] = table.baseline_gap/projects.reported_trend
table['group'] = projects.group
colors = {'timber':'#1f78b4','other':'k'} #'miscellaneous':'#a6cee3','conservation':'#33a02c', 'tribe':'#b2df8a', 
table['area'] = projects.area_ha

print('mean baseline gap', np.average(table.baseline_gap, weights=table.area))
burned = ['CAR1046','CAR1174']
table = table[~table.index.isin(burned)]
projects = projects[~projects.index.isin(burned)]
print('mean payout time', np.average(table.payoff_time, weights=table.area))
print('median payout time', np.median(table.payoff_time))


#Fig S3 accumulation ratse vs. baseline gap
for p in table.index:
    table.loc[p, 'color'] = colors[table.loc[p,'group']]

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot()
ax.scatter(table.baseline_gap, table.trend, c=table.color, s=table.area/200)
ax.axhline(0, color='gray',linestyle='--', linewidth=0.5)
ax.axvline(0, color='gray',linestyle='--', linewidth=0.5)
ax.set_xlabel('Initial stock minus baseline (ton C/ha)')
ax.set_ylabel('Carbon accumulation rate (ton C/ha/y)')
#ax.text(10,-11,'burned', fontsize=8)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='large timber\ncompanies', markerfacecolor=colors['timber'], markersize=8),
                   Line2D([0], [0], marker='o', color='w', label='other', markerfacecolor=colors['other'], markersize=8)]
                   
ax.legend(handles=legend_elements, fontsize=8, loc='lower left')



#calculate payoff times based on actual crediting -----------------------------
projects = pd.read_csv(root + 'all_projects.csv', index_col=0) #should have 37 projects

#exclude 7 projects that transferred from early action and one that burned
exclude = ['CAR1067','CAR1098','CAR1099','CAR1100','CAR1139','CAR1140','CAR1141', 'CAR1046']
projects = projects[~projects.index.isin(exclude)]

project_ids = projects.index

table = pd.read_csv(root + 'arboc_issuance_0122.csv')

payback_times = []
for p in project_ids:
    subset = table[table['OPR Project ID']==p]
    n = len(subset)
    if n > 1:
        c0 = subset.loc[subset['CARB Issuance ID'].str[-1]=='A', 'ARB Offset Credits Issued']
        c1 = subset.loc[subset['CARB Issuance ID'].str[-1]!='A', 'ARB Offset Credits Issued'].dropna()
        #print(subset)
        payback = c0.astype(float).mean() / (c1.astype(float).sum()/(n-1))
        payback_times.append(payback)
    

payback_times = np.array(payback_times)
mean = np.nanmean(payback_times)
med = np.nanmedian(payback_times)
stderr = np.nanstd(payback_times) / np.sqrt(len(payback_times))
print('mean payout time based on real crediting', mean, '+/-', stderr)