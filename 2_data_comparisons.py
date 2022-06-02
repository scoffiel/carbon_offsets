#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shane Coffield
scoffiel@uci.edu

Compare carbon stocks and accumulation rates: reports vs. eMapR vs. LEMMA
Fig 3 comparing reported to remote sensing time series for 9 projects
Fig S1 validation plots of eMapR/LEMMA stocks/accumulation vs projects
Table S1 with full data for reports, eMapR, LEMMA by project

units: carbon in tons/ha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.gridspec as gs
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW

root = '/Users/scoffiel/california/offsets/'

projects = pd.read_csv(root + 'all_projects.csv', index_col=0) #37 projects

projects['area'] = projects.area_ha
projects = projects.reindex(sorted(projects.columns), axis=1) #sort columns alphabetically so emapr, lemma years together

#populate emapr & lemma sections of table
#emapr goes from 1986-2017, b3-b34, 32 years
#lemma goes from 1986-2017, 1986_b1 to 2017_b1, 32 years
emapr = pd.read_csv(root + 'processed_data/carbon_emapr/projects_emapr.csv', index_col='project_id')
lemma = pd.read_csv(root + 'processed_data/carbon_lemma/projects_lemma.csv', index_col='project_id')

for p in projects.index:
    for yr in range(2012,2018):
        projects.loc[p, 'emapr'+str(yr)] = emapr.loc[p, 'b'+str(yr-1983)] * 0.47 #convert biomass to carbon
        projects.loc[p, 'lemma'+str(yr)] = lemma.loc[p, str(yr)+'_b1'] * 0.47 / 1000 #native units kg/ha biomass

#Fig 3 with reported, emapr, lemma timeseries for 9 projects -------------------------------------
project_ids_9 = ['ACR173','ACR182','ACR200','ACR262','CAR1013','CAR1041','CAR1046','CAR1066','CAR1095']
projects9 = projects.loc[project_ids_9, :].sort_values('area', ascending=False)
projects9 = projects9.reindex(sorted(projects9.columns), axis=1)
years = range(2012,2021)

reported = projects9.loc[:, 'reported2012':'reported2020']
emapr = projects9.loc[:, 'emapr2012':'emapr2017'] 
lemma = projects9.loc[:, 'lemma2012':'lemma2017']

fig = plt.figure(figsize=(10,10), tight_layout=True)
count = 0
for p in projects9.index: #9 projects
    count += 1
    ax = fig.add_subplot(3,3,count)
    
    ax.plot(years, reported.loc[p, :], c='red', label='Project\ndocumentation', linewidth=3)
    ax.plot(years[:-3], emapr.loc[p, :], c='gray', label='eMapR')
    ax.plot(years[:-3], lemma.loc[p, :], c='darkgray', label='LEMMA')
    ax.axhline(projects9.loc[p, 'baseline_tonCha'], linestyle='--')
    ax.set_title(p, fontweight='bold')
    ax.set_ylim(50,190)
    ax.set_xlim(2011.8, 2020.2)
    ax.set_xticks(years)
    ax.tick_params(axis='x', rotation=30)
    
    if p=='CAR1046': ax.text(2015,130,'*', fontsize=14)
    if (count-1)%3==0:
        ax.set_ylabel('Carbon (ton C/ha)')
    if count==3: ax.legend()

plt.savefig(root + 'figures/fig3_datasets.eps')


#calculate mean stock and trend for reported and emapr
for p in projects.index:
    start_year= int(projects.loc[p, 'start_year'])
    end_year = int(projects.loc[p, 'final_stocks_year'])
    end_year = min(2017, end_year) #2017

    reported = projects.loc[p, ['reported'+str(y) for y in range(start_year, end_year+1)]]
    emapr = projects.loc[p, ['emapr'+str(y) for y in range(start_year, end_year+1)]]
    lemma = projects.loc[p, ['lemma'+str(y) for y in range(start_year, end_year+1)]]
    projects.loc[p, 'reported_mean'] = reported.mean() 
    projects.loc[p, 'emapr_mean'] = emapr.mean()
    projects.loc[p, 'lemma_mean'] = lemma.mean()

    if len(emapr) > 1:
        projects.loc[p, 'reported_trend'] = stats.linregress(np.where(reported>0), list(reported[reported>0]))[0]
        projects.loc[p, 'reported_change'] = reported[-1] - reported[0]

        projects.loc[p, 'emapr_trend'] = stats.linregress(range(len(emapr)), list(emapr))[0]
        projects.loc[p, 'emapr_change'] = emapr[-1] - emapr[0]
    
        projects.loc[p, 'lemma_trend'] = stats.linregress(range(len(lemma)), list(lemma))[0]
        projects.loc[p, 'lemma_change'] = lemma[-1] - lemma[0]

projects['total_carbon'] = projects.area * projects.reported_mean #total carbon in MtC in the project

'''
#adjusting RS data for 1-year lag
for p in projects.index:
    start_year= int(projects.loc[p, 'start_year'])
    end_year = int(projects.loc[p, 'final_stocks_year'])
    end_year = min(2016, end_year) 

    reported = projects.loc[p, ['reported'+str(y) for y in range(start_year, end_year+1)]]
    emapr = projects.loc[p, ['emapr'+str(y) for y in range(start_year+1, end_year+2)]]
    lemma = projects.loc[p, ['lemma'+str(y) for y in range(start_year+1, end_year+2)]]
    projects.loc[p, 'reported_mean'] = reported.mean() 
    projects.loc[p, 'emapr_mean'] = emapr.mean()
    projects.loc[p, 'lemma_mean'] = lemma.mean()
    
    if len(emapr)>1:
        
        projects.loc[p, 'reported_trend'] = stats.linregress(np.where(reported>0), list(reported[reported>0]))[0]
        projects.loc[p, 'reported_change'] = reported[-1] - reported[0]
    
        projects.loc[p, 'emapr_trend'] = stats.linregress(range(len(emapr)), list(emapr))[0]
        projects.loc[p, 'emapr_change'] = emapr[-1] - emapr[0]
    
        projects.loc[p, 'lemma_trend'] = stats.linregress(range(len(lemma)), list(lemma))[0]
        projects.loc[p, 'lemma_change'] = lemma[-1] - lemma[0]
    
projects['total_carbon'] = projects.area * projects.reported_mean #total carbon in MtC in the project
'''

#Fig S1 with reported vs RS stocks and trends
fig = plt.figure(figsize=(7,10))

grid = gs.GridSpec(4,2, height_ratios=[3,1,3,1])
ax = plt.subplot(grid[0])
axb = plt.subplot(grid[2])
ax2 = plt.subplot(grid[1])
ax2b = plt.subplot(grid[3])
ax3 = plt.subplot(grid[4])
ax3b = plt.subplot(grid[6])
ax4 = plt.subplot(grid[5])
ax4b = plt.subplot(grid[7])

ax.set_title('Mean C stocks, eMapR', fontweight='bold')
subset = projects[['reported_mean', 'emapr_mean', 'area']].dropna()
x = np.array(subset.reported_mean)
y = np.array(subset.emapr_mean)
w = np.array(subset.area)
ax.scatter(x, y, s=w/100)
X = sm.add_constant(x)
wls_model = sm.WLS(y,X, weights=w)
results = wls_model.fit()
b,m = results.params
ax.plot([0,275], [b, 275*m + b])
ax.plot([0,275],[0,275], c='k', linewidth=1)
ax.set_xlim((-5,275))
ax.set_ylim((-5,275))
ax.set_xlabel('Reported C stock (ton C/ha)')
ax.set_ylabel('eMapR C stock (ton C/ha)')

counts, bins = np.histogram(y-x, bins=np.arange(-100,100,10))
axb.bar(bins[:-1], counts, width=10, align='edge')
axb.set_xlim((-100,100))
axb.set_xlabel('eMapR minus reported (ton C/ha')

ax2.set_title('Mean C stocks, LEMMA', fontweight='bold')
subset = projects[['reported_mean', 'lemma_mean', 'area']].dropna()
x = np.array(subset.reported_mean)
y = np.array(subset.lemma_mean)
w = np.array(subset.area)
ax2.scatter(x, y, s=w/100)
X = sm.add_constant(x)
wls_model = sm.WLS(y,X, weights=w)
results = wls_model.fit()
b,m = results.params
ax2.plot([0,275], [b, 275*m + b])
ax2.plot([0,275],[0,275], c='k', linewidth=1)
ax2.set_xlim((-5,275))
ax2.set_ylim((-5,275))
ax2.set_xlabel('Reported C stock (ton C/ha)')
ax2.set_ylabel('LEMMA C stock (ton C/ha)')

counts, bins = np.histogram(y-x, bins=np.arange(-100,100,10))
ax2b.bar(bins[:-1], counts, width=10, align='edge')
ax2b.set_xlim((-100,100))
ax2b.set_xlabel('LEMMA minus reported (ton C/ha)')

ax3.set_title('Mean C trend, eMapR', fontweight='bold')
subset = projects[['reported_trend', 'emapr_trend', 'area']].dropna()
x = np.array(subset.reported_trend)
y = np.array(subset.emapr_trend)
w = np.array(subset.area)
ax3.scatter(x, y, s=w/100)
X = sm.add_constant(x)
wls_model = sm.WLS(y,X, weights=w)
results = wls_model.fit()
b,m = results.params
ax3.plot([-12,12], [-12*m+b, 12*m+b])
ax3.plot([-12,12],[-12,12], c='k', linewidth=1)
ax3.set_xlim((-12,12))
ax3.set_ylim((-12,12))
ax3.set_xlabel('Reported C trend (ton C/ha/y)')
ax3.set_ylabel('eMapR C trend (ton C/ha/y)')

counts, bins = np.histogram(y-x, bins=np.arange(-6,6,1))
ax3b.bar(bins[:-1], counts, width=1, align='edge')
ax3b.set_xlim((-6,6))
ax3b.set_xlabel('eMapR minus reported (ton C/ha/y)')

ax4.set_title('Mean C trend, LEMMA', fontweight='bold')
subset = projects[['reported_trend', 'lemma_trend', 'area']].dropna()
x = np.array(subset.reported_trend)
y = np.array(subset.lemma_trend)
w = np.array(subset.area)
ax4.scatter(x, y, s=w/100)
X = sm.add_constant(x)
wls_model = sm.WLS(y,X, weights=w)
results = wls_model.fit()
b,m = results.params
ax4.plot([-12,12], [-12*m+b, 12*m+b])
ax4.plot([-12,12],[-12,12], c='k', linewidth=1)
ax4.set_xlim((-12,12))
ax4.set_ylim((-12,12))
ax4.set_xlabel('Reported C trend (ton C/ha/y)')
ax4.set_ylabel('LEMMA C trend (ton C/ha/y)')

counts, bins = np.histogram(y-x, bins=np.arange(-6,6,1))
ax4b.bar(bins[:-1], counts, width=1, align='edge')
ax4b.set_xlim((-6,6))
ax4b.set_xlabel('LEMMA minus reported (ton C/ha/y)')

plt.tight_layout()
#plt.savefig(root + 'figures/figS1_validation.eps')


#print out comparison stats ---------------------------------------------------
subset = projects[['area','reported_mean','emapr_mean', 'lemma_mean']].dropna()
n = len(subset)

#weighted RMSE
print('reported vs emapr', np.sqrt(np.average((subset.reported_mean - subset.emapr_mean)**2, weights=subset.area) / len(subset) ))
print('reported vs lemma', np.sqrt(np.average((subset.reported_mean - subset.lemma_mean)**2, weights=subset.area) / len(subset) ))

#weighted average of reported, emapr, and lemma stocks
print('area-weighted avg reported stock', DescrStatsW(subset.reported_mean, weights=subset.area).mean, '+/-', DescrStatsW(subset.reported_mean, weights=subset.area).std / np.sqrt(n))
print('area-weighted avg emapr stock', DescrStatsW(subset.emapr_mean, weights=subset.area).mean, '+/-', DescrStatsW(subset.emapr_mean, weights=subset.area).std / np.sqrt(n))
print('area-weighted avg lemma stock', DescrStatsW(subset.lemma_mean, weights=subset.area).mean, '+/-', DescrStatsW(subset.lemma_mean, weights=subset.area).std / np.sqrt(n))

#weighted average of reported vs emapr/lemma trends
subset = projects[['area','reported_trend','emapr_trend','lemma_trend']].dropna()
n = len(subset)

print('area-weighted avg reported carbon accumulation rate', DescrStatsW(subset.reported_trend, weights=subset.area).mean, '+/-', DescrStatsW(subset.reported_trend, weights=subset.area).std / np.sqrt(n))
print('area-weighted avg emapr carbon accumulation rate', DescrStatsW(subset.emapr_trend, weights=subset.area).mean, '+/-', DescrStatsW(subset.emapr_trend, weights=subset.area).std / np.sqrt(n))
print('area-weighted avg lemma carbon accumulation rate', DescrStatsW(subset.lemma_trend, weights=subset.area).mean, '+/-', DescrStatsW(subset.lemma_trend, weights=subset.area).std / np.sqrt(n))

#total carbon added by reported vs emapr/lemma
print('reported total carbon added',(projects.reported_change*projects.area).sum())
print('emapr total carbon added', (projects.emapr_change*projects.area).sum())
print('lemma total carbon added',(projects.lemma_change*projects.area).sum())

#export csv table for Table S1 comparing stocks and trends
subset = projects[['area','reported_mean','emapr_mean','lemma_mean','reported_trend','emapr_trend','lemma_trend']].sort_values('project_id')
#subset.to_csv(root + 'processed_data/dataset_comparisons_table.csv')

