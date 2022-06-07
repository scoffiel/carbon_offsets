#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bar charts for projects+surrounds before and after. Three versions
- project-by-project eMapR, LEMMA, harvest
- (Fig 7) landowner types eMapR, harvest
- landowner types eMapR, LEMMA, harvest
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Patch
import chowtest

root = '/Users/scoffiel/california/offsets/'

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
    projects.loc[p, 'project_initial_emapr'] = emapr_projects.loc[p, initial_emapr] * 0.47 
    projects.loc[p, 'surround_initial_emapr'] = emapr_surround.loc[p, initial_emapr] * 0.47 
    
    project_pre_lemma = list(lemma_projects.loc[p, pre_range_lemma]*0.47/1000)
    project_post_lemma = list(lemma_projects.loc[p, post_range_lemma].dropna()*0.47/1000)
    surround_pre_lemma = list(lemma_surround.loc[p, pre_range_lemma]*0.47/1000)
    surround_post_lemma = list(lemma_surround.loc[p, post_range_lemma].dropna()*0.47/1000)
    
    projects.loc[p, 'project_pre_lemma'] = stats.linregress(range(len(project_pre_lemma)),project_pre_lemma ).slope
    projects.loc[p, 'project_post_lemma'] = stats.linregress(range(len(project_post_lemma)),project_post_lemma ).slope
    projects.loc[p, 'surround_pre_lemma'] = stats.linregress(range(len(surround_pre_lemma)),surround_pre_lemma ).slope
    projects.loc[p, 'surround_post_lemma'] = stats.linregress(range(len(surround_post_lemma)),surround_post_lemma ).slope
    projects.loc[p, 'project_initial_lemma'] = lemma_projects.loc[p, initial_lemma] * 0.47/1000 
    projects.loc[p, 'surround_initial_lemma'] = lemma_surround.loc[p, initial_lemma] * 0.47/1000 
    
    
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
    

#first figure for all projects - emapr carbon, lemma carbon, and harvest proj vs surrounds (fig s3)
fig, (ax1,ax2, ax3) = plt.subplots(3,1, figsize=(8,10), tight_layout=True)

#ax1 emapr
i = 0
for p in projects.index:
    ax1.bar(i-0.3, projects.loc[p,'project_pre_emapr'], color='salmon', width=0.2)
    ax1.bar(i-0.1, projects.loc[p,'project_post_emapr'], color='red', width=0.2)
    if projects.loc[p,'p_value_project_emapr'] < 0.05:
        ax1.text(i-0.32, max(projects.loc[p,'project_pre_emapr'], projects.loc[p,'project_post_emapr']),'*', fontsize=12, color='red')
    ax1.bar(i+0.1, projects.loc[p,'surround_pre_emapr'], color='0.55', width=0.2)
    ax1.bar(i+0.3, projects.loc[p,'surround_post_emapr'], color='0.4', width=0.2)
    if projects.loc[p,'p_value_surround_emapr'] < 0.05:
        ax1.text(i+0.12, max(projects.loc[p,'surround_pre_emapr'], projects.loc[p,'surround_post_emapr']),'*', fontsize=12, color='0.4')
  
    i = i+1
    
ax1.set_xticks(range(len(projects)))
ax1.set_xticklabels(['']*len(projects))
ax1.set_ylabel('eMapR carbon accumulation rate\n(tonC/ha/y)')
ax1.set_ylim((-3,3.5))
ax1.text(0.02,0.9,'(a)',fontsize=12, fontweight='bold', transform=ax1.transAxes)

legend_elements = [Patch(facecolor='salmon', edgecolor='none',label='Projects, before'),
                   Patch(facecolor='red', edgecolor='none',label='Projects, after'),
                   Patch(facecolor='0.55', edgecolor='none',label='Surroundings, before'),
                   Patch(facecolor='0.4', edgecolor='none',label='Surroundings, after')]
ax1.legend(handles=legend_elements, ncol=2)


#ax2 lemma
i = 0
for p in projects.index:
    ax2.bar(i-0.3, projects.loc[p,'project_pre_lemma'], color='salmon', width=0.2)
    ax2.bar(i-0.1, projects.loc[p,'project_post_lemma'], color='red', width=0.2)
    if projects.loc[p,'p_value_project_lemma'] < 0.05:
        ax2.text(i-0.32, max(projects.loc[p,'project_pre_lemma'], projects.loc[p,'project_post_lemma']),'*', fontsize=12, color='red')
    ax2.bar(i+0.1, projects.loc[p,'surround_pre_lemma'], color='0.55', width=0.2)
    ax2.bar(i+0.3, projects.loc[p,'surround_post_lemma'], color='0.4', width=0.2)
    if projects.loc[p,'p_value_surround_lemma'] < 0.05:
        ax2.text(i+0.12, max(projects.loc[p,'surround_pre_lemma'], projects.loc[p,'surround_post_lemma']),'*', fontsize=12, color='0.4')
  
    i = i+1
    
ax2.set_xticks(range(len(projects)))
ax2.set_xticklabels(['']*len(projects))
ax2.set_ylabel('LEMMA carbon accumulation rate\n(tonC/ha/y)')
ax2.set_ylim((-3,3.5))
ax2.text(0.02,0.9,'(b)',fontsize=12, fontweight='bold', transform=ax2.transAxes)


#ax3 harvest
i = 0
for p in projects.index:
    ax3.bar(i-0.3, projects.loc[p,'project_pre_harvest']*100, color='salmon', width=0.2)
    ax3.bar(i-0.1, projects.loc[p,'project_post_harvest']*100, color='red', width=0.2)
    if projects.loc[p,'p_value_project_harvest'] < 0.05:
        ax3.text(i-0.32, max(projects.loc[p,'project_pre_harvest']*100, projects.loc[p,'project_post_harvest']*100),'*', fontsize=12, color='red')
    ax3.bar(i+0.1, projects.loc[p,'surround_pre_harvest']*100, color='0.55', width=0.2)
    ax3.bar(i+0.3, projects.loc[p,'surround_post_harvest']*100, color='0.4', width=0.2)
    if projects.loc[p,'p_value_surround_harvest'] < 0.05:
        ax3.text(i+0.12, max(projects.loc[p,'surround_pre_harvest']*100, projects.loc[p,'surround_post_harvest']*100),'*', fontsize=12, color='0.4')
    i = i+1
    
ax3.set_xticks(range(len(projects)))
ax3.set_xticklabels(projects.index, rotation=45, ha='right', rotation_mode='anchor')
ax3.set_ylabel('Harvest rate (fractional area per year)')
ax3.set_ylim((0,2.2))
ax3.text(0.02,0.9,'(c)',fontsize=12, fontweight='bold', transform=ax3.transAxes)
import matplotlib.ticker as mtick
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))



#second figure bars by landowner type ---------------------
fig = plt.figure(figsize=(7.3,8), tight_layout=True)
categories = ['other','timber']

ax1 = fig.add_subplot(211)
i = 0
for c in categories:
    #see if there are any significant
    subset = projects.loc[projects.group==c,['project_pre_emapr','project_post_emapr']]
    n = len(subset)
    sqrtn = np.sqrt(n)
    print('emapr projects',c,stats.ttest_rel(subset.project_pre_emapr, subset.project_post_emapr))
    
    subset = projects.loc[projects.group==c,['surround_pre_emapr','surround_post_emapr']]
    print('emapr surround',c,stats.ttest_rel(subset.surround_pre_emapr, subset.surround_post_emapr))
    
    pre_p = np.average(projects[projects.group==c]['project_pre_emapr'], weights = projects[projects.group==c]['area_ha'])
    pre_p_err = np.sqrt(np.average((projects[projects.group==c]['project_pre_emapr']-pre_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_p = np.average(projects[projects.group==c]['project_post_emapr'], weights = projects[projects.group==c]['area_ha'])
    post_p_err = np.sqrt(np.average((projects[projects.group==c]['project_post_emapr']-post_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    pre_s = np.average(projects[projects.group==c]['surround_pre_emapr'], weights = projects[projects.group==c]['area_ha'])
    pre_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_pre_emapr']-pre_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_s = np.average(projects[projects.group==c]['surround_post_emapr'], weights = projects[projects.group==c]['area_ha'])
    post_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_post_emapr']-post_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    ax1.bar(i-0.25, pre_p, color='salmon', width=0.15, yerr=pre_p_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i-0.1, post_p, color='red', width=0.15, yerr=post_p_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i+0.1, pre_s, color='0.55', width=0.15, yerr=pre_s_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i+0.25, post_s, color='0.4', width=0.15, yerr=post_s_err/sqrtn, ecolor='gray', capsize=5)
    
    ax1.annotate('', xy = (i-0.1, post_p ), xycoords='data',
                 xytext = (i-0.25, pre_p ), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax1.annotate('', xy = (i+0.25, post_s ), xycoords='data',
             xytext = (i+0.1, pre_s ), textcoords='data',
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
ax1.text(-0.21,1.4,'*', fontsize=18, fontweight='bold')

legend_elements = [Patch(facecolor='salmon', edgecolor='none',label='Projects, before'),
                   Patch(facecolor='red', edgecolor='none',label='Projects, after'),
                   Patch(facecolor='0.55', edgecolor='none',label='Surroundings, before'),
                   Patch(facecolor='0.4', edgecolor='none',label='Surroundings, after'),
                   Patch(facecolor='0.85', edgecolor='none',label='Northern CA, before'),
                   Patch(facecolor='0.7', edgecolor='none',label='Northern CA, after')]
ax1.legend(handles=legend_elements, ncol=3, loc='upper right')
    
ax1.set_xticks(range(3))
labels = ['other\n(n=12)', 'timber\n(n=4)','Northern CA']
ax1.set_xticklabels(labels)
ax1.set_ylim((-0.6,2.1))
ax1.set_ylabel('Carbon accumulation rate (eMapR) (tonC/ha/y)', fontsize=11)
ax1.text(0.01,0.94,'(a)',fontweight='bold', fontsize=12, transform=ax1.transAxes)


ax2 = fig.add_subplot(212)
i = 0
for c in categories:
    subset = projects.loc[projects.group==c,['project_pre_harvest','project_post_harvest']]
    print('harvest projects',c,stats.ttest_rel(subset.project_pre_harvest, subset.project_post_harvest))
    subset = projects.loc[projects.group==c,['surround_pre_harvest','surround_post_harvest']]
    print('harvest surround',c,stats.ttest_rel(subset.surround_pre_harvest, subset.surround_post_harvest))
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

    ax2.bar(i-0.25, pre_p*100, color='salmon', width=0.15, yerr=pre_p_err/sqrtn*100, ecolor='gray', capsize=5)
    ax2.bar(i-0.1, post_p*100, color='red', width=0.15, yerr=post_p_err/sqrtn*100, ecolor='gray', capsize=5)
    ax2.bar(i+0.1, pre_s*100, color='0.55', width=0.15, yerr=pre_s_err/sqrtn*100, ecolor='gray', capsize=5)
    ax2.bar(i+0.25, post_s*100, color='0.4', width=0.15, yerr=post_s_err/sqrtn*100, ecolor='gray', capsize=5)
    
    ax2.annotate('', xy = (i-0.1, post_p*100 ), xycoords='data',
                 xytext = (i-0.25, pre_p*100 ), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax2.annotate('', xy = (i+0.25, post_s*100 ), xycoords='data',
             xytext = (i+0.1, pre_s*100 ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    i = i+1
     
harvest_all_norcal = pd.read_csv(root + 'processed_data/harvest/norcal_harvest.csv')
pre_n = harvest_all_norcal.loc[:,'b25':'b31'].mean().mean()
post_n = harvest_all_norcal.loc[:,'b31':'b37'].mean().mean()
ax2.bar(i-0.08, pre_n*100, color='0.85', width=0.15)
ax2.bar(i+0.08, post_n*100, color='0.7', width=0.15)    
ax2.annotate('', xy = (i+0.08, post_n*100 ), xycoords='data',
             xytext = (i-0.08, pre_n*100 ), textcoords='data',
             arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))

ax2.set_xticks(range(3))
ax2.set_xticklabels(labels)
ax2.set_ylabel('Harvest (fractional area per year)', fontsize=11)
ax2.text(0.01,0.94,'(b)',fontweight='bold', fontsize=12, transform=ax2.transAxes)

import matplotlib.ticker as mtick
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

#plt.savefig(root + 'figures/fig7_before-after-barcharts.eps')


#3rd figure for supplement - copy of previous but including LEMMA --------------------------------------------

fig = plt.figure(figsize=(7,10), tight_layout=True)
categories = ['other','timber']

#emapr
ax1 = fig.add_subplot(311)
i = 0
for c in categories:
    #see if there are any significant
    subset = projects.loc[projects.group==c,['project_pre_emapr','project_post_emapr']]
    n = len(subset)
    sqrtn = np.sqrt(n)
    print('emapr projects',c,stats.ttest_rel(subset.project_pre_emapr, subset.project_post_emapr))
    
    subset = projects.loc[projects.group==c,['surround_pre_emapr','surround_post_emapr']]
    print('emapr surround',c,stats.ttest_rel(subset.surround_pre_emapr, subset.surround_post_emapr))
    
    pre_p = np.average(projects[projects.group==c]['project_pre_emapr'], weights = projects[projects.group==c]['area_ha'])
    pre_p_err = np.sqrt(np.average((projects[projects.group==c]['project_pre_emapr']-pre_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_p = np.average(projects[projects.group==c]['project_post_emapr'], weights = projects[projects.group==c]['area_ha'])
    post_p_err = np.sqrt(np.average((projects[projects.group==c]['project_post_emapr']-post_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    pre_s = np.average(projects[projects.group==c]['surround_pre_emapr'], weights = projects[projects.group==c]['area_ha'])
    pre_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_pre_emapr']-pre_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_s = np.average(projects[projects.group==c]['surround_post_emapr'], weights = projects[projects.group==c]['area_ha'])
    post_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_post_emapr']-post_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    ax1.bar(i-0.25, pre_p, color='salmon', width=0.15, yerr=pre_p_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i-0.1, post_p, color='red', width=0.15, yerr=post_p_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i+0.1, pre_s, color='0.55', width=0.15, yerr=pre_s_err/sqrtn, ecolor='gray', capsize=5)
    ax1.bar(i+0.25, post_s, color='0.4', width=0.15, yerr=post_s_err/sqrtn, ecolor='gray', capsize=5)
    
    ax1.annotate('', xy = (i-0.1, post_p ), xycoords='data',
                 xytext = (i-0.25, pre_p ), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax1.annotate('', xy = (i+0.25, post_s ), xycoords='data',
             xytext = (i+0.1, pre_s ), textcoords='data',
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
ax1.text(-0.21,1.4,'*', fontsize=18, fontweight='bold')

legend_elements = [Patch(facecolor='salmon', edgecolor='none',label='Projects, before'),
                   Patch(facecolor='red', edgecolor='none',label='Projects, after'),
                   Patch(facecolor='0.55', edgecolor='none',label='Surroundings, before'),
                   Patch(facecolor='0.4', edgecolor='none',label='Surroundings, after'),
                   Patch(facecolor='0.85', edgecolor='none',label='Northern CA, before'),
                   Patch(facecolor='0.7', edgecolor='none',label='Northern CA, after')]
ax1.legend(handles=legend_elements, ncol=3, loc='upper right', fontsize=9)
    
ax1.set_xticks(range(3))
labels = ['other\n(n=12)', 'timber\n(n=4)','Northern CA']
ax1.set_xticklabels(labels)
ax1.set_ylim((-0.8,2.2))
ax1.set_ylabel('eMapR carbon accumulation rate\n(tonC/ha/y)', fontsize=11)
ax1.text(0.01,0.94,'(a)',fontweight='bold', fontsize=12, transform=ax1.transAxes)


#lemma
ax2 = fig.add_subplot(312)
i = 0
for c in categories:
    #see if there are any significant
    subset = projects.loc[projects.group==c,['project_pre_lemma','project_post_lemma']]
    n = len(subset)
    sqrtn = np.sqrt(n)
    print('lemma projects',c,stats.ttest_rel(subset.project_pre_lemma, subset.project_post_lemma))
    
    subset = projects.loc[projects.group==c,['surround_pre_lemma','surround_post_lemma']]
    print('lemma surround',c,stats.ttest_rel(subset.surround_pre_lemma, subset.surround_post_lemma))
    
    pre_p = np.average(projects[projects.group==c]['project_pre_lemma'], weights = projects[projects.group==c]['area_ha'])
    pre_p_err = np.sqrt(np.average((projects[projects.group==c]['project_pre_lemma']-pre_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_p = np.average(projects[projects.group==c]['project_post_lemma'], weights = projects[projects.group==c]['area_ha'])
    post_p_err = np.sqrt(np.average((projects[projects.group==c]['project_post_lemma']-post_p)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    pre_s = np.average(projects[projects.group==c]['surround_pre_lemma'], weights = projects[projects.group==c]['area_ha'])
    pre_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_pre_lemma']-pre_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    post_s = np.average(projects[projects.group==c]['surround_post_lemma'], weights = projects[projects.group==c]['area_ha'])
    post_s_err = np.sqrt(np.average((projects[projects.group==c]['surround_post_lemma']-post_s)**2, weights=projects[projects.group==c]['area_ha'])) / np.sqrt(n)
    
    ax2.bar(i-0.25, pre_p, color='salmon', width=0.15, yerr=pre_p_err/sqrtn, ecolor='gray', capsize=5)
    ax2.bar(i-0.1, post_p, color='red', width=0.15, yerr=post_p_err/sqrtn, ecolor='gray', capsize=5)
    ax2.bar(i+0.1, pre_s, color='0.55', width=0.15, yerr=pre_s_err/sqrtn, ecolor='gray', capsize=5)
    ax2.bar(i+0.25, post_s, color='0.4', width=0.15, yerr=post_s_err/sqrtn, ecolor='gray', capsize=5)
    
    ax2.annotate('', xy = (i-0.1, post_p ), xycoords='data',
                 xytext = (i-0.25, pre_p ), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax2.annotate('', xy = (i+0.25, post_s ), xycoords='data',
             xytext = (i+0.1, pre_s ), textcoords='data',
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
ax2.set_ylim((-0.8,2.2))
ax2.set_ylabel('LEMMA carbon accumulation rate\n(tonC/ha/y)', fontsize=11)
ax2.text(0.01,0.94,'(b)',fontweight='bold', fontsize=12, transform=ax2.transAxes)



#harvest
ax3 = fig.add_subplot(313)
i = 0
for c in categories:
    subset = projects.loc[projects.group==c,['project_pre_harvest','project_post_harvest']]
    print('harvest projects',c,stats.ttest_rel(subset.project_pre_harvest, subset.project_post_harvest))
    subset = projects.loc[projects.group==c,['surround_pre_harvest','surround_post_harvest']]
    print('harvest surround',c,stats.ttest_rel(subset.surround_pre_harvest, subset.surround_post_harvest))
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

    ax3.bar(i-0.25, pre_p*100, color='salmon', width=0.15, yerr=pre_p_err/sqrtn*100, ecolor='gray', capsize=5)
    ax3.bar(i-0.1, post_p*100, color='red', width=0.15, yerr=post_p_err/sqrtn*100, ecolor='gray', capsize=5)
    ax3.bar(i+0.1, pre_s*100, color='0.55', width=0.15, yerr=pre_s_err/sqrtn*100, ecolor='gray', capsize=5)
    ax3.bar(i+0.25, post_s*100, color='0.4', width=0.15, yerr=post_s_err/sqrtn*100, ecolor='gray', capsize=5)
    
    ax3.annotate('', xy = (i-0.1, post_p*100 ), xycoords='data',
                 xytext = (i-0.25, pre_p*100 ), textcoords='data',
                 arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor='black', ec='black'))
    ax3.annotate('', xy = (i+0.25, post_s*100 ), xycoords='data',
             xytext = (i+0.1, pre_s*100 ), textcoords='data',
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
ax3.set_ylabel('Harvest (fractional area per year)', fontsize=11)
ax3.text(0.01,0.94,'(c)',fontweight='bold', fontsize=12, transform=ax3.transAxes)

import matplotlib.ticker as mtick
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
