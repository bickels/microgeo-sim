# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:33:07 2019

@author: bickels
"""
import glob
import numpy as np
import numba as nb
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as ndi
import pandas as pd

#%%
@nb.jit
def multivariate_hypergeometric(m, colors):
    """
    Parameters
    ----------
    m : number balls to draw from the urn
    colors : one-dimensional array of number balls of each color in the urn

    Returns
    -------
    One-dimensional array with the same length as `colors` containing the
    number of balls of each color in a random sample.
    source: https://stackoverflow.com/questions/35734026/numpy-drawing-from-urn
    """

    remaining = np.cumsum(colors[::-1])[::-1]
    result = np.zeros(len(colors), dtype=np.int)
    for i in range(len(colors)-1):
        if m < 1:
            break
        result[i] = np.random.hypergeometric(colors[i], remaining[i+1], m)
        m -= result[i]
    result[-1] = m
    return result

#%%
rarefy = True
nboot = 15
ncnts = 5000
Ntop = 2**9

try:
    df = pd.read_excel('./out/specis.xlsx')
    dft = pd.read_excel('./out/time.xlsx')
    dfs = pd.read_excel('./out/summary.xlsx')
except:
    files = glob.glob(r'./out/*.npz')
    #files = glob.glob(r'Z:\bickels\model\insilico\*.npz')
    #dat = np.load(r'./out/0.5CC_-0.2m_rep0.npz')
    #dat = np.load(r'./out/1.0CC_-0.2m_rep0.npz')
    #dat = np.load(r'./out/1.0CC_-0.5m_rep0.npz')
    cres = []
    tres = []
    sres = []
    for f in files:
        print f,'.',
        dat = np.load(f, mmap_mode='r')
    #    tmp, psi, rep = f.split('_')
        rep = float(dat['rep'])
        psi = float(dat['psi'])
        wc = float(dat['wc'])
        cc = float(dat['CC'])#tmp.split('\\')[-1]
        
        grid = dict(dat['grid'].tolist())
        n = grid['n']
    
        spt = dat['Spt']
        nt = dat['Nt']
        
        p = spt.astype(float)/nt.astype(float)
        p[p==0] = np.nan
        
        R = np.sum(spt>0,axis=0)
        H = np.array([-np.sum(pp[~np.isnan(pp)]*np.log(pp[~np.isnan(pp)])) for pp in p.T])
        
        if rarefy:
            spr = np.array([multivariate_hypergeometric(ncnts,spt[:,-1]) for i in range(nboot)])
            pr = np.mean(spr.astype(float)/ncnts,axis=0)
            
            prNtop = np.sort(pr)[-Ntop:]
            
            Rr = np.sum(pr>0,axis=0)
            Rrtop = np.sum(prNtop>0,axis=0)
            
            pr[pr==0] = np.nan
            Hr = -np.sum(pr[~np.isnan(pr)]*np.log(pr[~np.isnan(pr)]))
            
            prNtop[prNtop==0] = np.nan
            Hrtop = -np.sum(prNtop[~np.isnan(prNtop)]*np.log(prNtop[~np.isnan(prNtop)]))
            
        else:
            Rr = R[-1]
            Hr = H[-1]
            Rrtop = np.nan
            Hrtop = np.nan
        
        mot = dat['motb'].reshape(n)
        clu, ncl = ndi.label(mot,structure=np.ones((3,3)))
        clu = clu.ravel()
        
        NT = len(dat['masses'])
        T = len(nt)
        time = np.arange(T)/(60*24.)
        cr = pd.DataFrame({'wc':[str(wc)+'v/v']*NT,
                           'psi':[psi]*NT,
                           'CC':[str(cc)+'Fc']*NT,
                           'rep':[rep]*NT,
                           'T':[T]*NT,
                           'clu':clu[dat['hexids']],
                           'wft':dat['wft'][dat['hexids']],
                           'Ut':dat['Ut'][dat['hexids'],-1],
                           'masses':dat['masses'], 
                           'sp':dat['cell_color'], 
                           'mumax':dat['mumax'][dat['cell_color']],
                           'Ks':dat['Ks'][dat['cell_color']]})
    
        tr = pd.DataFrame({'wc':[str(wc)+'v/v']*T,
                           'psi':[psi]*T,
                           'CC':[str(cc)+'Fc']*T,
                           'rep':[rep]*T,
                           'T':[T]*T,
                           'Utm':np.mean(dat['Ut'],axis=0),
                           'Utn':np.min(dat['Ut'],axis=0),
                           'Utx':np.max(dat['Ut'],axis=0),
                           'time':time,
                           'Nt':nt,
                           'Rt':R,
                           'Ht':H})
     
        sr = pd.DataFrame({'wc':[str(wc)+'v/v'],
                           'psi':[psi],
                           'CC':[str(cc)+'Fc'],
                           'rep':[rep],
                           'T':[T],
                           'N':[nt[-1]],
                           'R':[Rr],
                           'H':[Hr],
                           'Rtop':[Rrtop],
                           'Htop':[Hrtop]})
        cres.append(cr)
        tres.append(tr)
        sres.append(sr)
        print 'done!'

    df = pd.concat(cres)
    df = df.reset_index()
    df.to_excel('./out/specis.xlsx',index=False)
    
    dft = pd.concat(tres)
    dft = dft.reset_index()
    #dft['wc'] = pd.Categorical(dft['wc'])
    dft.to_excel('./out/time.xlsx',index=False)
    
    dfs = pd.concat(sres)
    dfs = dfs.reset_index()
    #dfs['wc'] = pd.Categorical(dfs['wc'])
    dfs.to_excel('./out/summary.xlsx',index=False)

print dfs.groupby(['CC','wc']).describe()

#%%
wcc = ['0.01v/v','0.05v/v', '0.1v/v', '0.15v/v', '0.2v/v', '0.25v/v', '0.3v/v', '0.35v/v', '0.4v/v', '0.45v/v', '0.5v/v']
ccc = ['0.25Fc','0.5Fc','1.0Fc', '2.0Fc'][::-1]

df['wc'] = pd.Categorical(df['wc'],wcc, ordered=True)
df['Tg'] = np.log(2.)/(df['mumax']*60*60*24)
df['WC'] = [float(ww[0]) for ww in df.wc.str.split('v')]

dft['wc'] = pd.Categorical(dft['wc'],wcc, ordered=True)
dft['WC'] = [float(ww[0]) for ww in dft.wc.str.split('v')]
dft['CC'] = pd.Categorical(dft['CC'],['0.25Fc','0.5Fc','1.0Fc', '2.0Fc'], ordered=True)
dft['Ps'] = dft.Rt/dft.Nt

dfs['WC'] = [float(ww[0]) for ww in dfs.wc.str.split('v')]
dfs['E'] =  np.exp(dfs.H)/dfs.R

dfs['Sd'] =  dfs.R/(1e-3*1e-3*1e-5)
dfs['Sdtop'] =  dfs.Rtop/(1e-3*1e-3*1e-5)
dfs['Cd'] =  dfs.N/(1e-3*1e-3*1e-5)
dfs['CC'] = pd.Categorical(dfs['CC'],['0.25Fc','0.5Fc','1.0Fc', '2.0Fc'], ordered=True)

dfs.to_excel(r'C:\Users\bickels\Documents\Publication\2016-05-19_microgoegraphy\data\final\tab\NUM.xlsx')

#%%
#sns.boxplot('wc','N', hue='CC', data=dfs, order=wcc, dodge=False, boxprops={'facecolor':'None'})
sns.swarmplot('wc','N', hue='CC', palette='magma', data=dfs, order=wcc)#, dodge=False, zorder=-1)
plt.yscale('log')
plt.show()

sns.swarmplot('wc','H', hue='CC', palette='magma',data=dfs, order=wcc)
plt.show()

sns.boxplot('wc','R', hue='CC', palette='magma',data=dfs, order=wcc)
plt.show()

sns.scatterplot('Cd','E', style='CC', hue='wc', data=dfs,palette='Spectral', hue_order=wcc)
plt.xscale('log')
plt.xlabel('Cell density ($m^{-3}$)')
plt.ylabel('Evenness')
plt.show()

#sns.lineplot('time','Nt', hue='psi', style='CC', units='rep', estimator=None, data=dft, ci=None)
sns.lineplot('time','Nt', 
                     hue='wc', 
                     size='CC',
                     size_order=ccc,
#                     style='T',
                     palette='viridis_r',
                     data=dft, 
                     ci=None)
plt.ylabel('Cell numbers')
plt.xlabel('Time (d)')
plt.ylim(5e2,2e5)
plt.yscale('log')
plt.show()

#sns.lineplot('time','Ht', hue='psi', style='CC', units='rep', estimator=None, data=dft, ci=None)
sns.lineplot('time','Ht', 
                     hue='wc', 
                     size='CC',
                     size_order=ccc,
#                     style='T',
                     palette='viridis_r',
                     data=dft, 
                     ci=None)
plt.ylabel('Shannon index')
plt.xlabel('Time (d)')
plt.ylim(5,8.5)
plt.show()

#sns.lineplot('time','Rt', hue='psi', style='CC', units='rep', estimator=None, data=dft, ci=None)
sns.lineplot('time','Rt', 
                     hue='wc', 
                     size='CC',
                     size_order=ccc,
#                     style='T',
                     palette='viridis_r',
                     data=dft, 
                     ci=None)
plt.ylabel('Richness')
plt.xlabel('Time (d)')
plt.ylim(500,4100)
plt.show()

sns.lineplot('Nt','Ps',
                 hue='wc', 
                 size='CC',
                 size_order=ccc,
                 palette='viridis_r',
                 data=dft,
                 ci=None)
plt.show()
plt.xscale('log')
#g = sns.jointplot('Ks','Tg', data=df)
#g.ax_joint.loglog()
sns.lineplot('WC','Cd',hue='CC', palette='magma', data=dfs, ci='sd')
plt.yscale('log')
plt.xlabel('Climatic water content')
plt.ylabel('Cell density ($m^{-3}$)')

#sns.lineplot('WC','R',hue='CC', palette='magma', data=dfs[dfs['T']==23041], ci='sd', hue_order=['0.5Fc','1.0Fc','2.0Fc'])
sns.lineplot('WC','R',hue='CC', palette='magma', data=dfs, ci='sd', hue_order=['0.5Fc','1.0Fc','2.0Fc'])
#sns.lineplot('WC','R',hue='CC', palette='magma', data=dfs[dfs['T']==5761], ci='sd')
#plt.yscale('log')
plt.xlabel('Climatic water content')
plt.ylabel('Richness')
#plt.ylabel('Species density ($m^{-3}$)')
#plt.ylim(1e13,1e15)

sns.lineplot('WC','H',hue='CC', palette='magma', data=dfs, ci='sd')
plt.xlabel('Climatic water content')
plt.ylabel('Shannon diversity')

sns.lineplot('WC','E',hue='CC', palette='magma', data=dfs, ci='sd')
plt.xlabel('Climatic water content')
plt.ylabel('Evenness')

#%%
#gsp = cr.groupby(['wc','sp']).sum().reset_index()
#gsp['cells'] = gsp.masses/2.5e-14
#gsp = gsp.sort_values('cells')

df['cells'] = df.masses/9.3e-13

#gsp = df.groupby('sp').sum().sort_values('cells').reset_index()

sns.lineplot('clu','cells',hue='wc',palette='viridis_r',data=df,estimator=np.sum,ci='sd')
plt.loglog()

#%%
'''
dat = np.load(r'C:/Users/bickels/Documents/GitHub/insilico/out/1.0CC_0.18wc_rep0.npz', mmap_mode='r')
T = len(dat['Nt'])
time = np.arange(T)/(60*24.)
#ddt = pd.DataFrame(np.gradient(dat['Spt'],axis=0).T,index=time).reset_index()
#ddt = pd.melt(ddt,id_vars=['index'])
#sns.lineplot('index','value',hue='variable',data=ddt)

plt.plot(time,(np.gradient(dat['Spt'],axis=0)/dat['Nt']).T)

imu = np.argsort(-dat['Spt'][:,-1],axis=0)[:2]

plt.plot(*dat['Spt'][imu]/dat['Nt'])

#%%

sps = np.sum(spt,axis=1)

spm = np.mean(spt,axis=1)
spv = np.mean(spt,axis=1)**2

ir = np.argsort(-sps)

plt.plot(sps[ir]/sps.sum(),'o')
plt.loglog()
plt.ylim(1e-4,1e-2)
plt.show()

#%%
plt.plot(np.count_nonzero(spt,axis=0),label='Richness')
plt.plot(nt,label='Cells')
#plt.yscale('log')
plt.legend()
'''
#%%
