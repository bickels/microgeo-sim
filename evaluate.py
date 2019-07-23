# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:33:07 2019

@author: bickels
"""
import glob
import numpy as np
import numba as nb
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
    cres = []
    tres = []
    sres = []
    for f in files:
        print f,'.',
        dat = np.load(f, mmap_mode='r')
        rep = float(dat['rep'])
        psi = float(dat['psi'])
        wc = float(dat['wc'])
        cc = float(dat['CC'])
        
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
    dft.to_excel('./out/time.xlsx',index=False)
    
    dfs = pd.concat(sres)
    dfs = dfs.reset_index()
    dfs.to_excel('./out/summary.xlsx',index=False)

print dfs.groupby(['CC','wc']).describe() #CC = carrying capacity, wc = water content