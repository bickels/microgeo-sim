# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:40:54 2016

@author: bickels
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import numba as nb

#%%
def lognorm(x, mu, sigma):
    return 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))

def fracPSD(msilt, mclay):
    Dsilt = 3-(np.log(1+(msilt/mclay))/np.log(25))
    d50 = np.exp(((2-Dsilt)*np.log(2)-np.log(mclay))/(3-Dsilt))
    return Dsilt, d50

def fracPOR(por, Rmin, Rmax):
    return 3-(np.log(1-por)/np.log(Rmin/Rmax))

def surfaceA_tex(tex, rhosoil, rhopart=2.65e6):
    fsand,fsilt,fclay = tex
    #rhosoil = 1.35e6 #g m**-3
    rsand = 1e-3 #m
    rsilt = 2.5e-5 #m
    rclay = 1e-6 #m

    pVsand = 4/3.*np.pi*rsand**3
    pVsilt = 4/3.*np.pi*rsilt**3
    pVclay = rclay**2*np.pi*2e-8
    pVclay2 = rclay**2*np.pi*1e-7

    #Vsolid = rhosoil/rhopart
    Porosity = 1.-(rhosoil/rhopart)
    Vsolid = rhosoil**-1*(1-Porosity)

    Vsand = fsand*Vsolid
    Vsilt = fsilt*Vsolid
    Vclay = fclay*Vsolid

    Gsand = Vsand/pVsand
    Gsilt = Vsilt/pVsilt
    Gclay = Vclay/(pVclay+pVclay2)

    pAsand = 4*np.pi*rsand**2
    pAsilt = 4*np.pi*rsilt**2
    pAclay = 2*np.pi*rclay**2+2*np.pi*rclay*2e-8
    pAclay2 = 2*np.pi*rclay**2+2*np.pi*rclay*1e-7

    Asand = Gsand*pAsand
    Asilt = Gsilt*pAsilt
    Aclay = Gclay*np.mean([pAclay,pAclay2])
    return Asand+Asilt+Aclay

@nb.jit
def mean_curvature(Z,dx=1):
    Zy, Zx  = np.gradient(Z,dx)
    Zxy, Zxx = np.gradient(Zx,dx)
    Zyy, _ = np.gradient(Zy,dx)
    H = (Zx**2 + 1)*Zyy - 2*Zx*Zy*Zxy + (Zy**2 + 1)*Zxx
    H = -H/(2*(Zx**2 + Zy**2 + 1)**(1.5))
    return H

'''Computation of Surface Curvature from Range Images Using Geometrically Intrinsic Weights"*, T. Kurita and P. Boulanger, 1992.'''
def gaussian_curvature(Z,dx=1):
    Zy, Zx = np.gradient(Z,dx)
    Zxy, Zxx = np.gradient(Zx,dx)
    Zyy, _ = np.gradient(Zy,dx)
    K = (Zxx * Zyy - (Zxy ** 2)) /  (1 + (Zx ** 2) + (Zy **2)) ** 2
    return K

@nb.jit
def min_int(Z,psi,sigma, dx=1.):
    dW = (2*mean_curvature(Z, dx=dx)-psi/sigma)
    return dW

def meniscus(psi, sigma, theta, factor=1.):
    R = factor*2*sigma/psi
    a = R*np.sin(theta)
    b = R*np.cos(theta)
    return R, a, b

def radial_mask(a, b, nx, ny, r=1):
    xm, ym = np.ogrid[-a:nx-a, -b:ny-b]
    return xm*xm + ym*ym <= r*r

def spherical_kernel(R,points):
    x = np.linspace(-1,1,points)
    X,Y = np.meshgrid(x,x)
    return np.sqrt(R**2-X**2-Y**2)

feps = np.finfo(np.float).eps
@nb.jit
def wft_iter(ls, psi, sigma=0.07275, dx=1e-5):
    R, _, _, = meniscus(psi*1e4, sigma, np.deg2rad(20),factor=1)

    rpx = int(round(abs(R/dx)))
#    if rpx%2 == 0:
#        rpx +=1
    l = rpx*2+1
    ls = np.pad(ls,rpx, 'wrap')
    # FIXME: all wierd
    
    fp = radial_mask(l/2,l/2,l,l,r=rpx)
    spherical = spherical_kernel(rpx,l)-rpx
    print '...eroding.'
    wft0 = ndi.morphology.grey_erosion(ls,footprint=fp, structure=spherical, mode='wrap')
#    wft0 = np.full_like(ls,ls.max())
    
    plt.imshow(wft0)
    plt.colorbar(format='%.0e')
    plt.show()

    boundary = ndi.gaussian_gradient_magnitude(ls,9)

    boundary[1:-1,1:-1] = 0
    bB = boundary != 0
    wft0[bB] = boundary[bB]
    d0 = dx**2
    term = -psi*1e4/sigma
    contactb = wft0 < ls
    mcurve = mean_curvature(wft0, dx=dx)[~bB*~contactb]    
#    while np.mean(mcurve) > term and np.std(mcurve)**2 < (0.05*term)**2 and np.std(mcurve) != 0.:
    while np.max(mcurve) > term and np.std(mcurve) > np.mean(mcurve):
#    dWi = min_int(wft0,psi*1e4,sigma, dx=dx)
#    while abs(dWi[~bB*~contactb]).any()>feps:
        dWi = min_int(wft0,psi*1e4,sigma, dx=dx)
        wft0 -= dWi*d0
        contactb = wft0 < ls
        wft0[contactb] = ls[contactb]
        wft0[bB] = boundary[bB]
        '''
        plt.imshow(ls[1:-1,1:-1], cmap=mpl.cm.gray)
        plt.contourf(wft0[1:-1,1:-1],cmap=mpl.cm.Blues,alpha=0.5)
        plt.colorbar(format='%.1e')
        plt.show()'''
        mcurve = mean_curvature(wft0, dx=dx)[~bB*~contactb]
        print np.count_nonzero(contactb), np.max(mcurve), np.std(mcurve)#mean_curvature(wft0)[~bB*~contactb])

    plt.imshow(wft0)
    plt.show()
    plt.imshow(wft0-ls)
    plt.show()
    return wft0[rpx:-rpx,rpx:-rpx]

def cell_volume(a, r):
    return np.pi*r**2*(4./3.*r+a)

def integrate(f, a, b, N, pars):
    x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
    fx = f(x,*pars)
    area = np.sum(fx)*(b-a)/N
    return area

def satpow(x,sat,k):
    return sat/(1+sat*x**k)**(1/(1+k))**k

def POMs(Ls, POM0, POMdens, n):
    POM = np.zeros(n).ravel()
    VPOM = POM0/POMdens
    V_pores = np.pi*4/3*Ls**3

    for ar in np.argsort(V_pores):
        if VPOM-V_pores[ar] > 0:
            POM[ar] = V_pores[ar]*POMdens
            VPOM -= V_pores[ar]
        else:
            POM[ar] = VPOM*POMdens
            print 'all gone'
            break

    POM = POM.reshape(n)

    plt.imshow(POM)
    plt.show()
    plt.hist(POM.ravel(),bins=100)
    plt.show()
    return POM