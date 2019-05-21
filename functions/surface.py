# -*- coding: utf-8 -*-
"""
Created on Wed May 04 23:39:34 2016

@author: bickels
"""
import numpy as np
import numba as nb
#fft2jit = nb.jit(np.fft.fftpack.ifft2)
@nb.jit(nb.complex64(nb.int32,nb.float32))#, nopython=True,nogil=True
def genSurface(N,fD):
    '''Spectral synthesis method
    '''
    nx,ny = N
    H = 1-(fD-2)
    X = np.zeros((nx,ny),complex)
    A = np.zeros((nx,ny),complex)
    
    powerr = -(H+1.0)/2.0
    
    rx = int(nx/2)+1
    ry = int(ny/2)+1
    rani = range(rx)
    ranj = range(ry)
    ranp = range(1,rx-1)
    rank = range(1,ry-1)
    rndnorm1 = np.random.normal(size=(rx,ry))
    rndnorm2 = np.random.normal(size=(rx,ry))
    rndrand1 = np.random.random(size=(rx,ry))
    rndrand2 = np.random.random(size=(rx,ry))
    for i in rani:
        for j in ranj:
            
            phase=2*np.pi*rndrand1[i,j]#np.random.rand()
            
            if i is not 0 or j is not 0:
                rad=(i*i+j*j)**powerr*rndnorm1[i,j]#np.random.normal()
            else:
                rad=0.0
            
            A[i,j]=np.complex(rad*np.cos(phase),rad*np.sin(phase))
            
            if i is 0:
                i0=0
            else:
                i0=nx-i
            if j is 0:
                j0=0
            else:
                j0=ny-j
                
            A[i0,j0]=np.complex(rad*np.cos(phase),-rad*np.sin(phase))
                
    A.imag[nx/2][0]=0.0
    A.imag[0,ny/2]=0.0
    A.imag[nx/2][ny/2]=0.0
    
    for p in ranp:
        for k in rank:
            phase=2*np.pi*rndrand2[p-1,k-1]#np.random.rand()
            rad=(p*p+k*k)**powerr*rndnorm2[p-1,k-1]#np.random.normal()
            A[p,ny-k]=np.complex(rad*np.cos(phase),rad*np.sin(phase))
            A[nx-p,k]=np.complex(rad*np.cos(phase),-rad*np.sin(phase))
                        
    itemp=np.fft.fftpack.ifft2(A)#fft2jit(A)#
    itemp=itemp-itemp.min()
    X=itemp
    return X
