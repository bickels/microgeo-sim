# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:43:52 2015

@author: bickels
"""
import sys
import math as mat
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla
import scipy.optimize as opt
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, EllipseCollection
import time as tm
#from skimage.draw import polygon
import numba as nb

#%%IMPORT
from functions import surface
from functions.misc import wft_iter, POMs

#%% SET
plt.rcParams['figure.figsize'] = (10.0, 8.0)
#FIXME: seedy
#np.random.seed(666)

cdict = {'red':   [[0.0,  0.0, 0.25],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.25],
                   [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.25],
                   [1.0,  1.0, 1.0]]}
         
ccmap = mpl.colors.LinearSegmentedColormap('lessgrey',cdict)

#%% FUNCTIONS
def hex_corner(hexcenter, hexside, iccw):
    """ return coordinates of hexagon edge, starting in the east moving
        counterclockwise for pointy top hexagon """
    xx, yy = hexcenter
    angle_deg = 60 * iccw + 90
    angle_rad = mat.pi / 180 * angle_deg
    return (xx + hexside * mat.cos(angle_rad),
            yy + hexside * mat.sin(angle_rad))

def hex_NN(hexcenter):
    """ return hexagonal neighbour coordinates from
        center coordinates (ccw)"""
    col, row = hexcenter
    return [(col+row%2, row+1), (col-1+row%2, row+1), (col-1, row),
            (col-1+row%2, row-1), (col+row%2, row-1), (col+1, row)]

def monod(Mumax, K, S):
    """single substrate effective growth rate. Monod kinetics."""
    return Mumax*S/(K+S)

def substrate_utilization(Mu, X, Y):
    """rate of substrate utilization"""
    return Mu*X/Y

#@jit
def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)#, axis=0)
#    nodes = np.asarray(nodes)
#    dist_2 = np.sum((nodes - node)**2, axis=1)
#    return np.argmin(dist_2)
#def n_closest(x,n,d=1):
#    return x[n[0]-d:n[0]+d+1,n[1]-d:n[1]+d+1]
##    xi = range(n[0]-d, n[0]+d+1)
##    yi = range(n[1]-d, n[1]+d+1)
##    return xi, yi
#@nb.jit(nopython=True, nogil=True)
def radial_mask(a, b, nx, ny, r=1):
    xm, ym = np.ogrid[-a:nx-a, -b:ny-b]
    return xm*xm + ym*ym <= r*r
#    return np.array(mask,dtype='bool')

def MQ_Deff(Ds, wc, wc_s):
    """ Millington, Quirk
        Ds, wc, wc_s
    """
    return Ds*wc**(10.0/3.0)/wc_s**2

#def pm_Deff(Ds, porosity, constrictivity, tortuosity):
#    return (Ds*porosity*constrictivity)/tortuosity

def van_genuchten(ps, wc_s, wc_r, alpha, n):
    r""" van Genuchten parameters:
    notes:
        .. math::
            \psi, \theta_s, \theta_r, \alpha, n
    """
    return wc_r+(wc_s-wc_r)/(1.0+(alpha*abs(ps))**n)**(1.0-1.0/n)

def hmean(a):
    return len(a) / np.sum(1.0/a)

def gradient(Ad, De, S, d):
    return (np.sum(Ad*S,axis=1)-np.diag(De)*S)/d

def film_adsorbed(psi, rho):
    return (-1.9e-19/(6.*mat.pi*rho*psi))**(1./3.)

def radius_contact(psi, sigma, rho):
    return sigma/(-psi*rho)

def hexagonal_pyramid(psi, h, sigma=0.07275,rho=998e3,sub=1.):
    R = (1./sub)*dx/2.
    S = 2.*R/3.**0.5
    r = abs(2*sigma/(psi*rho*9.81))
    if r > R:
        return (3.**0.5/2.*S**2*h)/(Ah/sub)
    else:
        s = 2.*r/3.**0.5
        alpha = np.arctan(R/h)
        hf = r/np.tan(alpha)
        Vhf = 3.**0.5/2.*s**2*hf-4./3.*np.pi*r**3*0.5
        gamma = np.deg2rad(120./2.)
        theta = np.deg2rad(180.-90.-120./2.)
        b = r/np.tan(gamma)
        A = r*b/2.-r**2*theta/2.
        Vh = A*(h-hf)*12.
        As = 3.*R*S+3.*S*(R/np.sin(alpha))-(Ah/sub)
        return (Vh+Vhf+film_adsorbed(psi,rho)*As)/(Ah/sub)

#FIXME: if psi expected positive: uncomment
def effective_film_thickness(psi, beta, L, gamma, rho, sigma):
    wft = np.zeros_like(psi)
    gamma = np.deg2rad(gamma)
    psi *= 9.81
#    pb = psi < 0.0
#    rc = radius_contact(psi[pb], sigma, rho)
#    AF = film_adsorbed(psi[pb], rho)*(beta*L[pb]+2*(L[pb]/np.cos(gamma*0.5)-rc/np.tan(gamma*0.5)))
#    AC = rc**2*(1/np.tan(gamma*0.5)-(np.pi-gamma)*0.5)
#    wft[pb] = (AF+AC)/(L[pb]*(beta+2*np.tan(gamma*0.5)))
#    wft[~pb] = L[~pb]*np.tan(gamma*0.5)/(beta+2*np.tan(gamma*0.5))+AF.max()
    rc = radius_contact(psi, sigma, rho)
    AF = film_adsorbed(psi, rho)*(beta*L+2*(L/np.cos(gamma*0.5)-rc/np.tan(gamma*0.5)))
    AC = rc**2*(1/np.tan(gamma*0.5)-(np.pi-gamma)*0.5)
    wft = (AF+AC)/(L*(beta+2*np.tan(gamma*0.5)))
    return np.clip(wft,0,L*np.tan(gamma*0.5)/(beta+2*np.tan(gamma*0.5))+AF.max())

def rebin(a, shape, mode='mean'):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    if mode=='mean':
        return a.reshape(sh).mean(-1).mean(1)
    elif mode=='sum':
        return a.reshape(sh).sum(-1).sum(1)

def effective_velocity(v, chi, Kd, S):
    S = S.reshape(nx,ny)
    gradSx, gradSy = np.gradient(S, dx, dy)
    gradSn = (gradSx**2+gradSy**2)**0.5
    vx = (2./3.)*v*np.tanh((chi*Kd)*gradSn/(2.*v*(Kd+S)**2))*gradSx/gradSn
    vy = (2./3.)*v*np.tanh((chi*Kd)*gradSn/(2.*v*(Kd+S)**2))*gradSy/gradSn
    #vx[np.isnan(vx)] = 0.
    #vy[np.isnan(vy)] = 0.
    return vx, vy

def osmotic_force(R,T,V,gamma,xs,r):
    return +(R*T/V)*np.log(gamma*xs)*(np.pi*r**2)

def frictional_force(muk, m, g, alpha):
    return -muk*m*g*np.cos(alpha)

def osmotic_potential(dC,R,T):
    return dC*R*T

def capilary_force(R, sigma, psi, d, dd, friction):
    return (2*mat.pi*sigma/R-mat.pi*psi)*(R**2-(d+dd-R)**2)*friction

def capilary_potential(R, sigma, psi, d, dd, friction, m, rho):
    return capilary_force(R, sigma, psi, d, dd, friction)*cell_length(m, rho, R)

def cell_volume(a, r):
    return mat.pi*r**2*(4./3.*r+a)

@nb.jit(nopython=True, nogil=True)
def cell_A_proj2d(a,r):
    return 2*r*a+mat.pi*r**2

@nb.jit(nopython=True, nogil=True)
def cell_length(m, rho, r, tot=True):
    V = m/rho
    return V/(mat.pi*r**2)-4*r/3+2*r*tot

def cell_surface(a, r):
    return 2*mat.pi*r*(2*r+a)

def best_capsule(M, rho, r, S, K, Vm, P):
    l = cell_length(M, rho, r)
    #V = cell_volume(l, r)
    A = cell_surface(l, r)
    J = Vm/A*P
    return Vm*(S+K+J)*(1-(1-4*S*J/(S+K+J)**2)**0.5)/2*J

#@jit#(nopython=True, nogil=True)
#def unit_vector(v, direction2D=False):
#    if direction2D:
#        x = np.cos(v)
#        y = np.sin(v)
#        #v = np.array([x,y])
#        return [x,y]/np.sqrt(x*x + y*y)#(x,y)/np.linalg.norm((x,y))
#    return v/np.sqrt(v[0]*v[0] + v[1]*v[1])#v/np.sqrt(v.dot(v))#v/np.sqrt((v*v).sum(axis=1))#v/np.linalg.norm(v)

@nb.jit(nopython=True, nogil=True)
def py_histogram(samples, bins, range, weights):
    hmin, hmax = range
    hist = np.array([0.]*bins)
    bin_width = (hmax - hmin) / bins
    for element,w in zip(samples,weights):
        bin_index = int((element - hmin) / bin_width)
        if 0 <= bin_index < bins:
            hist[bin_index] += 1*w
        elif element == hmax:
            hist[-1] += 1*w  # Reproduce behavior of numpy
    return hist

@nb.jit(nopython=True, nogil=True)
def unit_vector(v):
    mag = np.sqrt(v[0]*v[0] + v[1]*v[1])
    return v/mag#v[0]/mag,v[1]/mag

@nb.jit(nopython=True, nogil=True)
def dir_vector(ANG):
    x = np.cos(ANG)
    y = np.sin(ANG)
    mag = np.sqrt(x*x + y*y)
    return x/mag,y/mag

@nb.jit(nopython=True, nogil=True)
def py_clip(x, l, u):
    return l if x < l else u if x > u else x

@nb.jit(nopython=True, nogil=True)
def d0(v1,v2):
    """
    d0 is Nominal approach:
    multiply/add in a loop
    """
    #check(v1,v2)
    out = 0
    for k in range(len(v1)):
        out += v1[k] * v2[k]
    return out

@nb.jit(nopython=True, nogil=True)
def closest_points_on_segments(ra,rb,a,b,la,lb, EPSILON = 1e-3):#np.sqrt and allnp. replaced with py
    hla = la/2.
    hlb = lb/2.
    r = rb-ra
    adotr = d0(a,r)
    bdotr = d0(b,r)
    adotb = d0(a,b)
#    adotr = np.dot(a,r)
#    bdotr = np.dot(b,r)
#    adotb = np.dot(a,b)
    denom = 1. - adotb*adotb

    ta = 0.
    tb = 0.

    twopts = False
    if mat.sqrt(abs(denom)) > EPSILON: #test or replace back: np.sqrt(denom)mat.sqrt(abs(denom))
        ta0 = (adotr-bdotr*adotb)/denom
        tb0 = (adotr*adotb-bdotr)/denom
        ona = abs(ta0) < hla
        onb = abs(tb0) < hlb
        if not ona and not onb:
            ca = mat.copysign(hla, ta0)
            cb = mat.copysign(hlb, tb0)
            dddta = 2*(ca-adotb*cb-adotr)
            #dddtb = 2*(cb-adotb*ca+bdotr)
            #if np.sign(dddta) == np.sign(ca):
            if mat.copysign(1,dddta) == mat.copysign(1,ca):
                tb = cb
                #ta = np.clip(tb*adotb+adotr, -hla, hla)
                ta = py_clip(tb*adotb+adotr, -hla, hla)
            else:
                ta = ca
                #tb = np.clip(ta*adotb-bdotr, -hlb, hlb)
                tb = py_clip(ta*adotb-bdotr, -hlb, hlb)
        elif ona and not onb:
            tb = mat.copysign(hlb, tb0)
            #ta = np.clip(tb*adotb+adotr, -hla, hla)
            ta = py_clip(tb*adotb+adotr, -hla, hla)
        elif not ona and onb:
            ta = mat.copysign(hla, ta0)
            #tb = np.clip(ta*adotb-bdotr, -hlb, hlb)
            tb = py_clip(ta*adotb-bdotr, -hlb, hlb)
        else:
            ta = ta0
            tb = tb0
    else:
        xdotr = mat.copysign(min(adotr,bdotr), adotr)
        al = -xdotr-hla
        ar = -xdotr+hla
        il = max(al, -hlb)#check this tunings...
        ir = min(ar, hlb)
        if il > ir:
            if al < -hlb:
                ta = hla
                tb = -hlb
            else:
                ta = -hla
                tb = hlb
            if adotb < 0.:
                tb = -tb
        else:
            twopts = True
            tb = il
            ta = tb + xdotr
            tb2 = ir
            ta2 = tb2 + xdotr
            if adotb < 0.:
                tb = -tb
                tb2 = -tb2
    pa = ra+ta*a
    pb = rb+tb*b
    if twopts:#DANGER!!! HERE BE DRAGONS!!!
        pa2 = ra+ta2*a
        pb2 = rb+tb2*b
        pa[0] = (pa[0]+pa2[0])*0.5
        pa[1] = (pa[1]+pa2[1])*0.5
        pb[0] = (pb[0]+pb2[0])*0.5
        pb[1] = (pb[1]+pb2[1])*0.5
        return pa, pb#, pa2, pb2
    else:
        return pa, pb

#@jit
"""
#PARALEL
def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

#def overlaps_chunk(job_id, data_slice, uv_queue, cp_queue, poss, ANG, lengths, ncells, r0):
def overlaps_chunk(data_slice, poss, ANG, lengths, ncells, r0):
    uv = np.zeros((ncells,2))
    cp = np.zeros((ncells,2))
    dv = unit_vector(ANG,direction2D=True)
    #for cella, cellb in it.combinations(xrange(ncells),2):
        #if cella is not cellb:
    for cella, cellb in data_slice:
        cpts = closest_points_on_segments(  poss[cella],poss[cellb],
                                            dv[:,cella],dv[:,cellb],
                                            lengths[cella],lengths[cellb])
        #print len(cpts)
        try:
            cpa,cpb = cpts
            #dst = spat.distance.pdist(cpts)
            cpx = cpb[0]-cpa[0]
            cpy = cpb[1]-cpa[1]
            dst = mat.sqrt(cpx*cpx + cpy*cpy)
#                print dst
            if (dst < r0*2):
                ov = unit_vector((cpb-cpa))*(dst-r0*2)/2.
                uv[cella] += ov
                uv[cellb] -= ov
                cp[cella] =  poss[cella]-cpa
                cp[cellb] =  poss[cella]-cpb
        except:
            cpa,cpb,cpa1,cpb1 = cpts
            #dst = spat.distance.pdist(cpts[:2])
            cpx = cpb[0]-cpa[0]
            cpy = cpb[1]-cpa[1]
            #dst = mat.sqrt(cpx*cpx + cpy*cpy)
            #dst += spat.distance.pdist(cpts[2:])
            cpx1 = cpb1[0]-cpa1[0]
            cpy1 = cpb1[1]-cpa1[1]
            dst = (mat.sqrt(cpx*cpx + cpy*cpy)+mat.sqrt(cpx1*cpx1 + cpy1*cpy1))*0.5
            #dst *= 0.5
#                print dst
            if (dst < r0*2):
                ov = unit_vector((cpb-cpa))*(dst-r0*2)/2.
                uv[cella] += ov
                uv[cellb] -= ov
                cp[cella] =  poss[cella]
                cp[cellb] =  poss[cella]
                #print 'yey'
    '''uv_queue.put(uv)
    cp_queue.put(cv)'''
    return uv, cp

def overlaps(poss, ANG, lengths, ncells, r0, job_number=4):
    tree = spat.cKDTree(poss)
    data = list(tree.query_pairs(3e-6,2,1e-6))
    total = len(data)
    chunk_size = total / job_number
    slices = chunks(data, chunk_size)
    '''
    uv_queue = mp.Queue()
    cp_queue = mp.Queue()
    jobs = [mp.Process(target=overlaps_chunk, args=(i, s, uv_queue, cp_queue, poss, ANG, lengths, ncells, r0)) for i, s in enumerate(slices)]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    uvs = [uv_queue.get() for j in jobs]
    cps = [cp_queue.get() for j in jobs]
    print uvs,cps'''
    #mp.freeze_support()
    pool = mp.Pool(processes=job_number, maxtasksperchild=5)
    results = [pool.apply(overlaps_chunk, args=(s, poss, ANG, lengths, ncells, r0)) for s in slices]
    pool.close()
    pool.join()
    return results[0][0]+results[1][0], results[0][1]+results[1][1]"""

#SERIAL
np64floateps = np.finfo(float).eps
@nb.jit(nopython=True, nogil=True)#, cache=True)
def collision_looper(pairs,uv,cp,dv,poss,lengths,r0):
    for i in xrange(len(pairs)):
        cella = pairs[i,0]
        cellb = pairs[i,1]
        cpts = closest_points_on_segments(  poss[cella],poss[cellb],
                                            dv[:,cella],dv[:,cellb],
                                            lengths[cella],lengths[cellb])
        cpa,cpb = cpts
        #dst = spat.distance.pdist(cpts)
        cpx = cpb[0]-cpa[0]
        cpy = cpb[1]-cpa[1]
        dst = mat.sqrt(cpx*cpx + cpy*cpy)
#                print dst
        if (dst < r0*2.+np64floateps):
            ov = unit_vector((cpb-cpa))*(dst-r0*2.)/2.
            uv[cella] += ov
            uv[cellb] -= ov
            cp[cella] =  poss[cella]-cpa
            cp[cellb] =  poss[cella]-cpb
    return uv, cp

@nb.jit(nopython=True, nogil=True, target='cpu', forceobj=True)
def overlaps(poss,ANG,lengths,ncells,r0):
    uv = np.zeros((ncells,2))
    cp = np.zeros((ncells,2))
    #dv = unit_vector(ANG,direction2D=True)
    dv = dir_vector(ANG)
    dv = np.array(dv)
    #for cella, cellb in it.combinations(xrange(ncells),2):
        #if cella is not cellb:
    tree = cKDTree(poss)#, leafsize=128)
    pairs = tree.query_pairs(3e-6,2,0.5e-6, output_type='ndarray')
    #pairs = np.array(list(pairs))
    uv,cp = collision_looper(pairs,uv,cp,dv,poss,lengths,r0)
    return uv, cp

'''
def overlaps(poss,ANG,lengths,ncells,r0):
    uv = np.zeros((ncells,2))
    cp = np.zeros((ncells,2))
    #dv = unit_vector(ANG,direction2D=True)
    dv = dir_vector(ANG)
    dv = np.array(dv)
    #for cella, cellb in it.combinations(xrange(ncells),2):
        #if cella is not cellb:
    tree = spat.cKDTree(poss)
    for cella, cellb in tree.query_pairs(3e-6,2,0.5e-6):
        cpts = closest_points_on_segments(  poss[cella],poss[cellb],
                                            dv[:,cella],dv[:,cellb],
                                            lengths[cella],lengths[cellb])
        #print len(cpts)
        try:
            cpa,cpb = cpts
            #dst = spat.distance.pdist(cpts)
            cpx = cpb[0]-cpa[0]
            cpy = cpb[1]-cpa[1]
            dst = mat.sqrt(cpx*cpx + cpy*cpy)
#                print dst
            if (dst < r0*2):
                ov = unit_vector((cpb-cpa))*(dst-r0*2)/2.
                uv[cella] += ov
                uv[cellb] -= ov
                cp[cella] =  poss[cella]-cpa
                cp[cellb] =  poss[cella]-cpb
        except:
            cpa,cpb,cpa1,cpb1 = cpts
            #dst = spat.distance.pdist(cpts[:2])
            cpx = cpb[0]-cpa[0]
            cpy = cpb[1]-cpa[1]
            #dst = mat.sqrt(cpx*cpx + cpy*cpy)
            #dst += spat.distance.pdist(cpts[2:])
            cpx1 = cpb1[0]-cpa1[0]
            cpy1 = cpb1[1]-cpa1[1]
            dst = (mat.sqrt(cpx*cpx + cpy*cpy)+mat.sqrt(cpx1*cpx1 + cpy1*cpy1))*0.5
            #dst *= 0.5
#                print dst
            if (dst < r0*2):
                ov = unit_vector((cpb-cpa))*(dst-r0*2)/2.
                uv[cella] += ov
                uv[cellb] -= ov
                cp[cella] =  poss[cella]
                cp[cellb] =  poss[cella]
                #print 'yey'
    return uv, cp'''

#@jit
def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)
    # ensure stop angle > start angleif tmax < tmin:
    tmax += 2*np.pi
    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)
    # circular mask
    circmask = r2 <= radius*radius
    # angular mask
    anglemask = theta <= (tmax-tmin)
    return circmask*anglemask

def cell_circle(r,N,x,y):
    ang_pc = 360./N
    ANG = np.deg2rad(np.linspace(ang_pc,N*ang_pc,N))
    xs = r*np.cos(ANG)+x
    ys = r*np.sin(ANG)+y
    return xs, ys, ANG

def inoculate(r,N,center,r0=0.):
    cx,cy = center
    rv = 2 * mat.pi * np.deg2rad(np.random.uniform(0,360,N))
    rs = r * np.random.uniform(r0,1,N)+r0
    xs = rs * np.cos(rv) + cx
    ys = rs * np.sin(rv) + cy
    return xs, ys

def Fm(R, nu, V0):
    return 6*mat.pi*R*nu*V0

def Fla(R, nu, V0, lamn, lamp):
    return (1-1/(lamn**2+lamp**2)**0.5)*Fm(R, nu, V0)

#def Fc(R, sigma, gamma1, gamma2):
#    return 2*mat.pi*sigma*R*(np.cos(gamma1)+np.cos(gamma2))

def Fc(R, sigma, psi, d, dd, fc):
    db = d > 2*R
    aFc = (2*mat.pi*sigma/R-mat.pi*abs(psi))*(R**2-(d+dd-R)**2)*fc
    aFc[db] = 0
    aFc[aFc<0] = 0
    return aFc

#FIXME: all..seediff2D_0.26.py
def potential_velocity(R, nu, V0, sigma, psi, d, fc):
    """psi in m"""
    fm = Fm(R, nu, V0)
    dd = -d-(R**2-R**2*np.sin(sigma)**2)**0.5+R
    zz = d-dd
    lamn = 1+9/8.*R/zz+(9/8.*R/zz)**2
    lamp = (1-9/16.*R/zz+1/8.*(R/zz)**3)**-1
    Fbal = fm-Fla(R, nu, V0, lamn, lamp)-Fc(R, sigma, psi, d, dd, fc)
    fb = Fbal < 0.
    vpot = V0*Fbal/fm
    vpot[fb] = 0.
    return vpot

#def svdsolve(a,b):
#    u,s,v = np.linalg.svd(a)
#    c = np.dot(u.T,b)
#    w = np.linalg.solve(np.diag(s),c)
#    x = np.dot(v.T,w)
#    return x

#%% VECTORIZED FUNCTIONS
def gain_masses(M, S, HID, mumax, Ks, Ys, r0=0.5e-6, rho=1.1e6, mass=False):
    HID = HID.astype(int)
    #ni = np.histogram(HID, bins=nx*ny, range=(0,nx*ny))[0]
    ni = py_histogram(HID, bins=nx*ny, range=(0,nx*ny), weights=np.ones_like(HID))
#    mu = monod(mumax, Ks[HID.astype(int)]*ni[HID.astype(int)], S[HID.astype(int)]/ni[HID.astype(int)])
#    mu = monod(mumax, Ks[HID], S[HID]/ni[HID])
    if mass:
        a = cell_length(M, rho, r0, tot=False)
        mu = monod(mumax, Ks*cell_volume(a, r0), S[HID])
    else:
        mu = monod(mumax, Ks, S[HID])
#    mu = monod(mumax, Ks*1e-18, S[HID])
#    mu = best_capsule(M, 1.1e6, 0.5e-6, S[HID]/ni[HID], Ks[HID], mumax, 1e4)
    rs = substrate_utilization(mu, M, Ys)
#    rsi = np.histogram(HID, bins=nx*ny, range=(0,nx*ny), weights=rs)[0]
#    frs = rs/rsi[HID]
#        rs = substrate_utilization(mu[self.hexid], self.mass, self.Ys)
#    nb = np.array(ni,dtype='bool')
#    gb = np.array(Ut[nb, t] - ni[nb]*rs*dt) >= 0
#    if t%360 == 0:
#        print 'rs:', np.mean(rs)/np.mean(Vf), ' mu:',np.mean(mu)
    S *= Vf
    fS = S[HID]/ni[HID]
    S0 = S[HID] - rs*dt
    gmb = S0 >= np64floateps
    
    S[HID] -= rs*dt*gmb
    S[HID] -= S[HID]*~gmb
    
    M[gmb] += rs[gmb]*dt
    M[~gmb] += fS[~gmb]*Ys
#    M *= 1+(S[HID]*frs)*Ys*~gmb
    S /= Vf
    return M, S, HID

def maintains(M, rom, mumax):
    M *= 1-rom*mumax*dt
    return M

def divides(M, m_asym, sd_asym, mtresh, ANG, xs, ys, HID, r0=0.5e-6, rho=1.1e6):
    global cell_color, zstack, vir, ass, mut
    b = np.array(M) > mtresh
#    fg = np.random.normal(m_asym, sd_asym, len(M[b]))
    fg = np.random.uniform(m_asym-sd_asym, m_asym+sd_asym, len(M[b]))
    #fg=0.5
    newM= M[b]*(1.-fg)
    M[b] *= fg
    cl = cell_length(M[b], rho, r0, tot=True)#+1e-7
    cln = cell_length(newM, rho, r0, tot=True)#+1e-7
    M = np.append(M, newM)
    #minx,miny = ndi.minimum_position(Peff)
    #minx,miny = sg.argrelmin(Peff)
    #ANG[b] = np.arctan2(minx,miny)
    #ub = unit_vector(ANG[b],direction2D=True)
    ub = dir_vector(ANG[b])

#    ir = np.cos(ANG[b])*cl#*2e-6
#    jr = np.sin(ANG[b])*cl#*2e-6
#    irn = np.cos(ANG[b])*cln#*2e-6
#    jrn = np.sin(ANG[b])*cln#*2e-6
##    ir = np.random.choice(choices, len(xs[b]))*2e-6
##    jr = np.random.choice(choices, len(ys[b]))*2e-6
#    newxs = xs[b]+irn
#    newys = ys[b]+jrn
#    xs[b] -= ir
#    ys[b] -= jr
    newxs = xs[b]+ub[0]*cln*0.5#ub[0,:]
    newys = ys[b]+ub[1]*cln*0.5
    xs[b] -= ub[0]*cl*0.5
    ys[b] -= ub[1]*cl*0.5
    xs = np.append(xs, newxs)
    ys = np.append(ys, newys)
    newHID = get_hexids(newxs,newys)
    HID = np.append(HID, newHID)
#    ANG = np.append(ANG, np.random.random(size=len(newHID))*2*mat.pi)
    cell_color = np.append(cell_color,cell_color[b])
    zstack = np.append(zstack,zstack[b])
    mut = np.append(mut,vir[b])#la sequenzia
    vir = np.append(vir,vir[b])
    ass = np.append(ass,ass[b])
    ANG = np.append(ANG, ANG[b])
    HID = HID.astype(int)
    return M, ANG, xs, ys, HID

def get_hexids(xs,ys):
    return [closest_node(pos, grid['centers']) for pos in zip(xs,ys)]

# TODO: check if io!
def selections(dtresh):
    global masses, posxs, posys, hexids, angles, cell_color, zstack, vir, ass, Ut, mut
    d = np.array(masses) < dtresh
    nd = np.nonzero(d)
    '''
    Ut[:,t] *= Vf
    Ut[hexids[d],t] += masses[d]
    Ut[:,t] /= Vf'''
    #print 'fatality', len(self.pop[d])
    masses = np.delete(masses, nd)
    posxs = np.delete(posxs, nd)
    posys = np.delete(posys, nd)
    hexids = np.delete(hexids, nd)
    angles = np.delete(angles, nd)
    cell_color = np.delete(cell_color, nd)
    zstack = np.delete(zstack, nd)
    vir = np.delete(vir, nd)
    ass = np.delete(ass, nd)
    mut = np.delete(mut, nd)
    #del self.pop[d]

#@nb.jit(nopython=True,nogil=True)#, target='cpu',forceobj=True)
def individual_trajectory(x0, y0, vpot, vvx0, vvy0):
    px0 = x0/dx
    py0 = y0/dy
#    px0 = np.clip(px0, 0, nx-1)
#    py0 = np.clip(py0, 0, ny-1)
    px0 = py_clip(px0, 0, nx-1)
    py0 = py_clip(py0, 0, ny-1)

    pxt = px0+vvx0*dt/dx
    pyt = py0+vvy0*dt/dy
#    pxt = np.clip(pxt, -nx+1, nx-1)
#    pyt = np.clip(pyt, -ny+1, ny-1)
    pxt = py_clip(pxt, -nx+1, nx-1)
    pyt = py_clip(pyt, -ny+1, ny-1)

    try:#if l != 0:
        l = int(np.hypot(pxt-px0, pyt-py0))

        px = np.linspace(px0, pxt, l)
        py = np.linspace(py0, pyt, l)

        vpi = vpot[px.astype(int), py.astype(int)]
        #v = (vx**2+vy**2)**0.5
        tix = np.cumsum(dx/(vvx0*vpi))
        tiy = np.cumsum(dy/(vvy0*vpi))
        return np.max(px[tix<=dt])*dx, np.max(py[tiy<=dt])*dy
    except:#else:
        return vvx0*dt, vvy0*dt

#@nb.jit(nopython=True,nogil=True)
def chemotaxis(vpot, ANG, xs, ys, HID, S, v0=30e-9):
#    print np.mean(np.log(gradient(Adj, Deg, Ut[:,t], dx)/Ut[:,t]))
#    if t%360 == 0:
#        hex_plot([], grid['corners'], gradient(Adj, Deg, Ut[:,t], dx))
#        plt.show()
#    grads = gradient(Adj, Deg, Ut[:,t], dx) > 2.25e-8
#    gradt = gradient(Adj, Deg, Ut[:,t], dx) < -2.25e-8
    HID = HID.astype(int)
#    grads = np.log((gradient(Adj, Deg, Ut[:,t], dx)+Ut[:,t])/Ut[:,t]) > 9e-11
#    gradt = np.log((gradient(Adj, Deg, Ut[:,t], dx)+Ut[:,t])/Ut[:,t]) < 0.
#    swim = grads[HID]*motb[HID]
    ni = py_histogram(HID, bins=nx*ny, range=(0,nx*ny), weights=np.ones_like(HID))
    mb = ni*2e-12 < Ah

    swim = motb[HID]&mb[HID]
#    tumble = gradt[HID]
    vx,vy = effective_velocity(v0*vpot, 7.5e-4*1e-4, 22.5195, S)#0.125e-3
#    print 'vx: {0:.1e} vy: {1:.1e}'.format(np.mean(vx),np.mean(vy))
    vvx = vx.ravel()[HID]*np.random.uniform(0.0, 2.0, size=HID.shape)
    vvy = vy.ravel()[HID]*np.random.uniform(0.0, 2.0, size=HID.shape)
    try:
        ddx, ddy = np.array([individual_trajectory(x0,y0,vpot,vvx0,vvy0) for x0,y0,vvx0,vvy0 in zip(xs[swim],ys[swim],vvx[swim],vvy[swim])]).T
        xs[swim] += ddx
        ys[swim] += ddy
    except:
        print '¦',#no swimmers',
#    xs[swim] += vvx[swim]*dt
#    ys[swim] += vvy[swim]*dt

#    xs[swim] += np.cos(ANG[swim])*vvx[swim]*dt#*(10*0.5**(dt/10))
#    ys[swim] += np.sin(ANG[swim])*vvy[swim]*dt#*(10*0.5**(dt/10))
#    sbxu = xs[swim] > Lx-dx
#    sbxl = xs[swim] < dx
#    sbyu = ys[swim] > Ly-dy
#    sbyl = ys[swim] < dy
#    xs[sbxu] -= (xs[sbxu] - Lx-dx)
#    xs[sbxl] += (dx - xs[sbxl])
#    ys[sbyu] = (ys[sbyu] - Ly-dy)#periodic
#    ys[sbyl] = Ly-dy-(dy - ys[sbyl])#periodic
#    ys[sbyu] = Ly-dy
#    ys[sbyl] = dy
#    HID[swim] = [pool.apply(closest_node, args=(pos, grid['centers'])) for pos in zip(xs[swim],ys[swim])]
    HID[swim] = get_hexids(xs[swim], ys[swim])
#    ANG[tumble] = np.random.random(size=len(ANG[tumble]))*2*mat.pi
    #ANG[swim] = np.random.random(size=len(ANG[swim]))*2*mat.pi
    #ANG[swim] = np.arctan(vvy[swim]/vvx[swim])
    ANG[swim] = np.arctan2(vvy[swim],vvx[swim])
    return ANG, xs, ys, HID

@nb.vectorize('float64(float64)')
def nb_round(a):
    return np.round(a)

@nb.jit(nopython=True,nogil=True)
def z_shove(hexids, r0, lengths, zs):
    inds = np.arange(len(hexids))
    Aproj = cell_A_proj2d(lengths,r0)
    Ahex = py_histogram(hexids, bins=nx*ny, range=(0,nx*ny), weights=Aproj)
    bhex = py_histogram(hexids, bins=nx*ny, range=(0,nx*ny), weights=np.ones_like(hexids)) <= 1
    zb = Ahex > Ah
    Nz = nb_round(((Ahex[zb]-Ah)/Ahex[zb]))
    nb = Nz > 0
    #zids = np.array([np.random.choice(inds[zb[hexids]], int(nn), replace=False) for nn in Nz[nb]],dtype='int')
    zids = np.array([0]*len(nb))
    for i,nn in enumerate(Nz[nb]):
        zids[i] = np.random.choice(inds[zb[hexids]], int(nn), replace=False)[0]
    zs[zids] += 1
    zs[bhex[hexids]] = 0
    return zs

@nb.jit(nopython=False,nogil=True, target='cpu',looplift=True)
def shove_numerical(poss, ANG, M, ncells, lengths, r0, fcoef=1.):
    dis,cps = overlaps(poss,ANG,lengths,ncells,r0)
    dis[np.isnan(dis)] = 0.
    bdis = np.abs(dis) > np64floateps
    dis[~bdis] = 0.
#    ANG += np.copysign(np.arctan2(np.sqrt((dis*dis).sum(axis=1)),np.sqrt((cps*cps).sum(axis=1))),np.sum(dis,axis=1))
    #nSteps = np.count_nonzero(dis)/20+1
    #fcoef /= nSteps
    #print '    shoving: ',nSteps,np.min(spat.distance.pdist(poss)),np.mean(lengths)#np.sum(np.abs(dis))
    #for step in xrange(1, nSteps + 1, 1):
#    sum0 = np.sum(np.abs(dis))
    '''n=0
    while np.any(bdis) and n < np.count_nonzero(bdis)+1:
#        sum0 = np.sum(np.abs(dis))
        #dis,cps = overlaps(poss,ANG,lengths,ncells,r0)
        #dis[np.isnan(dis)] = 0.
        #ANG += np.copysign(np.arctan2(np.linalg.norm(dis,axis=1),np.linalg.norm(cps,axis=1)),np.sum(dis,axis=1))
        #ANG += np.copysign(np.arctan2(np.sqrt((dis*dis).sum(axis=1)),np.sqrt((cps*cps).sum(axis=1))),np.sum(dis,axis=1))
        dis,cps = overlaps(poss,ANG,lengths,ncells,r0)
        dis[np.isnan(dis)] = 0.
        bdis = np.abs(dis) > np64floateps
        dis[~bdis] = 0.
#        dis[:,0] = (1-dis[:,0])**(3./2.)/lengths*(dt/nSteps)*1e-15
#        dis[:,1] = (1-dis[:,1])**(3./2.)/lengths*(dt/nSteps)*1e-15
        poss[:,0] += dis[:,0]*fcoef
        poss[:,1] += dis[:,1]*fcoef
        n += 1
    '''
    poss[:,0] += dis[:,0]*fcoef
    poss[:,1] += dis[:,1]*fcoef
    dis,cps = overlaps(poss,ANG,lengths,ncells,r0)
    dis[np.isnan(dis)] = 0.
    ANG += np.copysign(np.arctan2(np.sqrt((dis*dis).sum(axis=1)),np.sqrt((cps*cps).sum(axis=1))),np.sum(dis,axis=1))
    return poss, ANG

@nb.jit(nopython=True,nogil=True, target='cpu',forceobj=True)
def nbody_shove(poss, ANG, M, zs, hexids, r0=0.5e-6, rho=1.1e6):
    ncells = len(poss)
    lengths = cell_length(M,rho,r0,tot=False)
    Peff = None
    #poss, Peff, zs, ANG = shove_physical(poss, ANG, M, zs, ncells, lengths, r0)
    poss, ANG = shove_numerical(poss, ANG, M, ncells, lengths, r0)
    zs = z_shove(hexids, r0, lengths, zs)
    #poss *= np.random.normal(1, 1e-6, (ncells,2))
    return poss, Peff, zs, ANG

#@nb.jit(nopython=False,nogil=True, target='cpu',looplift=True)
"""
def shove_physical(poss, ANG, M, zs, ncells, lengths,r0, nSteps=1, dt=60, div=1.):
    dt = dt/nSteps
#    nSteps = 1
#    dt = 60
##    dt2 = dt**2
    #r0 = cell_length(masses,1.1e6,0.5e-6)
    #res = 100
#    res = Lx/(r0/3.)
#    xres = yres = res
    xres = int(Lx/(r0/div))
    yres = int(Ly/(r0/div))
    xstep = Lx/xres
    ystep = Ly/yres

    #u = unit_vector(ANG,direction2D=True)
    u = np.array(dir_vector(ANG))
    XX = np.linspace(0,Lx,xres)
    YY = np.linspace(0,Ly,yres)
    Xc,Yc = np.meshgrid(XX,YY)
    #poss += dis
    Peff = np.zeros_like(Xc)
##    cellvs = np.zeros_like(poss)
    for step in xrange(1, nSteps + 1, 1):
        PosmI = np.zeros_like(Xc)
        PcapI = np.zeros_like(Xc)
        MI = np.zeros_like(Xc)
        #mask = np.zeros_like(Xc,dtype='bool')
        for cell in xrange(ncells):
#            minx,miny = ndi.minimum_position(PosmI)
#            #ANG[b] = sg.argrelmin(np.random.random(10))
#            ANG[cell] = np.arctan2(minx*xstep,miny*ystep)
#            PosmI = np.zeros_like(Xc)
#            PcapI = np.zeros_like(Xc)
            xcell,ycell = poss[cell]
            xcell = np.clip(xcell,xstep,Lx-xstep)
            ycell = np.clip(ycell,ystep,Ly-ystep)

            #tip = np.array((xcell+np.cos(ANG[cell])*lengths[cell]/2., ycell+np.sin(ANG[cell])*lengths[cell]/2.))
            tip = poss[cell]+u[:,cell]*lengths[cell]*0.5
            xtip, ytip = tip
            xtip = np.clip(xtip,xstep,Lx-xstep)
            ytip = np.clip(ytip,ystep,Ly-ystep)
            #end = np.array((xcell-np.cos(ANG[cell])*lengths[cell]/2., ycell-np.sin(ANG[cell])*lengths[cell]/2.))
            end = poss[cell]-u[:,cell]*lengths[cell]*0.5
            xend, yend = end
            xend = np.clip(xend,xstep,Lx-xstep)
            yend = np.clip(yend,ystep,Ly-ystep)
#            xtip2, ytip2 = np.array((xcell+np.cos(angles[cell])*lengths[cell]/4., ycell+np.sin(angles[cell])*lengths[cell]/4.))
#            xend2, yend2 = np.array((xcell-np.cos(angles[cell])*lengths[cell]/4., ycell-np.sin(angles[cell])*lengths[cell]/4.))

            mask = np.zeros_like(Xc,dtype='bool')
            d = np.linalg.norm(end-tip)
            rl = np.array([-(ytip-yend),(xtip-xend)])*r0/d
            rr = np.array([(ytip-yend),-(xtip-xend)])*r0/d
            #corners = np.array([[xtip+r0, ytip+r0],[xtip-r0, ytip-r0],[xend-r0, yend-r0],[xend+r0, yend+r0]])
            corners = np.array([tip+rl,tip+rr,end+rr,end+rl])
            row,col = polygon(corners[:,1]/ystep,corners[:,0]/xstep)
            #row,col = polygon(corners[:,0]/xstep,corners[:,1]/ystep)
            row = np.clip(row,0,xres-1)
            col = np.clip(col,0,yres-1)

            mask[row,col] = True
            #mask += radial_mask(xtip/xstep,ytip/ystep,xres,yres,r=int(r0/xstep))
            #mask += radial_mask(xend/xstep,yend/ystep,xres,yres,r=int(r0/xstep))
            mask += radial_mask(ytip/ystep,xtip/xstep,yres,xres,r=int(r0/xstep))
            #mask += sector_mask(Peff.shape,(ytip/ystep,xtip/xstep), 1.5*int(r0/xstep), (np.rad2deg(ANG[cell])-45,np.rad2deg(ANG[cell])-315))
            mask += radial_mask(yend/ystep,xend/xstep,yres,xres,r=int(r0/xstep))
            #mask += sector_mask(Peff.shape,(yend/ystep,xend/xstep), 1.5*int(r0/xstep), (np.rad2deg(ANG[cell])+135,np.rad2deg(ANG[cell])-135))
            #mask *= ~radial_mask(ycell/ystep,xcell/xstep,yres,xres,r=int(0.5*r0/xstep))
            #mask = ndi.gaussian_filter(mask,r0/xstep)

            rs = ((Xc-xcell)**2+(Yc-ycell)**2)**0.5
            rs += ((Xc-xtip)**2+(Yc-ytip)**2)**0.5
            rs += ((Xc-xend)**2+(Yc-yend)**2)**0.5
##
##            rs += ((Xc-xtip2)**2+(Yc-ytip2)**2)**0.5
##            rs += ((Xc-xend2)**2+(Yc-yend2)**2)**0.5
##            rs = (((Xc-xcell)/(np.cos(angles[cell])*lengths[cell]/2.))**2+((Yc-ycell)/(np.sin(angles[cell])*lengths[cell]/2.))**2-1)**0.5
##            rs = rs.clip(max=r0)#[cell])
            rd = rs/r0#[cell]
##            PosmI += Posm[cell]*(1-rr**2)#*np.exp(-rr)#
##            PcapI += Pcap[cell]*(1-rr**2)#*np.exp(-rr)#
#            PosmI += Posm[cell]*np.exp(-rr)#
#            PcapI += Pcap[cell]*np.exp(-rr)#
            PosmI += Posm[cell]*mask*np.exp(-rd)#(np.exp(-rd)-rd**6)#
            PcapI += Pcap[cell]*mask*np.exp(-rd)#(np.exp(-rd)-rd**6)#
            MI += masses[cell]*mask
#            for c in xrange(ncells):
#                xcell,ycell = poss[c]
#                xtip, ytip = np.array((xcell+np.cos(angles[c])*lengths[c]/2., ycell+np.sin(angles[c])*lengths[c]/2.))
#                xend, yend = np.array((xcell-np.cos(angles[c])*lengths[c]/2., ycell-np.sin(angles[c])*lengths[c]/2.))
#                rs = ((Xc-xcell)**2+(Yc-ycell)**2)**0.5
#                rs += ((Xc-xtip)**2+(Yc-ytip)**2)**0.5
#                rs += ((Xc-xend)**2+(Yc-yend)**2)**0.5
#    #            rs = rs.clip(max=r0)#[c])
#                rr = rs/r0#[c]
#    #            PosmI += Posm[c]*(1-rr**2)#*np.exp(-rr)#
#    #            PcapI += Pcap[c]*(1-rr**2)#*np.exp(-rr)#
#                PosmI += Posm[c]*np.exp(-rr)#
#                PcapI += Pcap[c]*np.exp(-rr)#
#            xcell,ycell = poss[cell]
#            Peff = PosmI+PcapI
#            #Fx, Fy = np.gradient(Peff)
#            Fx = np.diff(Peff,n=1,axis=0)/0.25e-6
#            Fy = np.diff(Peff,n=1,axis=1)/0.25e-6
#            cellvs[cell,0] += dt * -Fx[xcell, ycell]/masses[cell]
#            cellvs[cell,1] += dt * -Fy[xcell, ycell]/masses[cell]
#            poss += 0.5 * cellvs * dt
#            Peff = PosmI+PcapI
#    #        Fx, Fy = np.gradient(Peff)
#            Fx = np.diff(Peff,n=1,axis=0)/0.25e-6
#            Fy = np.diff(Peff,n=1,axis=1)/0.25e-6
#            xid, yid =zip(*[(int(x/xres),int(y/yres)) for x,y in poss])
#            cellvs[:,0] += dt * -Fx[xid, yid]/masses
#            cellvs[:,1] += dt * -Fy[xid, yid]/masses
#            poss += 0.5 * cellvs * dt
        Peff = PosmI+PcapI
        '''labeled, nr_objects = ndi.label(Peff < np.min(Posm))'''
#        Peff = ndi.laplace(Peff)
        #Peff = ndi.minimum_filter(Peff,(3,3))
        #Peff = ndi.maximum_filter(Peff,(3,3))
        #Peff = ndi.gaussian_filter(Peff,1)
##        Fx, Fy = np.gradient(Peff, xstep, ystep)
#        Fx = np.diff(Peff,n=1,axis=0)/xstep
#        Fy = np.diff(Peff,n=1,axis=1)/ystep
#        xid, yid =zip(*[(int(x/xres),int(y/yres)) for x,y in poss])
        xid, yid =zip(*[(int(x/xstep),int(y/ystep)) for x,y in poss])
        xid = np.clip(xid,0,Peff.shape[0]-1)
        yid = np.clip(yid,0,Peff.shape[1]-1)
        hpot = -Peff/(MI*9.81)
        '''
        cellvs[:,0] += dt * -Fx[xid, yid]/M
        cellvs[:,1] += dt * -Fy[xid, yid]/M
#        cellvs[:,0] += dt * -Peff[xid, yid]/masses#*1e-12
#        cellvs[:,1] += dt * -Peff[xid, yid]/masses#*1e-12
#        poss[:,0] += cellvs[:,0] * dt + 0.5 * -Fx[xid, yid]/masses * dt2
#        poss[:,1] += cellvs[:,1] * dt + 0.5 * -Fy[xid, yid]/masses * dt2
        '''
        zupb = hpot[xid,yid] > 2.*r0#(Peff[xid,yid] > -5.5e-12) & (Peff[xid,yid] != 0)
        zs[zupb] += 1.
        '''
        #print np.max(cellvs)*dt

        if abs(cellvs.any())*dt <= abs(lengths.any()/2.):
            poss[:,0] += cellvs[:,0] * dt + 0.5 * -Fx[xid, yid]/M * dt2
            poss[:,1] += cellvs[:,1] * dt + 0.5 * -Fy[xid, yid]/M * dt2
        else:
            #print dis[:,0]
            poss[:,0] += dis[:,0]#np.random.standard_normal(ncells)*r0
            poss[:,1] += dis[:,1]#np.random.standard_normal(ncells)*r0'''
            #print 'oops',  dis
#        minx,miny = ndi.minimum_position(Peff)
#        #ANG[b] = sg.argrelmin(np.random.random(10))
#        ANG = np.arctan2(minx*xstep,miny*ystep)
        #poss += 0.5 * cellvs * dt
#        sbxu = poss[:,0] > Lx-dx
#        sbxl = poss[:,0] < dx
#        sbyu = poss[:,1] > Ly-dy
#        sbyl = poss[:,1] < dy
#        poss[:,0][sbxu] = Lx-dx
#        poss[:,0][sbxl] = dx
#        poss[:,1][sbyu] = dy#periodic
#        poss[:,1][sbyl] = Ly-dy#periodic'''
    return poss, Peff, zs, ANG"""

#@nb.njit
def position_boundaries(xs, ys):#quick-fix--ugly!!!
    sbxu = xs > Lx
    sbxl = xs < 0.
    sbyu = ys > Ly
    sbyl = ys < 0.
    xs[sbxu] = xs[sbxu]%Lx#Lx
    xs[sbxl] = Lx-np.abs(xs[sbxl])%Lx#0.
    ys[sbyu] = ys[sbyu]%Ly#periodic
    ys[sbyl] = Ly-np.abs(ys[sbyl])%Ly#periodic
    return xs, ys

def infect(vir, HID):
    virhex = np.histogram(HID, bins=nx*ny, range=(0,nx*ny), weights=vir)[0]
    vir = virhex[HID]
    return vir

def interact(ass, HID):
    asshex = py_histogram(HID, bins=nx*ny, range=(0,nx*ny), weights=ass)
    ni = py_histogram(HID, bins=nx*ny, range=(0,nx*ny), weights=np.ones_like(HID))
    return asshex[HID]/ni[HID]

#%% PLOTTING
def add_scalebar(ax, length, text, location=4, **kwargs):
    try:
        AnchoredSizeBar
    except:
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    scalebar = AnchoredSizeBar(ax.transData, length, text, location, **kwargs)
    ax.add_artist(scalebar)

#@jit
def hex_plot(hexcenters, hexcorners, z, save=False, svname='',alpha=1e-2, c='r',cmin=0, cmax=1):
    """ plot hexagonal grid by corners of hexagon. uses PolyCollection.
        hexcenters not yet used """
    coll = PolyCollection(hexcorners,
                          array=z,
                          cmap=ccmap,#mpl.cm.YlGnBu_r,
                          edgecolors='None')

    #fig, ax = plt.subplots(figsize=(16,4))
    coll2 = PolyCollection(hexcorners[motb],
                  linewidths = 1.5,
                  alpha=0.2,
                  edgecolors='darkblue',
                  facecolors='none')

    coll3 = PolyCollection(hexcorners[motb],
#                  linewidths = 10.,
                  array=z[motb],
                  cmap=ccmap,
                  edgecolors='None')

    fig, ax = plt.subplots()
    ax.add_collection(coll)
    ax.add_collection(coll2)
    ax.add_collection(coll3)

    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    plt.axis('scaled')
    #plot_scaler = (np.mean(ax.transData.transform((1e-6,1e-6)))/1e-6)

    cbar = fig.colorbar(coll,
                        ax=ax,
                        orientation='vertical',
#                        format = '%.2e',
                        shrink=0.5,
                        aspect=20,
                        fraction=.12,
                        pad=.02,
                        label = 'Carbon source ($g\; m^{-3}$)')

    cbar.ax.tick_params(labelsize=8)
    #cbar.ax.locator_params(nbins=3)

    try:
        lengths = cell_length(masses,1.1e6,0.5e-6,tot=True)
        '''plt.scatter(*zip(*hexcenters),
                     marker='o',
                     c=c,
                     s=lengths,
                    #s=((masses/310e3)/(mat.pi*(0.5e-6)**2))*1e7,
                     alpha=alpha)
        markers = [(2, 1, np.degrees(i)+90) for i in angles]
        [plt.scatter(posxs[i], posys[i], c=mpl.cm.rainbow(color_scaler*cell_color[i]), marker=markers[i], s=length[i]) for i in range(len(angles))]
        '''
        ecc = EllipseCollection(lengths,
                               np.zeros(len(angles))+1e-6,
                               np.degrees(angles),
                               units='xy',
                               offsets=hexcenters,
                               facecolors=c,
                               #facecolors=mpl.cm.gist_rainbow(color_scaler*cell_color),
                               edgecolors='None',#mpl.cm.Greys_r(255/np.max(zstack)*zstack),
                               alpha=alpha,
                               transOffset=ax.transData)
        ax.add_collection(ecc)
        #ax.autoscale_view('both')
    except:
        print 'hexcenters empty'
    plt.xlim(0.5*dx, nx*dx)
    plt.ylim(0.5*dy, ny*dy-0.5*dy)
#    ax.set_xticks([0,0.5*Lx*1e3,Lx*1e3])
#    ax.set_yticks([0,0.5*Lx*1e3,Lx*1e3])
    coll.set_clim(cmin, cmax)
    coll3.set_clim(cmin, cmax)
    
    bp = np.isnan(Spt[:,t-1])|(Spt[:,t-1]==0)
    p = Spt[~bp,t-1].astype(float)/Nt[t-1].astype(float)
    R = np.count_nonzero(~bp)
    plt.title(r'$t={0}h,\; N={1},\; R={2},\; H={3:.2f},\; \psi={4:.2f}m$'.format(t/60.,len(angles),R,-np.sum(p*np.log(p)),psi))
#    plt.xlabel('x (mm)')
#    plt.ylabel('y (mm)')
    plt.axis('off')
    add_scalebar(ax,1e-4,r'$100\mu m$',label_top=True,frameon=False, size_vertical=dx)

#    cbar.ax.set_xlabel('($g m^{-3}$)')
    left, bottom, width, height = [0.85, 0.15, 0.05, 0.05]
    ax2 = fig.add_axes([left, bottom, width, height])
#    ax2 = fig.add_subplot(111)
    mux1,ks1,mas1 = np.array([(mux0[cell_color==cc][0],ks0[cell_color==cc][0],np.mean(mas0[cell_color==cc])) for cc in np.unique(cell_color)]).T
    col1 = np.array([np.mean(col0[cell_color==cc],axis=0) for cc in np.unique(cell_color)])
#    ir = np.random.randint(0,len(col0),Ninit)
#    ax2.scatter(mux0[ir],ks0[ir],s=mas0[ir]*10,c=col0[ir],alpha=0.1)
    ax2.scatter(mux1,ks1,s=mas1,c=col1,alpha=0.5)
#    ax2.scatter(np.unique(mumax),np.unique(Ks),c=)
#    plt.imshow(mpl.mlab.griddata(mux0,ks0,mas0,*np.mgrid[0:1:0.1,0:1:0.1],interp='linear'),origin='lower')
    ax2.tick_params(axis=u'both', which=u'both',length=0)
    ax2.set_xticks([0,1])
    ax2.set_yticks([0,1])
    ax2.set_xlabel('Growthrate')
    ax2.set_ylabel('Affinity')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    try:
        left, bottom, width, height = [0.2, 0.05, 0.6, 0.05]
        ax3 = fig.add_axes([left, bottom, width, height])
        ax3.bar(range(np.sum(~bp)),Spt[~bp,t-1],color=col1,width=mas1*10)
        plt.axis('off')
    except:
        print 'no bars'
    if save == True:
        #plt.savefig('tmp\%d.png' % t, bbox_inches='tight')
        #plt.savefig('tmp/'+svname+'%d.svg' % t, bbox_inches='tight')
        #plt.savefig('tmp/'+svname+'%d.eps' % t, bbox_inches='tight')
#        plt.savefig('tmp/'+svname+'%d.png' % t, bbox_inches='tight',dpi=300)
        plt.savefig('./tmp/t-{:}_{:}CC_{:}m_rep{:}_{:}.png'.format(t,fU,psi,rep,svname), bbox_inches='tight',dpi=300)
#        plt.savefig('tmp/'+svname+'%d.pdf' % t, bbox_inches='tight')
    else:
        plt.show()

#%% CYTHONIZE
'''
CYTH = False
if CYTH:
    try:
        from subprocess import call
        call(["python","setup.py","build_ext","--inplace"],shell=True)
        from pyCfunc import *
    except ImportError:
        print 'ImportError: pyCfunc cython module broken'
        raw_input('Press Enter to continue...')
'''
#%% DOMAIN
por = 0.49
Lx = 1e-3#((5e-3*5e-3)/por)**0.5#512e-6*2
Ly = 1e-3#((5e-3*5e-3)/por)**0.5#512e-6/4

Ah = 100e-12#2*(0.08e-3/2.)**2*np.sqrt(3.)#100e-12#384e-12/128#128

def hexgrid(Lx,Ly,Ah):
    """ Lx = length in x [um]
    Ly = length in y [um]
    Ah = area of a regular hexagon [um**3]
    h = height of equilateral triangle; distance to neighbour hexagon
    s = side of regular hexagon
       .
     .h  .
     .-* . s
       .
    """
    s = (2**0.5 * Ah**0.5) / (3**(3./4.))
    h = 0.5 * 3**0.5 * s

    #distance between centers
    dx = h * 2. #also distance between neighbouring centers
    dy = s * 1.5

    #make columns and rows  even
    #nx = int(Lx/dx)
    #ny = int(Ly/dy)
    #
    if int(Lx/dx) % 2 == 0:
        nx = int(Lx/dx) + 1
    else:
        nx = int(Lx/dx)# + 1

    if int(Ly/dy) % 2 == 0:
        ny = int(Ly/dy) + 1
    else:
        ny = int(Ly/dy)# + 1


    x = np.linspace(dx*0.5, nx*dx-dx*0.5, nx)
    y = np.linspace(dy*0.5, ny*dy-dy*0.5, ny)

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    xv[:, 1::2] += h

    centers = zip(xv.ravel(), yv.ravel())

    corners = []
    neighbours = []

    for center in centers:
        #print center
        neighbours.append(hex_NN(center))
        points_c = []
        for i in range(7):
            points_c.append(hex_corner(center, s, i))
        corners.append(points_c)

    return dict(n=(nx,ny), d=(dx,dy), centers=np.asarray(centers), corners=np.asarray(corners), neighbours=np.asarray(neighbours))

grid = hexgrid(Lx,Ly,Ah)

nx,ny = grid['n']
dx,dy = grid['d']

#%% HYDRATION & DIFFUSIONCOEFFICIENT
D0 = np.empty((nx,ny))

try:
    wc = float(sys.argv[1])
    #psi = -float(sys.argv[1])
except:
    wc = 0.48
#    psi = -0.2#-0.295 #m

D = 6.7e-10 #m^2/s for Glucose
#D = 6.7e-11 #m^2/s

#http://hydrology.pnnl.gov/resources/unsath/drainage.asp
#wc_ceramic = van_genuchten(psi, 0.388, 0.387, 4.705*1e-4, 3.0)
#ceramic_rand = np.random.normal(1.0,0.0025,(nx,ny))#STAB
#D0[:] = MQ_Deff(D, wc_ceramic, 0.388)#*ceramic_rand
D0[:] = D
#wc = wc_ceramic*ceramic_rand
#%% ROUGHNESS and WATERFILM
#FIXME:check if ok
def bin2hexgrid(ls,nx,ny,mode='mean'):
    ls[:, 1::2] = 0.5*(np.roll(ls[:, 1::2],1,axis=0) + np.roll(ls[:, 1::2],-1,axis=0))
    return rebin(ls,(nx,ny),mode=mode)

def topography(rng,method='self-affine',path=None,sub=1):
    if method=='image':
        Lmin,Lmax = rng
        #path = r'C:\Users\bickels\Pictures\Microscopy\Keyence\holder-test\cortilt\20x-tiled-test-1pol-dry-cortilt.csv'
        #Vf = wc.ravel()*Ah*dx
        print 'Loading roughness:'
        height = np.genfromtxt(path, skip_header=52, delimiter=';',deletechars='"')
        print '.'
        heightresxy = np.genfromtxt(path, skip_header=40, skip_footer=len(height)+52-41, delimiter=';',deletechars='"')[1]*1e-9
        print '.'
        heightresz = np.genfromtxt(path, skip_header=41, skip_footer=len(height)+52-42, delimiter=';',deletechars='"')[1]*1e-9
        print '.'
        height *= heightresz
        height -= np.min(height)
        try:
            1/int(dx/heightresxy)
            1/int(dy/heightresxy)
            pxx = int(dx/heightresxy)*sub*nx
            pxy = int(dy/heightresxy)*sub*ny
            ls = height[550:550+pxx,550:550+pxy]
#            ls[:, 1::2] = 0.5*(np.roll(ls[:, 1::2],1,axis=0) + np.roll(ls[:, 1::2],-1,axis=0))
#            Ls = rebin(ls,(sub*nx,sub*ny))
            Ls = bin2hexgrid(ls,sub*nx,sub*ny)
            return Ls
        except:
            raise ValueError

    elif method=='self-affine':
        Lmin,Lmax = rng
        Ls = surface.genSurface((sub*nx,sub*ny),2.4).real
        Ls -= Ls.min()
        Ls /= (Ls.max()-Ls.min())
        Ls *= Lmax
        Ls += Lmin
        return Ls

    elif callable(method):
        Ls = method(*rng,size=(sub*nx,sub*ny))
        return Ls
    else:
        print 'not valid'

def aqueous_phase(Ls,psi,method='hexagonal cones',sub=1):
    if method == 'hexagonal cones':
        wft = hexagonal_pyramid(psi,Ls,sub=sub)
    elif method == 'constant':
        wft = np.full_like(Ls,5e-5)
    elif method == 'effective':
        wft = effective_film_thickness(psi, 0, Ls, 120, 998, 0.07275)
    elif method == 'curvature':
        try:
            wft = wft_iter(Ls, psi, dx=dx)-Ls+1e-8
        except:
            wft = Ls.max()-Ls+1e-8#+2.5e-5
            print 'saturation at current resolution'
    return wft

def abstract_soil(por):
    w = np.random.uniform(0, 1, size=(nx,ny))
    w /= w.sum()
    Vs = Lx*Ly*dx
    Vp = por*Vs
    Ls = w*Vp/Ah
    return w, Ls.ravel(), Vs

def abstract_water(w, wc, por, Vs, sigma=0.07275,rho=998e3):
    p = w < np.percentile(w,100*wc/por)
    clu,ncl = ndi.label(p,structure=np.ones((3,3)))
    rg = (ndi.sum(p*w,labels=clu,index=range(1,ncl+1))*Ah)**0.5
    psi = np.mean(2*sigma/(rg*rho*9.81))
    Vw = wc*Vs
    Vf = w*Vw
    wft = Vf/Ah
    return -psi, p, wft.ravel(), Vf.ravel()

'''    
subgrid = 1
ls = topography(np.array([2*Lx]),np.random.exponential,sub=subgrid)#+dx#.ravel()
wft = aqueous_phase(ls,psi,sub=subgrid)

Ls = bin2hexgrid(ls,nx,ny).ravel()
wft = bin2hexgrid(wft,nx,ny,mode='sum').ravel()
Vf = wft*Ah'''
w, Ls, Vs = abstract_soil(por)
psi, p, wft,Vf = abstract_water(w,wc,por,Vs)
motb = p.ravel()
#motb = wft >= 1e-6

#plt.imshow(wft.reshape(nx,ny),norm=mpl.colors.LogNorm())
#plt.colorbar()
#plt.imshow(motb.reshape(nx,ny),'Reds',alpha=0.5,vmin=0,vmax=1)
#plt.show()
#plt.hist(wft)
#plt.show()

print 'Done!'

#%% PERCOLATION
'''
npsi = 50
nsim = 100

Ncs = np.zeros((nsim,npsi))
ncl = np.zeros((nsim,npsi))

psis = -np.logspace(-2,1,npsi)

for sim in range(nsim):
    ll = topography(np.array([2*Lx]),np.random.exponential,sub=subgrid)
    for i,ps in enumerate(psis):
        ww = aqueous_phase(ll,ps,sub=subgrid)
        mo = ww >= 1e-6
        Ncs[sim,i] = np.count_nonzero(mo)/float(nx*ny*subgrid**2)
        _,nobj = ndi.label(mo,structure=np.ones((3,3)))
        ncl[sim,i] = nobj

fig,ax = plt.subplots()
plt.title(r'Aqeuous patches that allow cell motility ($\geq 1\mu m$ waterfilm); n={:}'.format(nsim))
plt.plot(-psis, Ncs.T,'k', alpha=2./nsim)
plt.plot(-psis, np.mean(Ncs,axis=0),'k')
plt.ylim(0,1)
plt.ylabel('Fraction of domain that allows motility')
plt.xlabel('-Matric potential (m)')
ax2 = plt.twinx(ax)
ax2.plot(-psis,ncl.T,'darkblue', alpha=2./nsim)
ax2.plot(-psis, np.mean(ncl,axis=0),'darkblue')
ax2.tick_params(axis='y', colors='darkblue')
ax2.set_ylabel('Number of aqueous patches', color='darkblue')
plt.ylim(0,ncl.max())
plt.xscale('log')
plt.show()'''

nwc = 2**5
nsim = 2**4

Ncs = np.zeros((nsim,nwc))
ncl = np.zeros((nsim,nwc))
psis = np.zeros((nsim,nwc))
wfts = np.zeros((nwc,nsim,nx*ny))

wcs = np.linspace(0.01,0.9*por,nwc)

for sim in range(nsim):
    ws, _, _ = abstract_soil(por)
    for i,ww in enumerate(wcs):
        ps, mo, wf,_ = abstract_water(ws,ww,por,Vs)
        wfts[i,sim,:] = wf
        psis[sim,i] = ps
        Ncs[sim,i] = np.count_nonzero(mo)/float(nx*ny)
        _,nobj = ndi.label(mo,structure=np.ones((3,3)))
        ncl[sim,i] = nobj

psim = -np.mean(psis,axis=0)
fig,ax = plt.subplots()
plt.title(r'Aqeuous patches that allow cell motility; n={:}'.format(nsim))
plt.plot(-psis.T, Ncs.T,'k', alpha=0.01)
plt.plot(psim, np.mean(Ncs,axis=0),'k')
plt.plot([-psi,-psi],[0,1],'--',label=r'$\theta={:}$'.format(wc))
plt.legend()
plt.ylim(0,1)
plt.ylabel('Fraction of domain that allows motility')
plt.xlabel('-Matric potential (m)')
ax2 = plt.twinx(ax)
ax2.plot(-psis.T,ncl.T,'darkblue', alpha=0.01)
ax2.plot(psim, np.mean(ncl,axis=0),'darkblue')
ax2.tick_params(axis='y', colors='darkblue')
ax2.set_ylabel('Number of aqueous patches', color='darkblue')
plt.ylim(0,ncl.max())
plt.xscale('log')
plt.show()

#%% MOTILITY
potv = lambda ps,wf: potential_velocity(0.5e-6, 8.94e-4*1e3, 1, 0.07275, ps, wf, 0.01)
vpot = potv(psi,wft)
vpot = vpot.reshape(nx,ny)

wftm = np.mean(wfts,axis=1)
#psis = np.logspace(-3,3)
#wfts = np.array([bin2hexgrid(aqueous_phase(ls,ps,sub=subgrid),nx,ny,mode='sum').ravel() for ps in -psis])
#wfts = np.array([aqueous_phase(Ls,ps,sub=subgrid) for ps in -psis])

vpotmaxs = [np.percentile(potv(ps,wf),95) for ps,wf in zip(psim,wftm)]
vpotmeans = [potv(ps,wf).mean() for ps,wf in zip(psim,wftm)]
vpotmins = [np.percentile(potv(ps,wf),5) for ps,wf in zip(psim,wftm)]

plt.plot(psim, vpotmeans,'b-')
plt.plot(psim, vpotmaxs,'b--')
plt.plot(psim, vpotmins,'b--')
plt.plot(abs(psi),np.mean(vpot),'r*',markersize=10.0)

plt.plot(psim, np.mean(wftm,axis=1),'g-')
plt.plot(psim, np.percentile(wfts,95,axis=(1,2)),'g--')
plt.plot(psim, np.percentile(wfts,5,axis=(1,2)),'g--')

plt.plot(abs(psi),np.mean(wft),'r*',markersize=10.0)

plt.plot([1e-6, 1e-6], 'k--', lw=1)

plt.annotate(r'$\bar{\delta}_{eff}$',
            xy=(abs(psi), np.mean(wft)),
            xytext=(1, 1e-4),
            arrowprops=dict(width=0.1, headwidth=3.0, facecolor='black', shrink=0.1))

plt.annotate(r'motility cutoff',
            xy=(1.5, 1e-6))

plt.xscale('log')
plt.yscale('log')
#plt.show()

#fig,ax = plt.subplots()
#plt.title('Water retention')
#plt.plot(psis, np.mean(wfts,axis=1),'g-')
#plt.plot(psis, np.percentile(wfts,95,axis=1),'g--')
#plt.plot(psis, np.percentile(wfts,5,axis=1),'g--')
#plt.plot([1e-3, 15],[1e-6, 1e-6], 'k--', lw=1)
#plt.plot(abs(psi),np.mean(wft),'r*',markersize=10.0)
#plt.annotate(r'motility cutoff', xy=(16, 1e-6))
#plt.ylabel('Water film thickness (m)')
#plt.xlabel('-Matric potential (m)')
#plt.loglog()
#ax2 = ax.twinx()
#ax2.plot(psis, vpotmeans,'b-')
#ax2.plot(psis, vpotmaxs,'b--')
#ax2.plot(psis, vpotmins,'b--')

#plt.plot(effective_film_thickness(-np.linspace(0,1,100), 4, np.mean(Ls), 120, 998, 0.07275), -np.linspace(0,1,100))
#%% EPS WATERBINDING
def exponential(x,k,x0,res):
    return x0*np.exp(k*x)+res

def logistic(x, r, K, x0, res):
    return (K*x0*np.exp(r*x))/(K+x0*(np.exp(r*x)-1))+res

def gaussian(x, a, b, res):
    return a*np.exp(-(x)/b)**2+res

def power(x,a,k,res):
    return a*x**k+res

def invexp(x, sat, k, shift, res):
    return sat/(1+np.exp(k*x-shift))+res

def satpow(x,sat,k):
    return sat/(1+sat*x**k)**(1/(1+k))**k

def eps_watercontent(psi):
    epsdat = [
    (0.0,323.),#from:https://books.google.ch/books?id=sx7aSd-f4EwC&pg=PA90&lpg=PA90&dq=water+holding+capacity+of+xanthan+gum&source=bl&ots=ueTMmqdnPH&sig=E9WD4clA0TpT6JnKeFOXpzXzF1Q&hl=en&sa=X&sqi=2&ved=0CCYQ6AEwAWoVChMI4vzw9ciAxwIVBm8UCh0SZg57#v=onepage&q=water%20holding%20capacity%20of%20xanthan%20gum&f=false
    (0.0,232.),#http://www.sciencedirect.com/science/article/pii/0023643895900217# assuming 1%residual water -> reduction of head
    (22.35673240574367, 19.742489270386265),#rosenzweig2012
    (50.42617282060831, 15.493562231759656),
    (124.36011879198018, 10.214592274678111),
    (476.8694922136621, 4.935622317596568),
    (1527.1351375678803, 3.326180257510728),
    (3141.7272499832816, 2.8755364806866943),
    (5264.856688893006, 1.5236051502145926),
    (8070.157970239357, 1.0729613733905587),
    (11322.871741282399, 0.8154506437768241),
    (0.3059932002, 15.28),#Chenu1996
    (10.19977334, 9.83),
    (25.49943335, 4.78),
    (30.59932002, 3.63),
    (50.9988667, 2.73),
    (101.9977334,2.46),
    (1.05206709053235, 67.94475971268702),#Chenu1993
    (10.5751269161937, 19.134357183486117),
    (52.1327578960476, 16.012011058639885),
    (253.051755813154, 4.630607034670689),
    (2112.19504126702, 0.9003002764659982),
    (10252.5683540134, 0.7283357805203963)
    ]
    psis, wcs = zip(*epsdat)
    plt.plot(psis[:2],wcs[:2],'o',
            psis[2:11],wcs[2:11],'h',
            psis[11:17], wcs[11:17],'+',
            psis[17:], wcs[17:],'^')

    peps = opt.curve_fit(satpow, psis, wcs, p0=[300,0.25])[0]

    plt.plot(np.logspace(-3,4,50), satpow(np.logspace(-3,4,50), peps[0], peps[1]))

    wc_eps = satpow(abs(psi), peps[0], peps[1])

    plt.plot(abs(psi),wc_eps,'r*',markersize=10.0)
    plt.annotate(r'$\theta_{eps}$',
                xy=(abs(psi),wc_eps),
                xytext=(1e-3,1),
                arrowprops=dict(width=0.1, headwidth=3.0, facecolor='black', shrink=0.1))

    plt.xscale('log')
    plt.yscale('log')

    return wc_eps

wc_eps = eps_watercontent(psi)

#avg Xanthan production per biomass on yeast extract from 1-4g/l:http://bipublication.com/files/IJABRv2i3201101.pdf
xanth = np.mean([3.0,2.6,3.6,2.9])*1e3
biom = np.mean([5.2,4.5,5.6,4.5])*1e3
Yeps = (xanth/biom)
eps_Vfm = Yeps*wc_eps*1e-6

#%% EPS DIFFUSIVITY
def eps_diffusivity(psi):
    Depsdat = [#Chenu1996
    (0., 3.983366447184775),
    (0.3059932002, 2.358146828459302),
    (10.19977334, 1.7801596032905858),
    (25.49943335, 1.4004449844830127),
    (50.9988667, 0.8426955222410135),
    (101.9977334, 0.7520155662468593)
    ]
    psis, Deps_glc = zip(*Depsdat)
    Deps_glc = np.array(Deps_glc)*1e-10
    plt.plot(psis,Deps_glc,'o')

    pDeps = opt.curve_fit(power, psis, Deps_glc, p0=[1, 0.1, 1])[0]
    plt.plot(np.linspace(0,100,100), power(np.linspace(0,100,100), pDeps[0], pDeps[1], pDeps[2]))

    Deps = power(abs(psi), pDeps[0], pDeps[1], pDeps[2])

    plt.plot(abs(psi),Deps,'r*',markersize=10.0)
    plt.annotate(r'$D_{eps}$',
                xy=(abs(psi),Deps),
                xytext=(40,3e-10),
                arrowprops=dict(width=0.1, headwidth=3.0, facecolor='black', shrink=0.1))
    return Deps

Deps = eps_diffusivity(psi)

#%% ADJACENCY and HETEROGENEOUS DIFFUSION
def connectivity(grid, wft, D0, xperiodic=False, yperiodic=True):
    xc, yc = zip(*grid['centers'])
    nn = [0]*(nx*ny)
    Adj = np.zeros((nx*ny, nx*ny), dtype='bool')

    for kk in range(nx*ny):
        dists = ((xc-xc[kk])**2 + (yc-yc[kk])**2)**0.5
        nnid, = np.nonzero(dists <= dx*1.5)
        nn[kk] = nnid
        Adj[kk, nnid] = 1
        Adj[nnid, kk] = 1
        Adj[kk, kk] = 0

    if xperiodic:
        Adj[0, nx-1] = 1
        Adj[nx*ny-1,nx*ny-nx] = 1
        Adj[nx-1, 0] = 1
        Adj[nx*ny-nx, nx*ny-1] = 1
        for aa in range(1,ny):
            Adj[aa*nx-1, aa*nx] = 1
            Adj[aa*nx, (aa+1)*nx-1] = 1
            Adj[aa*nx, aa*nx-1] = 1
            Adj[(aa+1)*nx-1, aa*nx] = 1

    if yperiodic:
        Adj[0, ny-1] = 1
        Adj[nx*ny-1,nx*ny-ny] = 1
        Adj[ny-1, 0] = 1
        Adj[nx*ny-ny, nx*ny-1] = 1
        for aa in range(1,nx):
            Adj[aa*ny-1, aa*ny] = 1
            Adj[aa*ny, (aa+1)*ny-1] = 1
            Adj[aa*ny, aa*ny-1] = 1
            Adj[(aa+1)*ny-1, aa*ny] = 1

    D0 = D0.ravel()
    Davg = np.zeros((nx*ny, nx*ny))

    for kk in range(nx*ny):
        nnid, = np.nonzero(Adj[kk,:])
        for nni in nnid:
            DD = np.array([D0[kk], D0[nni]])
            Davg[kk, nni] = hmean(DD)*np.minimum(wft[kk],wft[nni])/(3*wft[kk])
        Davg[kk, kk] = np.mean(Davg[kk, nnid])#np.append(Davg[kk, nnid], D0[kk])) #CHECK VALIDITY!!!

    Deg = np.identity(Adj.shape[0])*np.sum(Adj, axis=1)
    return Adj, Deg, Davg

Adj, Deg, Davg = connectivity(grid, wft, D0, xperiodic=True, yperiodic=True)

#%% GRID
grid.update({'adjacency': Adj,'degree': Deg})

#%% INIT TIME
dt = 60 #s
T = 60*192+1#24+1#/4#*2#*3#*7#*14 #real time: T*dT s
time = xrange(1, T)
pulses = np.random.poisson(4/(24.*60.),T)>0

#%% INIT BOUNDARY and CONCENTRATION
try:
    fU = float(sys.argv[2])
except:
    fU = 1.0

Uinit = 1e17*2.5e-14/pulses.sum()#/4#*Ah*dx#*1e12 #g/m^3 5.5mM * volHexagon Ah*dx m^3
Uinit *= fU

def boundary_conditions(bottom, top, left, right, constant=False, periodic=True, loc=False, POM=False, rnd=False):
    if constant:
        U = np.zeros((nx, ny))
        U[:1] = 1.
        source = np.array(U, dtype=bool)

        U = np.zeros((nx, ny))
        U[-1:] = 1.
        sink = np.array(U, dtype=bool)
    else:
        source = np.zeros((nx, ny), dtype=bool)
        sink = np.zeros((nx, ny), dtype=bool)

    if periodic:
        U = np.zeros((nx, ny))
        U[1:-1, 1:-1] = 1.
        periodic = ~np.array(U, dtype=bool)*~sink*~source
    else:
        periodic = ~np.zeros((nx, ny), dtype=bool)*~sink*~source

    U = np.zeros((nx, ny))
#    if POM:
#        U = POMs(Ls.max()-Ls,Uinit*Vf.sum(),1e-2,(nx,ny))/Vf.reshape((nx,ny))

    U[:, :1] = bottom     # (0, 0) to (nx, 0) 'bottom'
    U[:, -1:] = top    # (0, ny) to (nx, ny) 'top'
    U[:1] = left    # (0, 0) to (0, ny) 'left'
    U[-1:] = right    # (nx, 0) to (nx, ny) 'right'

    if loc:
        U[radial_mask(loc[0],loc[1],nx,ny,2)] = Uinit
        local = U > 0
        local *= ~source*~sink*~periodic
        interior = ~np.array(source+sink+periodic+local)
        U[local] = loc[2]

    elif POM:
        U = POMs(Ls.max()-Ls,Uinit*Vf.sum(),1e-6,(nx,ny))/Vf.reshape((nx,ny))
        local = U > 0
        local *= ~source*~sink*~periodic
        interior = ~np.array(source+sink+periodic+local)

    elif rnd:
        ip = np.random.choice(range(nx*ny),size=int((nx*ny)/4),p=Vf/Vf.sum(),replace=False)
        U = U.ravel()
        U[ip] = Uinit
        U = U.reshape(nx,ny)
        local = U > 0
        local *= ~source*~sink*~periodic
        interior = ~np.array(source+sink+periodic+local)

    else:
        local = np.full_like(U,False)
        interior = ~np.array(source+sink+periodic)

    U0 = U.copy()

    Ub = dict(source=source.ravel(),
              sink=sink.ravel(),
              periodic=periodic.ravel(),
              interior=interior.ravel(),
              local=local.ravel())

    ID = dict(source = np.nonzero(Ub['source']),
              sink = np.nonzero(Ub['sink']),
              periodic = np.nonzero(Ub['periodic']),
              loc = np.nonzero(Ub['local']),
              interior = np.nonzero(Ub['interior']))

    return U0, Ub, ID

U0, Ub, ID = boundary_conditions(0,0,0,0)#,loc=(nx/2,ny/2,Uinit))

#%% STEADY STATE
def steady_state(U0,Davg,Adj,Deg,ID):
    U00 = U0.ravel()
    Uss = np.empty_like(U00)
    rs = Davg/dx**2

    Lss = Adj*-rs+(Deg*rs)

    Lss[ID['source'],:] = 0.
    Lss[ID['source'],ID['source']] = 1.#Deg[sourceID,sourceID]*r[sourceID,sourceID]#1.
    #TODO: locID

    Lss[ID['loc'],:] = 0.
    Lss[ID['loc'],ID['loc']] = 1.

    Lss[ID['sink'],:] = 0.
    Lss[ID['sink'],ID['sink']] = 1.#Deg[sinkID,sinkID]*r[sinkID,sinkID]#1.

    Lss = spa.lil_matrix(Lss)
    Lss = Lss.tocsr()

    Uss = spla.spsolve(Lss, U00)
    return Uss

Uss = steady_state(U0,Davg,Adj,Deg,ID)

#%% FLUX COEFFICIENT MATRIX
def flux_coeff(Davg, Adj, Deg):
    r = (Davg*dt)/dx**2

    L = Adj*-r+(np.eye(nx*ny)+Deg*r)

    L[ID['source'],:] = 0.
    #L[:,sourceID] = 0.
    L[ID['source'],ID['source']] = 1.#Deg[sourceID,sourceID]*r[sourceID,sourceID]#1.
    #TODO: locID

    L[ID['loc'],:] = 0.
    L[ID['loc'],ID['loc']] = 1.

    #L= np.matrix(L)
    #
    L[ID['sink'],:] = 0.
    #L[:,sinkID] = 0.
    L[ID['sink'],ID['sink']] = 1.#Deg[sinkID,sinkID]*r[sinkID,sinkID]#1.

    L = spa.lil_matrix(L)
    L = L.tocsr()
    return L

L = flux_coeff(Davg, Adj, Deg)

#%% INIT NUTRIENTS
Ut = np.zeros((nx*ny, T))
Ut[:, 0] = np.full_like(Uss,Uinit)#Uss#

#%% INIT BACTERIA
Ninit = 2**12#len(grid['centers'][Ub['interior']])#[::5])#32
mux = 1.14/3600
#mua = np.random.normal(mux, mux*0.1, size=Ninit)
#mub = np.random.normal(mux*0.5, mux*0.1, size=Ninit)
mumax = np.random.uniform(1e-4*mux,mux,Ninit)#np.random.choice(np.append(mua,mub),Ninit)#np.linspace(0.01*mux,mux,Ninit)##np.array([mux]*Ninit)#np.array([mux]*Ninit)#np.random.uniform(mux*0.05,mux,Ninit)# #s from 1.14 h-1 E.coli
#mumax *= np.random.choice([-1.,1.],size=Ninit)
#mumax[mumax<0] = 0.01*mux

Kx = 10*68.#e-2
#Ka = np.random.normal(Kx, Kx*0.1, size=Ninit)
#Kb = np.random.normal(Kx*0.5, Kx*0.1, size=Ninit)
Ks = np.random.uniform(1e-2*Kx,Kx,Ninit)#np.random.choice(np.append(Ka,Kb),Ninit)#np.array([Kx]*Ninit)#np.array([Kx]*Ninit)#np.random.uniform(Kx*0.05,Kx,Ninit)#np.logspace(np.log(Kx)-2,np.log(Kx),Ninit)#np.linspace(Kx,Kx,Ninit)#*Ah*dx #g/m^3
#Ks[Ks<0] = 0.01*Kx

Ys = 0.5 #g_Bm/g_S

Mb0 = 9.5e-13 #g_Bm

def place_cells(Ninit, rnd=True, circ=False, inoc=False, rast=True, kill=False):
    masses = np.array([Mb0]*Ninit)
    dd = 2e-6 #check: cKDTree TypeError
    rinoc = (dd*Ninit)/(2*mat.pi)#3.18e-5

    if Ninit == 1:
        posxs = np.array([Lx/2])
        posys = np.array([Ly/2])

    elif not rnd and Ninit == 9:
        posxs = np.array([Lx/2,Lx/2-dd,Lx/2+dd]*3)
        posys = np.array([Ly/2]*3+[Ly/2-dd]*3+[Ly/2+dd]*3)

    elif not rnd and Ninit == 2:
        posxs = np.array([Lx/2-dd, Lx/2+dd])
        posys = np.array([Ly/2, Ly/2])

    elif not rnd and Ninit == 3:
        posxs = np.array([Lx/2-dd, Lx/2+dd, Lx/2])
        posys = np.array([Ly/2, Ly/2, Ly/2+dd])

    elif not rnd and circ:
        #rinoc = np.linspace(0.25*rinoc,rinoc,Ninit)
        posxs, posys, angles = cell_circle(rinoc, Ninit, (Lx+dx)/2, (Ly+dy)/2)

    elif inoc:
        posxs,posys = inoculate(rinoc, Ninit, (Lx/2,Ly/2), r0=rinoc*0.5)

    elif rast and rnd:
        idx = np.random.randint(0,np.count_nonzero(Ub['interior']),Ninit)
        posxs,posys = grid['centers'][Ub['interior']][idx,:].T

    elif rast:
        posxs,posys = grid['centers'][Ub['interior']].T#[::5].T

    else:
        posxs = np.random.uniform(dx,Lx-dx,Ninit)#np.random.standard_normal(Ninit)*Lx/8+Lx/2
        posys = np.random.uniform(dy,Ly-dy,Ninit)#np.random.standard_normal(Ninit)*Ly/8+Ly/2

    hexids = np.array(get_hexids(posxs, posys), dtype='int')
    #angles = np.array([rnd.random()*2*mat.pi]*Ninit)
    if Ninit == 9: #or rast:
        angles = np.ones(Ninit)*mat.pi/4.

    elif Ninit == 3:
        angles = np.deg2rad([210.1,330.1,90.1])

    elif circ:
        print 'circle mode'

    elif inoc:
        angles = np.random.uniform(0.,2*mat.pi,Ninit)
        print 'inoculation mode'

    else:
        angles = np.abs(np.random.standard_normal(Ninit)*2*mat.pi)

    #species = np.array(range(Ninit))%2
    #FIXME: posst&selections
    '''posst = np.empty((Ninit,2,T))'''
    poss = np.vstack([posxs,posys]).T
    '''posst[:,:,0] = poss'''

#    wdiag = int(2**0.5*4)
#    choices = [1,-1]+[0]*wdiag

    if kill:
        m0 = masses.copy()
        for t0 in range(1):
            masses = gain_masses(masses, Uss, hexids, mumax,Ks,Ys)[0]
            masses = maintains(masses,1e-8,1)
        masses[masses-m0<0] =  1e-16
        print Ninit, 'inoculated & ', len(masses[masses-m0<0]), 'died'

    return poss, angles, masses, hexids

poss, angles, masses, hexids = place_cells(Ninit)

posxs,posys = poss.T
Nt = np.zeros(T, dtype='int32')
Nt[0] = len(masses)

#%% RUN
#import sympy
#from sympy.abc import x, y
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import mlab
#from scipy import integrate
#from scipy import interpolate
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.ion()
plt.savefig('./tmp/hydration.pdf')
plt.show()

try:
    rep = int(sys.argv[3])
except:
    rep = 0

#plot_scaler = (np.mean(ax.transData.transform((1e-6,1e-6)))/1e-6)
#FIXME: do it, just do it!
color_scaler = 255/Ninit
cell_color = np.arange(Ninit)
np.random.shuffle(cell_color)

zstack = np.zeros(Ninit)
ass = np.random.uniform(-1.,1.,Ninit)#np.random.gamma(2,0.5,Ninit)#np.random.uniform(0.,2.,Ninit)
vir = np.random.choice([True]*10+[False]*90,Ninit)
mut = np.ones_like(vir,dtype='float')
virt = np.zeros(T)
virt[0] = np.sum(vir)

asst = np.zeros(T)
asst[0] = np.mean(ass)

Tt = np.empty(T)
Spt = np.empty((Ninit,T))
Spt[:,0] = np.histogram(cell_color,bins=Ninit,range=(0,Ninit-1))[0]

poss0=poss
import cProfile
pr = cProfile.Profile()
pr.enable()

verbose = False
plot = True
frame = 0
timed = 0
for t in time:
    tstart = tm.time()
    Ut[:, t] = spla.spsolve(L, Ut[:, t-1])
    
    if plot and (t%30==0 or t==1):
        if not verbose:
            print '{:.0f}% ct={:.3f}s N={:.1e} cells'.format(t*100./T, timed, Nt[t-1])
#        dis0 = ((poss-poss0)/dt)**2
#        dis0 = np.linalg.norm(dis0,axis=1)*masses*0.5
#        dis0[dis0==0] = np.nan
        mas0 = masses/(2*Mb0)#1.-(ass-ass.min())/(ass.max()-ass.min())#
        mux0 = mumax[cell_color]/mux
        aff = mumax[cell_color]/Ks[cell_color]
        ks0 = aff/aff.max()#1-Ks[cell_color]/Kx
        col0 = np.vstack([ks0,mux0,mas0]).T
        col0 = col0/col0.max(axis=0)
#        col0 = mpl.cm.Spectral(cell_color/float(cell_color.max()))
#        posass = ass < 0
#        col0[vir & ~posass] *= (1.0, 0.8, 0.8, 1.0)
#        col0[vir & posass] *= (0.8, 1.0, 0.8, 1.0)
#        col0[:,3] = mas0
        #poss = zip(posxs,posys)
        hex_plot(poss, grid['corners'], Ut[:,t],
                 save=True,
                 svname='{:}'.format(frame),
                 alpha=1.,#/(1+np.max(zstack)),#0.5,
                 c=col0,#mpl.cm.Spectral(cell_color/float(Ninit)),#cell_color,
                 cmax=np.clip(Ut[:,t].max(),0.1,Uinit),
                 cmin=np.clip(Ut[:,t].min(),1e-2,Uinit))
        plt.show()
        plt.close()
        frame += 1
    elif not verbose and (t%30==0 or t==1):
        print '{:.0f}% ct={:.3f}s N={:.1e} cells'.format(t*100./T, timed, Nt[t-1])
#    Ut[:,t] *= Vf#_eff
    '''
    vir = infect(vir, hexids)
    mut -= 1e-12*mut*dt
    vir[mut < 0.05] = False
    ass[mut < 0.05] += np.random.standard_normal()
    meanass = interact(ass, hexids)

    for z in np.unique(zstack):
        if z > 0: print z
        zb = zstack == z
        for sp in range(Ninit):
            sb = cell_color == sp
            masses[zb*sb], Ut[:,t], hexids[zb*sb] = \
                gain_masses(masses[zb*sb], Ut[:,t], hexids[zb*sb], mumax[sp], Ks[sp], Ys)

            #masses[zb*sb] = maintains(masses[zb*sb], vir[zb*sb]*meanass[zb*sb], mumax[sp])
            masses[zb*sb] = maintains(masses[zb*sb], 1e-8, 1)#mumax[sp])
    '''
    masses, Ut[:,t], hexids = \
                gain_masses(masses, Ut[:,t], hexids, mumax[cell_color], Ks[cell_color], Ys)
    masses = maintains(masses, 0.01, mumax[cell_color])
    #FIXME: is it the killing?
    selections(0.95*Mb0-np64floateps)

    masses, angles, posxs, posys, hexids = \
        divides(masses, 0.5, 0.025, 2.*Mb0, angles, posxs, posys, hexids)

#    Ut[:,t] /= Vf#_eff

    angles, posxs, posys, hexids = \
        chemotaxis(vpot, angles, posxs, posys, hexids, Ut[:,t])#, v0=30e-6)

    #Ut[:,t] /= Vf

#    Posm = -osmotic_potential((Ut[hexids,t]*Vf[hexids]-masses),8.31/180.16,273.15)#*1e6
    poss = np.vstack([posxs,posys]).T

#    Pcap = capilary_potential(0.5e-6, 72.8, psi, wft[hexids]*1e-2, wft[hexids]*1e-3, 0.05, masses, 1.1e6)#1e-3

    poss0 = poss.copy()

    for z in np.unique(zstack):
        zb = zstack==z
        try:
            poss[zb], Peff, zstack[zb],angles[zb] = nbody_shove(poss[zb],angles[zb],masses[zb], zstack[zb], hexids[zb])
            if z > 0: print ';',#z,
        except:
            print 'maybe next time',
    '''
    try:
        poss, Peff, zstack,angles = nbody_shove(poss,angles,masses, zstack, hexids)
    except:
        print 'maybe next time','''

    posxs, posys = np.array(zip(*poss))
    posxs, posys = position_boundaries(posxs,posys)

#    Ut[ID['source'], t] += (U0.ravel()[ID['source']]-Ut[ID['source'], t])#*ft
#    Ut[ID['sink'], t] += (U0.ravel()[ID['sink']]-Ut[ID['sink'], t])#*ft
#    Ut[ID['loc'], t] += (U0.ravel()[ID['loc']]-Ut[ID['loc'], t])
    # FIXME:
    #Ut[interiorID,t] += D*((Uss[interiorID]-Ut[interiorID,t])/Ls[interiorID])*Ah*dt
    if pulses[t]:
        Ut[:,t] = Uinit
        
    Nt[t] = len(masses)
    Spt[:,t] = np.histogram(cell_color,bins=Ninit,range=(0,Ninit-1))[0]
    virt[t] = np.sum(vir)
    asst[t] = np.mean(ass)
    '''posst[:,:,t] = poss[:Ninit]'''
    timed = tm.time()-tstart
    Tt[t] = timed

    if verbose:
        print 'timestep {0} computed in {1:.3f}s for {2:.1e} individuals'.format(t, timed, Nt[t])
    else:
        print '|',
    #

pr.disable()
pr.dump_stats('myprog.prof')

#%%
#Ut[ID['source'], t] += (U0.ravel()[ID['source']]-Ut[ID['source'], t])#*ft
#Ut[ID['sink'], t] += (U0.ravel()[ID['sink']]-Ut[ID['sink'], t])#*ft
#Ut[ID['loc'], t] += (U0.ravel()[ID['loc']]-Ut[ID['loc'], t])

Ut[:, t] = spla.spsolve(L, Ut[:, t-1])

print 'total: {0:.1f}min, average: {1:.3f}s'.format(np.sum(Tt)/60., np.mean(Tt))

np.savez('./out/{:}CC_{:}wc_rep{:}.npz'.format(fU,wc,rep),
         psi=psi,
         wc=wc,
         Uinit=Uinit,
         pulses=pulses,
         CC=fU,
         rep=rep,
         Nt=Nt,
         Tt=Tt,
         Spt=Spt,
         Ut=Ut,
         cell_color=cell_color,
         mumax=mumax,
         Ks=Ks,
         masses=masses,
         hexids=hexids,
         poss=poss,
         grid=grid,
         wft=wft,
         motb=motb,
         Vf=Vf,
         Ls=Ls)

#%%
if plot:
    print 'wft'
    hex_plot([], grid['corners'], wft, True, svname='wft', cmin=np.min(wft),cmax=np.max(wft))
    print 'motb'
    hex_plot([], grid['corners'], motb.astype(int), True, svname='motb', cmin=np.min(motb),cmax=np.max(motb))
    print 'Davg'
    hex_plot([], grid['corners'], np.diag(Davg), True, svname='Davg', cmin=np.min(np.diag(Davg)),cmax=np.max(np.diag(Davg)))

tt = np.arange(T)/float(60*24)
fig, ax1 = plt.subplots()
plt.xlabel(r'Time (d)')
ax2 = ax1.twinx()
ax1.plot(tt, Nt)
ax1.set_ylabel('Number of cells',color='b')
ax2.step(tt, 60*np.gradient(Nt)/np.array(Nt,dtype='float'),'g')
ax1.tick_params(axis='y', colors='b')
ax2.tick_params(axis='y', colors='g')
#plt.ylim(0,0.1)
plt.ylabel(r'$\frac{dN}{dt}\; (h^{-1}$)',color='g')
#plt.plot(time, Tt[1:])
plt.show()

plt.plot(tt, Nt,'k')
for ii,spt in enumerate(Spt):
    #plt.plot(spt,c=mpl.cm.gist_rainbow(mumax[ii]/mumax.max()),lw=3*(1-Ks[ii]/Ks.max())+1,label='{0:.2f}, {1:.2f}'.format(mumax[ii]*3600,Ks[ii]))
    plt.plot(tt,spt,c=mpl.cm.gist_rainbow(ii*color_scaler),lw=3*(1-Ks[ii]/Ks.max())+1,label='{0:.2f}, {1:.2f}'.format(mumax[ii]*3600,Ks[ii]),alpha=0.2)
plt.yscale('log')
if Ninit <= 64:
    plt.legend(loc=0, fontsize='xx-small', title='$\mu_{{max}}[h^{{-1}}],\ K_{{s}}[g\ m^{{-3}}]$', ncol=Ninit/8)
plt.xlabel('Time (d)')
plt.ylabel('Number of cells')
plt.show()

Rt = np.count_nonzero(Spt,axis=0)
p = Spt/Spt.sum(axis=0)
bp = np.isnan(p)|(p==0)
Ht = [-np.sum(pt[~b]*np.log(pt[~b]))for pt,b in zip(p.T,bp.T)]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(tt, Rt,'k')
ax1.set_ylabel('Richness')
ax2.plot(tt, Ht,'darkred')
ax2.tick_params(axis='y', colors='darkred')
plt.ylabel('Shannon diversity',color='darkred')
ax1.set_xlabel('Time (d)')
plt.show()

#
#Xx,Yy = np.array(zip(*grid['centers']))
#plt.contourf(Xx.reshape(nx,ny),Yy.reshape(nx,ny),Ut[:,t].reshape(nx,ny))
#plt.scatter(posxs, posys)
#plt.show()
'''
poss = zip(posxs,posys)#+np.random.normal(0.0,1e-6,(len(posxs),2))
if Ninit is 1:
    hex_plot(poss, grid['corners'], Ut[:,t], True)
else:
    hex_plot(poss, grid['corners'], Ut[:,t], True, c=cell_color,alpha=0.5)'''
#hex_plot(poss, grid['corners'], D0)
##%%
#psand
#psilt
#pclay
#plt.show()
#plt.scatter(*zip(*bac.poss))
#plt.show()


#for t in time:
##    hex_plot(grid['centers'], grid['corners'], Ut[:, t])
#    plt.contour(Ut[:,t].reshape(nx,ny))
#    plt.show()
'''
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
#ax.scatter(posxs,posys,masses)
surf = ax.plot_trisurf(grid['xcenters'], grid['ycenters'], np.histogram(hexids, bins=nx*ny, range=(0,nx*ny))[0],
                       cmap=mpl.cm.coolwarm,
                       norm=mpl.colors.LogNorm(vmin=1,vmax=1e6),
                       shade=True)
#ax.plot_trisurf(grid['xcenters'],grid['ycenters'], np.histogram(poss, bins=int((Lx/2e-6)*(Ly/2e-6)), range=(0,int((Lx/2e-6)*(Ly/2e-6))))[0])
#ax.zaxis.set_major_locator(mpl.ticker.LogLocator(10))
#ax.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()'''
#%%
'''
#Cytoplasm volume     Bacteria Escherichia coli     0.67     µm^3
#Average density     bacteria     1.1     g/cm^3

Fosm = osmotic_force(8.31, 273.15, 1.1e6*m_hex, 1.0, Ut[:,t], 0.5e-6)

plt.imshow(Fosm.reshape(nx,ny),interpolation='None')
plt.show()

Fosm_i = osmotic_force(8.31, 273.15, 1.1e6*masses, 1.0, Ut[hexids,t], 0.5e-6)
Fosm0 = np.histogram(hexids, bins=nx*ny, range=(0,nx*ny), weights=Fosm_i)[0]

Posm = osmotic_potential((masses-Ut[hexids,t]*Vf_eff[hexids]),8.31/180.16,273.15)

ncells = np.histogram(hexids, bins=nx*ny, range=(0,nx*ny))[0]
ang_avg = np.histogram(hexids, bins=nx*ny, range=(0,nx*ny), weights=angles)[0]/ncells
plt.quiver(grid['xcenters'], grid['ycenters'], np.cos(ang_avg), np.sin(ang_avg), scale=Fosm/1e-7)

#%%
lengths0 = cell_length(masses,1.1e6, 0.5e-6)

rxs = np.ptp(posxs)
rys = np.ptp(posys)

masshist, massposx, massposy = np.histogram2d(posxs,posys,weights=masses, bins=(rxs/2e-6,rys/2e-6))
tree = spat.cKDTree(masshist)

#%%osmotic potential summation

#xcell,ycell = (1e-5,1e-5)
res = 100
XX = np.linspace(0,Lx,res)
YY = np.linspace(0,Ly,res)
Xc,Yc = np.meshgrid(XX,YY)
r0 = 0.5e-6
PosmI = np.zeros_like(Xc)

for cell in range(len(poss)):
    xcell,ycell = poss[cell]
    rs = ((Xc-xcell)**2+(Yc-ycell)**2)**0.5
    rr = rs/r0
    PosmI += -Posm[cell]*np.exp(-rr)

plt.imshow(PosmI,interpolation='None')

plt.quiver(XX,YY,np.gradient(PosmI)[0],np.gradient(PosmI)[1],scale=1e-10)
'''
#Pcap = capilary_potential(0.5e-6, 72.8, psi, wft[hexids], wft[hexids]*0.1, 0.01, masses, 1.1e6)
#%% Shove_CTRL_PLOTS
Peff = Ut[:,t].reshape(nx,ny)
#Peff = np.histogram(hexids, bins=nx*ny, range=(0,nx*ny), weights=masses)[0].reshape(nx,ny)

'''
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
fig = plt.figure()
dat = Peff
xx = range(0, dat.shape[1])
yy = range(0, dat.shape[0])
X, Y = np.meshgrid(xx,yy)
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, dat, rstride=8, cstride=8, alpha=0.05)
cset = ax.contour(X, Y, dat, zdir='z', offset=np.min(dat), cmap=cm.coolwarm)
cset = ax.contour(X, Y, dat, zdir='x', offset=np.min(X), cmap=cm.coolwarm)
cset = ax.contour(X, Y, dat, zdir='y', offset=np.max(Y), cmap=cm.coolwarm)
plt.show()'''

N_hex = np.histogram(hexids, bins=nx*ny, range=(0,nx*ny))[0]
plt.hist(N_hex[N_hex != 0],normed=True, bins=20)
plt.title('cells per hexagon')
plt.show()
#%%
plt.contour(Peff)
UU, VV = np.gradient(Peff)
plt.quiver(UU,VV,(UU**2+VV**2)**0.5)
#plt.axis('equal')
plt.show()
'''
x = np.linspace(dx*0.5, nx*dx-dx*0.5, nx)
y = np.linspace(dy*0.5, ny*dy-dy*0.5, ny)
UU, VV = np.gradient(Peff)
plt.streamplot(x,y,UU.T,VV.T,norm=True,linewidth=1e12*(UU.T**2+VV.T**2)**0.5,density=2)
plt.xlim(0,Lx)
plt.ylim(0,Ly)
#plt.axis('equal')
plt.show()

import scipy.spatial as spat
import scipy.cluster as cl

pdis = spat.distance.pdist(poss)
cdis = spat.distance.pdist(cell_color[:,None])
cl.hierarchy.dendrogram(cl.hierarchy.linkage(pdis/cdis))
#plt.yscale('log')
#plt.ylim(1e-7,1e-5)
plt.xlabel('cell')
plt.ylabel('distance [m]')
plt.show()
'''
#%%
from sklearn.decomposition import PCA

dat = np.vstack([posxs,posys,masses])
pcs = PCA(n_components=2,whiten=True).fit(dat)

plt.scatter(*pcs.components_,c=cell_color,cmap='Spectral')
plt.xlabel('PC-1-{:.2f}%'.format(pcs.explained_variance_ratio_[0]*100))
plt.ylabel('PC-2-{:.2f}%'.format(pcs.explained_variance_ratio_[1]*100))
plt.show()

#%%trace
#plt.plot(posst[0,0,0],posst[0,1,0],'b*', markersize=20)
#plt.plot(posst[1,0,0],posst[1,1,0],'r*', markersize=20)
#plt.plot(posst[0,0,:],posst[0,1,:],'b')
#plt.plot(posst[1,0,:],posst[1,1,:],'r')
fig, ax = plt.subplots()
ax.set_prop_cycle(color=mpl.cm.gist_rainbow(color_scaler*cell_color[:Ninit]))
'''ax.plot(posst[:,0,0].T,posst[:,1,0].T,'k*')
ax.plot(posst[:,0,:].T,posst[:,1,:].T)'''
ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
plt.xlabel('x[m]')
plt.ylabel('y[m]')
#plt.xlim(0,Lx)
#plt.ylim(0,Ly)
plt.show()
'''
plt.imshow(wft.reshape(nx,ny).T,extent=(0.,Lx,0., Ly),origin='lower',interpolation='None',cmap=mpl.cm.terrain_r)
plt.plot(posst[0,0,0],posst[0,1,0],'b*', markersize=20)
plt.plot(posst[1,0,0],posst[1,1,0],'r*', markersize=20)
plt.plot(posst[2,0,0],posst[2,1,0],'g*', markersize=20)
plt.plot(posst[0,0,:],posst[0,1,:],'b')
plt.plot(posst[1,0,:],posst[1,1,:],'r')
plt.plot(posst[2,0,:],posst[2,1,:],'g')
plt.show()'''
#%%
plt.hist(zstack)
plt.show()

#%% depletion zones
#import scipy.ndimage as ndi
#def radial_mean(a, pos=(0,0),bins=20):
#    sx, sy = a.shape
#    ox,oy = pos
#    X,Y = np.ogrid[0:sx,0:sy]
#    r = np.hypot(X - ox, Y - oy)
#    rbin = (bins* r/r.max()).astype(np.int)
#    return ndi.mean(a, labels=rbin, index=np.arange(1, rbin.max() +1))
#
#
#Urad = radial_mean(Ut[:,t].reshape(nx,ny), pos=(poss[0,0]/dx,poss[0,1]/dy),bins=100)
#
#rsrad = gain_masses(np.repeat(masses,100),Urad*Vf[0],np.arange(100),mumax,Ks,Ys)[0]
#growth = rsrad >= maintains(masses, 0.1,mumax)
#
#rg = np.arange(100)*dx
#plt.plot(rg, Urad)
#plt.plot(rg, growth*Urad.max())
##plt.plot(substrate_utilization(monod(mumax,Ks,Urad),masses,Ys))
##plt.yscale('log')
#plt.show()
#%% SPECIES MAP
from scipy.interpolate import griddata

MAS0 = griddata(poss,masses,grid['centers']).reshape(nx,ny)
SPE0 = griddata(poss,cell_color,grid['centers']).reshape(nx,ny)
KS0 = griddata(poss,Ks[cell_color],grid['centers']).reshape(nx,ny)
MUX0 = griddata(poss,mumax[cell_color],grid['centers']).reshape(nx,ny)

plt.imshow(MAS0,'viridis',interpolation='None')
plt.show()

#%% interaction network
import networkx as ntx

G = ntx.MultiDiGraph()

#cells
ids = np.arange(len(masses))
bhs = [(hexids==hi) for hi in hexids]
idhs = [ids[bh].tolist() for bh in bhs]
G.add_nodes_from(ids)
ntx.set_node_attributes(G, name='cell', values=dict(zip(ids.tolist(),[True]*len(ids))))
ntx.set_node_attributes(G, name='color', values=dict(zip(ids.tolist(),cell_color.tolist())))
ntx.set_node_attributes(G, name='mumax', values=dict(zip(ids.tolist(),mumax[cell_color].tolist())))
ntx.set_node_attributes(G, name='Ks', values=dict(zip(ids.tolist(),Ks[cell_color].tolist())))
ntx.set_node_attributes(G, name='vir', values=dict(zip(ids.tolist(),vir[cell_color].tolist())))
ntx.set_node_attributes(G, name='ass', values=dict(zip(ids.tolist(),ass[cell_color].tolist())))
ntx.set_node_attributes(G, name='masses', values=dict(zip(ids.tolist(),masses[cell_color].tolist())))


ws = [{'weight':float(aa)} for aa in ass]
edges = []
for i,w in zip(ids,ws):
    for ih in idhs[i]:
        edges.append((i,ih,w))
G.add_edges_from(edges)

#hexagons
hids = np.unique(hexids)
G.add_nodes_from(['hex '+str(hi) for hi in hids])
ntx.set_node_attributes(G,name='Uss', values=dict([('hex '+str(hi),float(Uss[hi])) for hi in hids]))
ntx.set_node_attributes(G,name='cell', values=dict([('hex '+str(hi),False) for hi in hids]))

hedges = []
for hi in hids:
    for nh in np.nonzero(Adj[hi,:])[0]:
        if nh in hids:
            hedges.append(('hex '+str(hi), 'hex '+str(nh), {'flux coeff':float(Davg[hi,nh])}))
G.add_edges_from(hedges)

#hexagon-cell
#wms = [{'mean weight':float(aa)} for aa in meanass]
hcedges = zip(['hex '+str(hi) for hi in hexids],ids)
G.add_edges_from(hcedges)
meanass = interact(ass, hexids)
ntx.set_edge_attributes(G,name='mean weight',values=dict([[(e1,e2,k) ,float(wm)] for (e1,e2),k,wm in zip(hcedges,[0,]*len(meanass),meanass)]))
ntx.write_graphml(G,r'tmp\ass.graphml')

#%%
'''
ids = np.arange(len(masses))
bhs = [(hexids==hi) for hi in hexids]
idhs = [ids[bh].tolist() for bh in bhs]
ws = [{'weight':float(aa)} for aa in ass]
nodes = []
for i,w in zip(ids,ws):
    for ih in idhs[i]:
        #if not i==ih:
        nodes.append((i,ih,w))

hids = np.unique(hexids)
wms = [{'mean weight':float(aa)} for aa in np.unique(meanass)]
hnodes = []
for hi, wm in zip(hids,wms):
    for ih in ids[hexids==hi]:
        hnodes.append(('hex '+str(hi),ih,wm))
    for nh in np.nonzero(Adj[hi,:])[0]:
        if nh in hids:
            hnodes.append(('hex '+str(hi),'hex '+str(nh),{'flux coeff':float(Davg[hi,nh])}))

#nodes = zip(ids.tolist(), idhs, ws)
G = ntx.MultiDiGraph(nodes)
ntx.set_node_attributes(G,'color', dict(zip(ids.tolist(),cell_color.tolist())))
ntx.set_node_attributes(G,'mumax', dict(zip(ids.tolist(),mumax[cell_color].tolist())))
ntx.set_node_attributes(G,'Ks', dict(zip(ids.tolist(),Ks[cell_color].tolist())))
#ntx.draw(G)
G.remove_edges_from(G.selfloop_edges())

G.add_edges_from(hnodes)
ntx.set_node_attributes(G,'Uss', dict([('hex '+str(hi),float(Uss[hi])) for hi in hids]))

ntx.write_graphml(G,r'tmp\ass.graphml')
'''