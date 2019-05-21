#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
#import scipy as sp
#import scipy.linalg as la
import scipy.sparse as spa
import scipy.sparse.linalg as spla
import scipy.optimize as opt
import scipy.ndimage as ndi
import scipy.spatial as spat
#import scipy.stats as st
import matplotlib.pyplot as plt
import math as mat
from matplotlib.collections import PolyCollection#, RegularPolyCollection
import matplotlib as mpl
import time as tm
#from skimage.draw import polygon#line
import itertools as it
import multiprocessing as mp


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
#@jit
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
    return sigma/(psi*rho)

def effective_film_thickness(psi, beta, L, gamma, rho, sigma):
    return (film_adsorbed(psi, rho)*(beta*L+2*(L/np.cos(gamma*0.5)-radius_contact(psi, sigma, rho)/np.tan(gamma*0.5)))
    /L*(beta+np.tan(gamma*0.5)))

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)
'''
def effective_velocity(v, chi, Kd, S):
    S = S.reshape(nx,ny)
    gradSx, gradSy = np.gradient(S, dx, dy)
    vx = (2./3.)*v*np.tanh((chi*Kd)*gradSx/(2.*v*(Kd+S)**2))
    vy = (2./3.)*v*np.tanh((chi*Kd)*gradSy/(2.*v*(Kd+S)**2))
    return vx, vy'''

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

#@autojit
def unit_vector(v, direction2D=False):
    if direction2D:
        x = np.cos(v)
        y = np.sin(v)
        #v = np.array([x,y])
        return [x,y]/np.sqrt(x*x + y*y)#(x,y)/np.linalg.norm((x,y))
    return v/np.sqrt(v[0]*v[0] + v[1]*v[1])#v/np.sqrt(v.dot(v))#v/np.sqrt((v*v).sum(axis=1))#v/np.linalg.norm(v)


#@jit
def py_clip(x, l, u):
    return l if x < l else u if x > u else x

#@jit()
def closest_points_on_segments(ra,rb,a,b,la,lb, EPSILON = 1e-3):#np.sqrt and allnp. replaced with py
    hla = la/2.
    hlb = lb/2.
    r = rb-ra
    adotr = np.dot(a,r)
    bdotr = np.dot(b,r)
    adotb = np.dot(a,b)
    denom = 1. - adotb*adotb
    
    ta = 0.
    tb = 0.
    
    twopts = False
    if mat.sqrt(abs(denom)) > EPSILON: #test or replace back: np.sqrt(denom)
        ta0 = (adotr-bdotr*adotb)/denom
        tb0 = (adotr*adotb-bdotr)/denom
        ona = abs(ta0) < hla
        onb = abs(tb0) < hlb
        if not ona and not onb:
            ca = mat.copysign(hla, ta0)
            cb = mat.copysign(hlb, tb0)
            dddta = 2*(ca-adotb*cb-adotr)
            #dddtb = 2*(cb-adotb*ca+bdotr)
            if np.sign(dddta) == np.sign(ca):
                tb = cb;
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
        il = np.max(al, -hlb)#check this tunings...
        ir = np.min(ar, hlb)
        if il > ir:
            if al < -hlb:
                ta = hla
                tb = -hlb
            else:
                ta = -hla
                tb = hlb
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
    if twopts:
        pa2 = ra+ta2*a
        pb2 = rb+tb2*b
        return pa, pb, pa2, pb2
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
def overlaps(poss,ANG,lengths,ncells,r0):
    uv = np.zeros((ncells,2))
    cp = np.zeros((ncells,2))
    dv = unit_vector(ANG,direction2D=True)
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
    return uv, cp

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

'''
def Fm(R, nu, V0):
    return 6*mat.pi*R*nu*V0
    
def Fla(R, nu, V0, lamn, lamp):
    return (1-1/(lamn**2+lamp**2)**0.5)*Fm(R, nu, V0)

def Fc(R, sigma, gamma1, gamma2):
    return 2*mat.pi*sigma*R*(np.cos(gamma1)+np.cos(gamma2))

def potential_velocity(R, nu, V0, lamn, lamp, sigma, gamma1, gamma2):
    fm = Fm(R, nu, V0)
    return V0*(fm-Fla(R, nu, V0, lamn, lamp)-Fc(R, sigma, gamma1, gamma2))/fm
'''
#def svdsolve(a,b):
#    u,s,v = np.linalg.svd(a)
#    c = np.dot(u.T,b)
#    w = np.linalg.solve(np.diag(s),c)
#    x = np.dot(v.T,w)
#    return x

#%% VECTORIZED FUNCTIONS
'''
def gain_masses(M, S, HID, mumax, Ks, Ys):
    HID = HID.astype(int)
    ni = np.histogram(HID, bins=nx*ny, range=(0,nx*ny))[0]
#    mu = monod(mumax, Ks[HID.astype(int)]*ni[HID.astype(int)], S[HID.astype(int)]/ni[HID.astype(int)])
#    mu = monod(mumax, Ks[HID], S[HID]/ni[HID])
    a = cell_length(M, 1.1e6, 0.5e-6, tot=False)
    mu = monod(mumax, Ks*cell_volume(a, 0.5e-6), S[HID]/ni[HID])
#    mu = best_capsule(M, 1.1e6, 0.5e-6, S[HID]/ni[HID], Ks[HID], mumax, 1e4)
    rs = substrate_utilization(mu, M, Ys)
#        rs = substrate_utilization(mu[self.hexid], self.mass, self.Ys)
#    nb = np.array(ni,dtype='bool')
#    gb = np.array(Ut[nb, t] - ni[nb]*rs*dt) >= 0
#    if t%360 == 0:
#        print 'rs:', np.mean(rs)/np.mean(Vf), ' mu:',np.mean(mu)
    gmb = S[HID] - rs*dt >= 0.
    S[HID] -= rs*dt*gmb
    M *= 1+mu*dt*gmb
    return M, S, HID'''
'''
def maintains(M, rom, mumax):
    M *= 1-rom*mumax*dt
    return M'''
'''
def divides(M, m_asym, sd_asym, mtresh, ANG, xs, ys, HID):
    global cell_color, zstack
    b = np.array(M) > mtresh
    #fg = np.random.normal(m_asym, sd_asym, len(M[b]))
    fg=0.5
    newM= M[b]*(1.-fg)
    M[b] *= fg
    cl = cell_length(M[b],1.1e6,0.5e-6,tot=True)#+1e-7
    cln = cell_length(newM,1.1e6,0.5e-6,tot=True)#+1e-7
    M = np.append(M, newM)
    #minx,miny = ndi.minimum_position(Peff)
    #minx,miny = sg.argrelmin(Peff)
    #ANG[b] = np.arctan2(minx,miny)
    ub = unit_vector(ANG[b],direction2D=True)
    
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
    newxs = xs[b]+ub[0,:]*cln*0.5
    newys = ys[b]+ub[1,:]*cln*0.5
    xs[b] -= ub[0,:]*cl*0.5
    ys[b] -= ub[1,:]*cl*0.5
    xs = np.append(xs, newxs)
    ys = np.append(ys, newys)
    newHID = get_hexids(newxs,newys)
    HID = np.append(HID, newHID)
#    ANG = np.append(ANG, np.random.random(size=len(newHID))*2*mat.pi)
    cell_color = np.append(cell_color,cell_color[b])
    zstack = np.append(zstack,zstack[b])
    ANG = np.append(ANG, ANG[b])
    HID = HID.astype(int)
    return M, ANG, xs, ys, HID'''

'''    
def get_hexids(xs,ys):
    return [closest_node(pos, grid['centers']) for pos in zip(xs,ys)]'''
'''
def selections(dtresh):
    global masses, posxs, posys, hexids, angles, cell_color, zstack
    d = np.array(masses) < dtresh
    #print 'fatality', len(self.pop[d])
    masses = np.delete(masses, np.nonzero(d))
    posxs = np.delete(posxs, np.nonzero(d))
    posys = np.delete(posys, np.nonzero(d))
    hexids = np.delete(hexids, np.nonzero(d))
    angles = np.delete(angles, np.nonzero(d))
    cell_color = np.delete(cell_color, np.nonzero(d))
    zstack = np.delete(zstack, np.nonzero(d))
    #del self.pop[d]'''
'''
def chemotaxis(v, ANG, xs, ys, HID):
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
    swim = motb[HID]
#    tumble = gradt[HID]
    vx,vy = effective_velocity(v, 7.5e-4*1e-4, 22.5195, Ut[:,t])#0.125e-3
#    print 'vx: {0:.1e} vy: {1:.1e}'.format(np.mean(vx),np.mean(vy))
    vvx = vx.ravel()[HID]
    vvy = vy.ravel()[HID]
    xs[swim] += vvx[swim]*dt
    ys[swim] += vvy[swim]*dt
#    xs[swim] += np.cos(ANG[swim])*vvx[swim]*dt#*(10*0.5**(dt/10))
#    ys[swim] += np.sin(ANG[swim])*vvy[swim]*dt#*(10*0.5**(dt/10))
    sbxu = xs[swim] > Lx-dx
    sbxl = xs[swim] < dx
    sbyu = ys[swim] > Ly-dy
    sbyl = ys[swim] < dy
    xs[sbxu] = Lx-dx
    xs[sbxl] = dx
    ys[sbyu] = dy#periodic
    ys[sbyl] = Ly-dy#periodic
#    ys[sbyu] = Ly-dy
#    ys[sbyl] = dy
#    HID[swim] = [pool.apply(closest_node, args=(pos, grid['centers'])) for pos in zip(xs[swim],ys[swim])]
    HID[swim] = get_hexids(xs[swim], ys[swim])
#    ANG[tumble] = np.random.random(size=len(ANG[tumble]))*2*mat.pi
    #ANG[swim] = np.random.random(size=len(ANG[swim]))*2*mat.pi
    #ANG[swim] = np.arctan(vvy[swim]/vvx[swim])
    ANG[swim] = np.arctan2(vvx[swim],vvy[swim])
    return ANG, xs, ys, HID'''

#@jit
def nbody_shove(poss, ANG, M, zs, r0=0.5e-6, rho=1.1e6):
    ncells = len(poss)
    lengths = cell_length(M,rho,r0,tot=False)
    Peff = None
    #poss, Peff, zs, ANG = shove_physical(poss, ANG, M, zs, ncells, lengths, r0)
    poss, ANG = shove_numerical(poss, ANG, M, zs, ncells, lengths, r0)
    #poss *= np.random.normal(1, 1e-6, (ncells,2))
    return poss, Peff, zs, ANG

def shove_numerical(poss, ANG, M, zs, ncells, lengths, r0):
    dis,cps = overlaps(poss,ANG,lengths,ncells,r0)
    dis[np.isnan(dis)] = 0
    nSteps = np.count_nonzero(dis)/2+1
    print nSteps,np.min(spat.distance.pdist(poss)),np.mean(lengths)#np.sum(np.abs(dis))
    for step in xrange(1, nSteps + 1, 1):
        dis,cps = overlaps(poss,ANG,lengths,ncells,r0)
        dis[np.isnan(dis)] = 0
        #ANG += np.copysign(np.arctan2(np.linalg.norm(dis,axis=1),np.linalg.norm(cps,axis=1)),np.sum(dis,axis=1))
        ANG += np.copysign(np.arctan2(np.sqrt((dis*dis).sum(axis=1)),np.sqrt((cps*cps).sum(axis=1))),np.sum(dis,axis=1))
        dis,cps = overlaps(poss,ANG,lengths,ncells,r0)
        dis[np.isnan(dis)] = 0
        poss[:,0] += dis[:,0]
        poss[:,1] += dis[:,1]
    return poss, ANG

'''
def shove_physical(poss, ANG, M, zs, ncells, lengths,r0, nSteps=2, dt=60, div = 2.):
    dt = dt/nSteps
#    nSteps = 1
#    dt = 60
    dt2 = dt**2
    #r0 = cell_length(masses,1.1e6,0.5e-6)
    #res = 100
#    res = Lx/(r0/3.)
#    xres = yres = res
    xres = int(Lx/(r0/div))
    yres = int(Ly/(r0/div))
    xstep = Lx/xres
    ystep = Ly/yres
    
    u = unit_vector(ANG,direction2D=True)
    XX = np.linspace(0,Lx,xres)
    YY = np.linspace(0,Ly,yres)
    Xc,Yc = np.meshgrid(XX,YY)
    #poss += dis
    Peff = np.zeros_like(Xc)
    cellvs = np.zeros_like(poss)
    for step in xrange(1, nSteps + 1, 1):
        PosmI = np.zeros_like(Xc)
        PcapI = np.zeros_like(Xc)
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
        #labeled, nr_objects = ndi.label(Peff < np.min(Posm))
#        Peff = ndi.laplace(Peff)
        #Peff = ndi.minimum_filter(Peff,(3,3))
        #Peff = ndi.maximum_filter(Peff,(3,3))
        #Peff = ndi.gaussian_filter(Peff,1)
        Fx, Fy = np.gradient(Peff, xstep, ystep)
#        Fx = np.diff(Peff,n=1,axis=0)/xstep
#        Fy = np.diff(Peff,n=1,axis=1)/ystep
#        xid, yid =zip(*[(int(x/xres),int(y/yres)) for x,y in poss])
        xid, yid =zip(*[(int(x/xstep),int(y/ystep)) for x,y in poss])
        xid = np.clip(xid,0,Peff.shape[0]-1)
        yid = np.clip(yid,0,Peff.shape[1]-1)
        cellvs[:,0] += dt * -Fx[xid, yid]/M
        cellvs[:,1] += dt * -Fy[xid, yid]/M
#        cellvs[:,0] += dt * -Peff[xid, yid]/masses#*1e-12
#        cellvs[:,1] += dt * -Peff[xid, yid]/masses#*1e-12
#        poss[:,0] += cellvs[:,0] * dt + 0.5 * -Fx[xid, yid]/masses * dt2
#        poss[:,1] += cellvs[:,1] * dt + 0.5 * -Fy[xid, yid]/masses * dt2
        #zupb = (Peff[xid,yid] > -5.5e-12) & (Peff[xid,yid] != 0)
        #zs[zupb] += 1
        #print np.max(cellvs)*dt
        
        if abs(cellvs.any())*dt <= abs(lengths.any()/2.):
            poss[:,0] += cellvs[:,0] * dt + 0.5 * -Fx[xid, yid]/M * dt2
            poss[:,1] += cellvs[:,1] * dt + 0.5 * -Fy[xid, yid]/M * dt2
        else:
            #print dis[:,0]
            poss[:,0] += dis[:,0]#np.random.standard_normal(ncells)*r0
            poss[:,1] += dis[:,1]#np.random.standard_normal(ncells)*r0
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
#        poss[:,1][sbyl] = Ly-dy#periodic
    return poss, Peff, zs, ANG'''
#%% PLOTTING
'''
def hex_plot(hexcenters, hexcorners, z, save=False, alpha=1e-2, c='r'):
    """ plot hexagonal grid by corners of hexagon. uses PolyCollection. 
        hexcenters not yet used """
    coll = PolyCollection(hexcorners,
                          array=z, 
                          cmap=mpl.cm.YlGnBu_r, 
                          edgecolors='None')
                          #linewidths = 1/3.,
                          #edgecolors='white')
    fig, ax = plt.subplots()
    ax.add_collection(coll)
    #ax.autoscale_view('both')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    plt.axis('scaled')
    plot_scaler = (np.mean(ax.transData.transform((1e-6,1e-6)))/1e-6)
    fig.colorbar(coll, ax=ax)
    try: 
        plt.scatter(*zip(*hexcenters),
                     marker='o',
                     c=c,
                     s=cell_length(masses,1.1e6,0.5e-6)*plot_scaler,
                    #s=((masses/310e3)/(mat.pi*(0.5e-6)**2))*1e7,
                     alpha=alpha)
    except: 
        print 'hexcenters empty'
    plt.xlim(0.5*dx, nx*dx)
    plt.ylim(0.5*dy, ny*dy-0.5*dy)
    coll.set_clim(0.0, Uinit)
    if save == True:
        plt.savefig('gif\%d.png' % t, bbox_inches='tight')
    else:
        plt.show()
'''
