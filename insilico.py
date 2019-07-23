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
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, EllipseCollection
import time as tm
import numba as nb

#%% SET
plt.rcParams['figure.figsize'] = (10.0, 8.0)
#TODO: remove seed
#np.random.seed(123)

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
    return np.argmin(dist_2)

def radial_mask(a, b, nx, ny, r=1):
    xm, ym = np.ogrid[-a:nx-a, -b:ny-b]
    return xm*xm + ym*ym <= r*r

def hmean(a):
    return len(a) / np.sum(1.0/a)

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
    return vx, vy

def cell_volume(a, r):
    return mat.pi*r**2*(4./3.*r+a)

@nb.jit(nopython=True, nogil=True)
def cell_A_proj2d(a,r):
    return 2*r*a+mat.pi*r**2

@nb.jit(nopython=True, nogil=True)
def cell_length(m, rho, r, tot=True):
    V = m/rho
    return V/(mat.pi*r**2)-4*r/3+2*r*tot

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
    return v/mag

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
def closest_points_on_segments(ra,rb,a,b,la,lb, EPSILON = 1e-3):
    hla = la/2.
    hlb = lb/2.
    r = rb-ra
    adotr = d0(a,r)
    bdotr = d0(b,r)
    adotb = d0(a,b)
    denom = 1. - adotb*adotb

    ta = 0.
    tb = 0.

    twopts = False
    if mat.sqrt(abs(denom)) > EPSILON: 
        ta0 = (adotr-bdotr*adotb)/denom
        tb0 = (adotr*adotb-bdotr)/denom
        ona = abs(ta0) < hla
        onb = abs(tb0) < hlb
        if not ona and not onb:
            ca = mat.copysign(hla, ta0)
            cb = mat.copysign(hlb, tb0)
            dddta = 2*(ca-adotb*cb-adotr)
            if mat.copysign(1,dddta) == mat.copysign(1,ca):
                tb = cb
                ta = py_clip(tb*adotb+adotr, -hla, hla)
            else:
                ta = ca
                tb = py_clip(ta*adotb-bdotr, -hlb, hlb)
        elif ona and not onb:
            tb = mat.copysign(hlb, tb0)
            ta = py_clip(tb*adotb+adotr, -hla, hla)
        elif not ona and onb:
            ta = mat.copysign(hla, ta0)
            tb = py_clip(ta*adotb-bdotr, -hlb, hlb)
        else:
            ta = ta0
            tb = tb0
    else:
        xdotr = mat.copysign(min(adotr,bdotr), adotr)
        al = -xdotr-hla
        ar = -xdotr+hla
        il = max(al, -hlb)
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
    if twopts:
        pa2 = ra+ta2*a
        pb2 = rb+tb2*b
        pa[0] = (pa[0]+pa2[0])*0.5
        pa[1] = (pa[1]+pa2[1])*0.5
        pb[0] = (pb[0]+pb2[0])*0.5
        pb[1] = (pb[1]+pb2[1])*0.5
        return pa, pb
    else:
        return pa, pb

np64floateps = np.finfo(float).eps
@nb.jit(nopython=True, nogil=True)
def collision_looper(pairs,uv,cp,dv,poss,lengths,r0):
    for i in xrange(len(pairs)):
        cella = pairs[i,0]
        cellb = pairs[i,1]
        cpts = closest_points_on_segments(  poss[cella],poss[cellb],
                                            dv[:,cella],dv[:,cellb],
                                            lengths[cella],lengths[cellb])
        cpa,cpb = cpts
        cpx = cpb[0]-cpa[0]
        cpy = cpb[1]-cpa[1]
        dst = mat.sqrt(cpx*cpx + cpy*cpy)
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
    dv = dir_vector(ANG)
    dv = np.array(dv)
    tree = cKDTree(poss)
    pairs = tree.query_pairs(3e-6,2,0.5e-6, output_type='ndarray')
    uv,cp = collision_looper(pairs,uv,cp,dv,poss,lengths,r0)
    return uv, cp

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

def Fc(R, sigma, psi, d, dd, fc):
    db = d > 2*R
    aFc = (2*mat.pi*sigma/R-mat.pi*abs(psi))*(R**2-(d+dd-R)**2)*fc
    aFc[db] = 0
    aFc[aFc<0] = 0
    return aFc

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

#%% VECTORIZED FUNCTIONS
def gain_masses(M, S, HID, mumax, Ks, Ys, r0=0.5e-6, rho=1.1e6, mass=False):
    HID = HID.astype(int)
    ni = py_histogram(HID, bins=nx*ny, range=(0,nx*ny), weights=np.ones_like(HID))
    if mass:
        a = cell_length(M, rho, r0, tot=False)
        mu = monod(mumax, Ks*cell_volume(a, r0), S[HID])
    else:
        mu = monod(mumax, Ks, S[HID])
    rs = substrate_utilization(mu, M, Ys)
    S *= Vf
    fS = S[HID]/ni[HID]
    S0 = S[HID] - rs*dt
    gmb = S0 >= np64floateps
    S[HID] -= rs*dt*gmb
    S[HID] -= S[HID]*~gmb
    M[gmb] += rs[gmb]*dt
    M[~gmb] += fS[~gmb]*Ys
    S /= Vf
    return M, S, HID

def maintains(M, rom, mumax):
    M *= 1-rom*mumax*dt
    return M

def divides(M, m_asym, sd_asym, mtresh, ANG, xs, ys, HID, r0=0.5e-6, rho=1.1e6):
    global cell_color, zstack
    b = np.array(M) > mtresh
    fg = np.random.uniform(m_asym-sd_asym, m_asym+sd_asym, len(M[b]))
    newM= M[b]*(1.-fg)
    M[b] *= fg
    cl = cell_length(M[b], rho, r0, tot=True)
    cln = cell_length(newM, rho, r0, tot=True)
    M = np.append(M, newM)
    ub = dir_vector(ANG[b])
    newxs = xs[b]+ub[0]*cln*0.5
    newys = ys[b]+ub[1]*cln*0.5
    xs[b] -= ub[0]*cl*0.5
    ys[b] -= ub[1]*cl*0.5
    xs = np.append(xs, newxs)
    ys = np.append(ys, newys)
    newHID = get_hexids(newxs,newys)
    HID = np.append(HID, newHID)
    cell_color = np.append(cell_color,cell_color[b])
    zstack = np.append(zstack,zstack[b])
    ANG = np.append(ANG, ANG[b])
    HID = HID.astype(int)
    return M, ANG, xs, ys, HID

def get_hexids(xs,ys):
    return [closest_node(pos, grid['centers']) for pos in zip(xs,ys)]

def selections(dtresh):
    global masses, posxs, posys, hexids, angles, cell_color, zstack, vir, ass, Ut, mut
    d = np.array(masses) < dtresh
    nd = np.nonzero(d)
    masses = np.delete(masses, nd)
    posxs = np.delete(posxs, nd)
    posys = np.delete(posys, nd)
    hexids = np.delete(hexids, nd)
    angles = np.delete(angles, nd)
    cell_color = np.delete(cell_color, nd)
    zstack = np.delete(zstack, nd)

def individual_trajectory(x0, y0, vpot, vvx0, vvy0):
    px0 = x0/dx
    py0 = y0/dy
    px0 = py_clip(px0, 0, nx-1)
    py0 = py_clip(py0, 0, ny-1)
    pxt = px0+vvx0*dt/dx
    pyt = py0+vvy0*dt/dy
    pxt = py_clip(pxt, -nx+1, nx-1)
    pyt = py_clip(pyt, -ny+1, ny-1)
    try:
        l = int(np.hypot(pxt-px0, pyt-py0))
        px = np.linspace(px0, pxt, l)
        py = np.linspace(py0, pyt, l)
        vpi = vpot[px.astype(int), py.astype(int)]
        tix = np.cumsum(dx/(vvx0*vpi))
        tiy = np.cumsum(dy/(vvy0*vpi))
        return np.max(px[tix<=dt])*dx, np.max(py[tiy<=dt])*dy
    except:
        return vvx0*dt, vvy0*dt

def chemotaxis(vpot, ANG, xs, ys, HID, S, v0=30e-9):
    HID = HID.astype(int)
    ni = py_histogram(HID, bins=nx*ny, range=(0,nx*ny), weights=np.ones_like(HID))
    mb = ni*2e-12 < Ah
    swim = motb[HID]&mb[HID]
    vx,vy = effective_velocity(v0*vpot, 7.5e-4*1e-4, 22.5195, S)
    vvx = vx.ravel()[HID]*np.random.uniform(0.0, 2.0, size=HID.shape)
    vvy = vy.ravel()[HID]*np.random.uniform(0.0, 2.0, size=HID.shape)
    try:
        ddx, ddy = np.array([individual_trajectory(x0,y0,vpot,vvx0,vvy0) for x0,y0,vvx0,vvy0 in zip(xs[swim],ys[swim],vvx[swim],vvy[swim])]).T
        xs[swim] += ddx
        ys[swim] += ddy
    except:
        print '¦',#no swimmers',
    HID[swim] = get_hexids(xs[swim], ys[swim])
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
    poss, ANG = shove_numerical(poss, ANG, M, ncells, lengths, r0)
    zs = z_shove(hexids, r0, lengths, zs)
    return poss, Peff, zs, ANG

def position_boundaries(xs, ys):
    sbxu = xs > Lx
    sbxl = xs < 0.
    sbyu = ys > Ly
    sbyl = ys < 0.
    xs[sbxu] = xs[sbxu]%Lx#Lx
    xs[sbxl] = Lx-np.abs(xs[sbxl])%Lx#0.
    ys[sbyu] = ys[sbyu]%Ly#periodic
    ys[sbyl] = Ly-np.abs(ys[sbyl])%Ly#periodic
    return xs, ys

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
                          cmap=ccmap,
                          edgecolors='None')

    coll2 = PolyCollection(hexcorners[motb],
                  linewidths = 1.5,
                  alpha=0.2,
                  edgecolors='darkblue',
                  facecolors='none')

    coll3 = PolyCollection(hexcorners[motb],
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

    cbar = fig.colorbar(coll,
                        ax=ax,
                        orientation='vertical',
                        shrink=0.5,
                        aspect=20,
                        fraction=.12,
                        pad=.02,
                        label = 'Carbon source ($g\; m^{-3}$)')

    cbar.ax.tick_params(labelsize=8)

    try:
        lengths = cell_length(masses,1.1e6,0.5e-6,tot=True)

        ecc = EllipseCollection(lengths,
                               np.zeros(len(angles))+1e-6,
                               np.degrees(angles),
                               units='xy',
                               offsets=hexcenters,
                               facecolors=c,
                               edgecolors='None',
                               alpha=alpha,
                               transOffset=ax.transData)
        ax.add_collection(ecc)
    except:
        print 'hexcenters empty'
    plt.xlim(0.5*dx, nx*dx)
    plt.ylim(0.5*dy, ny*dy-0.5*dy)
    coll.set_clim(cmin, cmax)
    coll3.set_clim(cmin, cmax)
    
    bp = np.isnan(Spt[:,t-1])|(Spt[:,t-1]==0)
    p = Spt[~bp,t-1].astype(float)/Nt[t-1].astype(float)
    R = np.count_nonzero(~bp)
    plt.title(r'$t={0}h,\; N={1},\; R={2},\; H={3:.2f},\; \psi={4:.2f}m$'.format(t/60.,len(angles),R,-np.sum(p*np.log(p)),psi))

    plt.axis('off')
    add_scalebar(ax,1e-4,r'$100\mu m$',label_top=True,frameon=False, size_vertical=dx)

    left, bottom, width, height = [0.85, 0.15, 0.05, 0.05]
    ax2 = fig.add_axes([left, bottom, width, height])
    mux1,ks1,mas1 = np.array([(mux0[cell_color==cc][0],ks0[cell_color==cc][0],np.mean(mas0[cell_color==cc])) for cc in np.unique(cell_color)]).T
    col1 = np.array([np.mean(col0[cell_color==cc],axis=0) for cc in np.unique(cell_color)])

    ax2.scatter(mux1,ks1,s=mas1,c=col1,alpha=0.5)
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
        plt.savefig('./tmp/t-{:}_{:}CC_{:}m_rep{:}_{:}.png'.format(t,fU,psi,rep,svname), bbox_inches='tight',dpi=300)
    else:
        plt.show()

#%% DOMAIN
por = 0.49
Lx = 1e-3
Ly = 1e-3

Ah = 100e-12

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

    if int(Lx/dx) % 2 == 0:
        nx = int(Lx/dx) + 1
    else:
        nx = int(Lx/dx)

    if int(Ly/dy) % 2 == 0:
        ny = int(Ly/dy) + 1
    else:
        ny = int(Ly/dy)

    x = np.linspace(dx*0.5, nx*dx-dx*0.5, nx)
    y = np.linspace(dy*0.5, ny*dy-dy*0.5, ny)

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
    xv[:, 1::2] += h

    centers = zip(xv.ravel(), yv.ravel())

    corners = []
    neighbours = []

    for center in centers:
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
except:
    wc = 0.48

D = 6.7e-10 #m^2/s for Glucose

D0[:] = D

#%% ROUGHNESS and WATERFILM
def bin2hexgrid(ls,nx,ny,mode='mean'):
    ls[:, 1::2] = 0.5*(np.roll(ls[:, 1::2],1,axis=0) + np.roll(ls[:, 1::2],-1,axis=0))
    return rebin(ls,(nx,ny),mode=mode)

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


w, Ls, Vs = abstract_soil(por)
psi, p, wft,Vf = abstract_water(w,wc,por,Vs)
motb = p.ravel()

print 'Done!'

#%% MOTILITY
potv = lambda ps,wf: potential_velocity(0.5e-6, 8.94e-4*1e3, 1, 0.07275, ps, wf, 0.01)
vpot = potv(psi,wft)
vpot = vpot.reshape(nx,ny)

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
        Davg[kk, kk] = np.mean(Davg[kk, nnid])

    Deg = np.identity(Adj.shape[0])*np.sum(Adj, axis=1)
    return Adj, Deg, Davg

Adj, Deg, Davg = connectivity(grid, wft, D0, xperiodic=True, yperiodic=True)

#%% GRID
grid.update({'adjacency': Adj,'degree': Deg})

#%% INIT TIME
dt = 60 #s
T = 60*192+1 #time: T*dT (s)
time = xrange(1, T)
pulses = np.random.poisson(4/(24.*60.),T)>0

#%% INIT BOUNDARY and CONCENTRATION
try:
    fU = float(sys.argv[2])
except:
    fU = 1.0

Uinit = 1e17*2.5e-14/pulses.sum()
Uinit *= fU

def boundary_conditions(bottom, top, left, right, constant=False, periodic=True, loc=False, rnd=False):
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

U0, Ub, ID = boundary_conditions(0,0,0,0)

#%% STEADY STATE
def steady_state(U0,Davg,Adj,Deg,ID):
    U00 = U0.ravel()
    Uss = np.empty_like(U00)
    rs = Davg/dx**2

    Lss = Adj*-rs+(Deg*rs)

    Lss[ID['source'],:] = 0.
    Lss[ID['source'],ID['source']] = 1.

    Lss[ID['loc'],:] = 0.
    Lss[ID['loc'],ID['loc']] = 1.

    Lss[ID['sink'],:] = 0.
    Lss[ID['sink'],ID['sink']] = 1.

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
    L[ID['source'],ID['source']] = 1.

    L[ID['loc'],:] = 0.
    L[ID['loc'],ID['loc']] = 1.

    L[ID['sink'],:] = 0.
    L[ID['sink'],ID['sink']] = 1.

    L = spa.lil_matrix(L)
    L = L.tocsr()
    return L

L = flux_coeff(Davg, Adj, Deg)

#%% INIT NUTRIENTS
Ut = np.zeros((nx*ny, T))
Ut[:, 0] = np.full_like(Uss,Uinit)

#%% INIT BACTERIA
Ninit = 2**12
mux = 1.14/3600

mumax = np.random.uniform(1e-4*mux,mux,Ninit)

Kx = 10*68.
Ks = np.random.uniform(1e-2*Kx,Kx,Ninit)

Ys = 0.5

Mb0 = 9.5e-13

def place_cells(Ninit, rnd=True, circ=False, inoc=False, rast=True, kill=False):
    masses = np.array([Mb0]*Ninit)
    dd = 2e-6 
    rinoc = (dd*Ninit)/(2*mat.pi)

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
        posxs, posys, angles = cell_circle(rinoc, Ninit, (Lx+dx)/2, (Ly+dy)/2)

    elif inoc:
        posxs,posys = inoculate(rinoc, Ninit, (Lx/2,Ly/2), r0=rinoc*0.5)

    elif rast and rnd:
        idx = np.random.randint(0,np.count_nonzero(Ub['interior']),Ninit)
        posxs,posys = grid['centers'][Ub['interior']][idx,:].T

    elif rast:
        posxs,posys = grid['centers'][Ub['interior']].T

    else:
        posxs = np.random.uniform(dx,Lx-dx,Ninit)
        posys = np.random.uniform(dy,Ly-dy,Ninit)

    hexids = np.array(get_hexids(posxs, posys), dtype='int')
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

    poss = np.vstack([posxs,posys]).T

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
try:
    rep = int(sys.argv[3])
except:
    rep = 0

color_scaler = 255/Ninit
cell_color = np.arange(Ninit)
np.random.shuffle(cell_color)

zstack = np.zeros(Ninit)

Tt = np.empty(T)
Spt = np.empty((Ninit,T))
Spt[:,0] = np.histogram(cell_color,bins=Ninit,range=(0,Ninit-1))[0]

poss0=poss

verbose = False
plot = False
frame = 0
timed = 0
for t in time:
    tstart = tm.time()
    Ut[:, t] = spla.spsolve(L, Ut[:, t-1])
    
    if plot and (t%30==0 or t==1):
        if not verbose:
            print '{:.0f}% ct={:.3f}s N={:.1e} cells'.format(t*100./T, timed, Nt[t-1])

        mas0 = masses/(2*Mb0)
        mux0 = mumax[cell_color]/mux
        aff = mumax[cell_color]/Ks[cell_color]
        ks0 = aff/aff.max()
        col0 = np.vstack([ks0,mux0,mas0]).T
        col0 = col0/col0.max(axis=0)

        hex_plot(poss, grid['corners'], Ut[:,t],
                 save=True,
                 svname='{:}'.format(frame),
                 alpha=1.,
                 c=col0,
                 cmax=np.clip(Ut[:,t].max(),0.1,Uinit),
                 cmin=np.clip(Ut[:,t].min(),1e-2,Uinit))
        plt.show()
        plt.close()
        frame += 1
    elif not verbose and (t%30==0 or t==1):
        print '{:.0f}% ct={:.3f}s N={:.1e} cells'.format(t*100./T, timed, Nt[t-1])

    masses, Ut[:,t], hexids = \
                gain_masses(masses, Ut[:,t], hexids, mumax[cell_color], Ks[cell_color], Ys)
    masses = maintains(masses, 0.01, mumax[cell_color])
    
    selections(0.95*Mb0-np64floateps)

    masses, angles, posxs, posys, hexids = \
        divides(masses, 0.5, 0.025, 2.*Mb0, angles, posxs, posys, hexids)

    angles, posxs, posys, hexids = \
        chemotaxis(vpot, angles, posxs, posys, hexids, Ut[:,t])

    poss = np.vstack([posxs,posys]).T

    poss0 = poss.copy()

    for z in np.unique(zstack):
        zb = zstack==z
        try:
            poss[zb], Peff, zstack[zb],angles[zb] = nbody_shove(poss[zb],angles[zb],masses[zb], zstack[zb], hexids[zb])
            if z > 0: print ';',#z,
        except:
            print 'maybe next time',

    posxs, posys = np.array(zip(*poss))
    posxs, posys = position_boundaries(posxs,posys)

    if pulses[t]:
        Ut[:,t] = Uinit
        
    Nt[t] = len(masses)
    Spt[:,t] = np.histogram(cell_color,bins=Ninit,range=(0,Ninit-1))[0]
    timed = tm.time()-tstart
    Tt[t] = timed

    if verbose:
        print 'timestep {0} computed in {1:.3f}s for {2:.1e} individuals'.format(t, timed, Nt[t])
    else:
        print '|',


#%%
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
plt.ylabel(r'$\frac{dN}{dt}\; (h^{-1}$)',color='g')
plt.show()

plt.plot(tt, Nt,'k')
for ii,spt in enumerate(Spt):
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