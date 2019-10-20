import numpy as np
from .rotation import *

Bohr = 0.52917721092 #Angstrom
Hartree = 27.2113860217 #eV

# Orbital angular momentum matrices for real spherical harmonics
# p-orbitals
Lpxr = np.zeros((3,3),dtype=complex)
Lpxr[2,1],Lpxr[1,2] =  1.j, -1.j
Lpyr = np.zeros((3,3),dtype=complex)
Lpyr[2,0],Lpyr[0,2] = -1.j,  1.j
Lpzr = np.zeros((3,3),dtype=complex)
Lpzr[1,0],Lpzr[0,1] =  1.j, -1.j
# d-orbitals
Ldxr = np.zeros((5,5),dtype=complex)
Ldxr[4,0],Ldxr[4,1],Ldxr[3,2],Ldxr[2,3],Ldxr[1,4],Ldxr[0,4] = \
    -1.j*np.sqrt(3.), -1.j,  1.j, -1.j,  1.j,  1.j*np.sqrt(3.)
Ldyr = np.zeros((5,5),dtype=complex)
Ldyr[3,0],Ldyr[3,1],Ldyr[4,2],Ldyr[2,4],Ldyr[1,3],Ldyr[0,3] = \
     1.j*np.sqrt(3.), -1.j, -1.j,  1.j,  1.j, -1.j*np.sqrt(3.)
Ldzr = np.zeros((5,5),dtype=complex)
Ldzr[2,1],Ldzr[4,3],Ldzr[3,4],Ldzr[1,2] = \
     2.j,  1.j, -1.j, -2.j
# f-orbitals
Lfxr = np.zeros((7,7),dtype=complex)
Lfxr[2,0],Lfxr[0,2] = -1.j*np.sqrt(6.),  1.j*np.sqrt(6.)
Lfxr[4,1],Lfxr[3,2],Lfxr[2,3],Lfxr[1,4] = \
  -1.j*np.sqrt(5./2.), 1.j*np.sqrt(5./2.),-1.j*np.sqrt(5./2.), 1.j*np.sqrt(5./2.)
Lfxr[6,3],Lfxr[5,4],Lfxr[4,5],Lfxr[3,6] = \
  -1.j*np.sqrt(3./2.), 1.j*np.sqrt(3./2.),-1.j*np.sqrt(3./2.), 1.j*np.sqrt(3./2.)
Lfyr = np.zeros((7,7),dtype=complex)
Lfyr[1,0],Lfyr[0,1] =  1.j*np.sqrt(6.), -1.j*np.sqrt(6.)
Lfyr[3,1],Lfyr[4,2],Lfyr[2,4],Lfyr[1,3] = \
   1.j*np.sqrt(5./2.), 1.j*np.sqrt(5./2.),-1.j*np.sqrt(5./2.),-1.j*np.sqrt(5./2.)
Lfyr[5,3],Lfyr[6,4],Lfyr[4,6],Lfyr[3,5] = \
   1.j*np.sqrt(3./2.), 1.j*np.sqrt(3./2.),-1.j*np.sqrt(3./2.),-1.j*np.sqrt(3./2.)
Lfzr = np.zeros((7,7),dtype=complex)
Lfzr[2,1],Lfzr[4,3],Lfzr[6,5],Lfzr[5,6],Lfzr[3,4],Lfzr[1,2] = \
     1.j,  2.j,  3.j, -3.j, -2.j, -1.j



def kpath(k1,k2,n):
    path = []
    for x in np.linspace(0.,1.,n):
        path.append(k1*(1.-x)+k2*x)
    return path

def mod2Pi(x, shift=0., two=False):
    half_mod = 1.
    if (two == False): half_mod = np.pi
    if (x > (half_mod+shift)): return mod2Pi(x-2.*half_mod)
    elif (x <= (-1.*half_mod+shift)): return mod2Pi(x+2.*half_mod)
    return x

def PlotBand(Band_list, kticks_label=None, yrange=None,
             eV=False, EF=None, highlight=None, save=False,
             fname=None, c1='b', c2='r', figsize=None):
    if (save):
        #del(sys.modules['matplotlib'])
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        len(Band_list[0][0])
    except:
        Band_list = [Band_list]
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot()
    kticks = []
    kstart = 0.
    i1,i2 = 1,-1
    if (highlight != None):
        i1,i2 = highlight[0],highlight[1]
    for Bands in Band_list:
        klist = Bands[0]+kstart
        kticks.append(klist[0])
        for i in range(len(Bands)-1):
            E = Bands[i+1].copy()
            if (eV == True): E *= Hartree
            col = c1
            if ((i1<=i)and(i<=i2)): col = c2
            plt.plot(klist,E,color=col)
        kstart = klist[-1]
    kticks.append(kstart)
    ax.set_xticks(kticks, minor=False)
    ax.xaxis.grid(True, which='major')
    if (kticks_label != None):
        ax.set_xticklabels(kticks_label)
    if (EF != None):
        if (eV == True): EF *= Hartree
        plt.plot([0.,kstart],[EF,EF],lw=0.25,color='gray',ls='--')
    plt.xlim(0,kstart)
    if (yrange != None):
        plt.ylim([yrange[0],yrange[1]])
    if(save):
        if(fname != None):
            plt.savefig(fname)
        else:
            plt.savefig('./pymx_band.png')
    else: plt.show()

def Band_Overlap(Band_lists, slist=None, kticks_label=None,
                 yrange=None, eV=False, EF=None, 
                 save=False, fname=None, defc='b', figsize=None):
    if (save):
        #del(sys.modules['matplotlib'])
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        len(Band_lists[0][0][0])
    except:
        Band_lists = [Band_lists]
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot()
    kticks = []
    for b,Band_list in enumerate(Band_lists):
        kstart = 0.
        for Bands in Band_list:
            klist = Bands[0]+kstart
            kticks.append(klist[0])
            for i in range(len(Bands)-1):
                E = Bands[i+1].copy()
                if (eV == True): E *= Hartree
                if (slist != None):
                    plt.plot(klist,E,slist[b])
                else:
                    plt.plot(klist,E,color=defc)
            kstart = klist[-1]
        kticks.append(kstart)
    cut = 0.0001*max(kticks)
    kticks0 = []
    for kt in kticks:
        tf = True
        for kt0 in kticks0:
            if (abs(kt-kt0) < cut):
                tf = False
        if (tf):
            kticks0.append(kt)
    kticks0.sort()
    ax.set_xticks(kticks0, minor=False)
    ax.xaxis.grid(True, which='major')
    if (kticks_label != None):
        ax.set_xticklabels(kticks_label)
    if (EF != None):
        if (eV == True): EF *= Hartree
        plt.plot([0.,kstart],[EF,EF],lw=0.25,color='gray',ls='--')
    plt.xlim(0,kstart)
    if (yrange != None):
        plt.ylim([yrange[0],yrange[1]])
    if(save):
        if(fname != None):
            plt.savefig(fname)
        else:
            plt.savefig('./pymx_band_overlap.png')
    else: plt.show()

def PlotWCC(WCC, save=False, fname=None, figsize=None):
    if (save):
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot()
    kaxis = WCC[0,:]
    for i in range(WCC.shape[0]-1):
        plt.plot(kaxis,WCC[i+1,:],'b.')
    ax.set_yticklabels([r'$-\pi$','0',r'$\pi$'])
    ax.set_yticks([-np.pi,0.,np.pi], minor=False)
    plt.ylim(-np.pi-0.05,np.pi+0.05)
    plt.xlim(0,kaxis[-1])
    if(save):
        if(fname != None):
            plt.savefig(fname)
        else:
            plt.savefig('./pymx_wccplot.png')
    fig2 = plt.figure(figsize=figsize)
    ax2 = plt.subplot()
    wccsum = np.sum(WCC[1:,:],axis=0)
    for i,x in enumerate(wccsum):
        wccsum[i] = mod2Pi(x)
    plt.plot(kaxis,wccsum,'r.')
    ax2.set_yticklabels([r'$-\pi$','0',r'$\pi$'])
    ax2.set_yticks([-np.pi,0.,np.pi], minor=False)
    plt.ylim(-np.pi-0.05,np.pi+0.05)
    plt.xlim(0,kaxis[-1])
    if(save):
        if(fname != None):
            plt.savefig('sum_'+fname)
        else:
            plt.savefig('./sum_pymx_wccplot.png')
    else: plt.show()

def matrix_visual(M, crange=1., cmap='RdBu_r'):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    class MidNorm(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)
        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))
    Nx = M.shape[0]
    Ny = M.shape[1]
    Mr = M.real
    Mi = M.imag
    MR = np.zeros((Nx+1,Ny+1),dtype=float) 
    MR[:Nx,:Ny] = Mr
    MI = np.zeros((Nx+1,Ny+1),dtype=float) 
    MI[:Nx,:Ny] = Mi
    x = np.linspace(0.,1.,Nx+1)
    y = np.linspace(1.,0.,Ny+1)
    X,Y = np.meshgrid(x,y)
    fig1 = plt.figure('real')
    plt.pcolormesh(X,Y,MR,norm=MidNorm(midpoint=0.),vmax=crange,vmin=-crange,cmap=cmap)
    plt.colorbar()
    plt.axes().set_aspect('equal')
    fig2 = plt.figure('imag')
    plt.pcolormesh(X,Y,MI,norm=MidNorm(midpoint=0.),vmax=crange,vmin=-crange,cmap=cmap)
    plt.colorbar()
    plt.axes().set_aspect('equal')
    plt.show()


