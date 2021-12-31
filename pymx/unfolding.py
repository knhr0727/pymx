# initial version: 2021/Dec/28

from .pymx import * 
import scipy.sparse as spr

Otxt = '0 0 0'
                
def mat_print(X, l=0, delimiter=''):
    I,J = X.shape
    if (X.dtype==complex):
        for i in range(I):
            x = ' '
            for j in range(J):
                XX = X[i,j]
                x+= ("%."+str(l)+"f+"+"%."+str(l)+"fj"+delimiter)%(XX.real,XX.imag)
            print(x)
    else:
        for i in range(I):
            x = ' '
            for j in range(J):
                x+= ("%."+str(l)+"f"+delimiter)%X[i,j]
            print(x)

def load_band(fname):
    f = np.load(fname)
    ln = len(f)
    if (ln==4):
        klist  = f['klist' ]
        Elists = f['Elists']
        Wlists = f['Wlists']
        EF     = f['EF'    ]
        f.close()
        return [[klist,Elists,Wlists],EF]
    elif (ln==6):
        klist   = f['klist' ]
        Elists1 = f['Elists1']
        Wlists1 = f['Wlists1']
        Elists2 = f['Elists2']
        Wlists2 = f['Wlists2']
        EF     = f['EF'    ]
        f.close()
        return [[klist,Elists1,Wlists1,Elists2,Wlists2],EF]
    elif (ln==7):
        klist  = f['klist' ]
        Elists = f['Elists']
        Wlists = f['Wlists']
        Sxlists = f['Sxlists']
        Sylists = f['Sylists']
        Szlists = f['Szlists']
        EF     = f['EF'    ]
        f.close()
        return [[klist,Elists,Wlists,Sxlists,Sylists,Szlists],EF]
    else:
        f.close()
        raise Exception("error : invalid input of save_band.")

def print_openmxstyle(ufbands, cut1, cut2, EF=0., fname='unfolded_bands_omx',\
                      labels=None):
    if len(ufbands[0])==3:
        SP = 0 #spinless or spin non-collinear
    elif len(ufbands[0])==5:
        SP = 1 #spin collinear
    elif len(ufbands[0])==6:
        SP = 4 #spin texture
    else:
        raise Exception("error : invalid input format in print_openmxstyle.\
 it must be a list of outputs of Unfolding_band or Unfolding_spintexture_band")
    Nb = len(ufbands)
    ticks = [0.]
    k0 = 0.
    if (SP==0):
        f = open(fname+'.dat','w')
    elif (SP==1):
        f1 = open(fname+'.up.dat','w')
        f2 = open(fname+'.dn.dat','w')
    elif (SP==4):
        f1 = open(fname+'.dat','w')
        f2 = open(fname+'.sx.dat','w')
        f3 = open(fname+'.sy.dat','w')
        f4 = open(fname+'.sz.dat','w')
    for j in range(Nb):
        k = ufbands[j][0]
        if (j==0):
            i0 = 0
        else:
            i0 = 1
        for i in range(i0,len(k)):
            kout = k[i] + k0
            if (SP==0):
                E = ufbands[j][1][i,:]
                E = (E-EF)*Hartree
                W = ufbands[j][2][i,:]
                for e,w in zip(E,W):
                    if (cut1 <= e <= cut2):
                        outstr = "%.6f %.6f %.7f \n"%(kout,e,w)
                        f.write(outstr)
            elif (SP==1):
                E1 = ufbands[j][1][i,:]
                E1 = (E1-EF)*Hartree
                W1 = ufbands[j][2][i,:]
                E2 = ufbands[j][3][i,:]
                E2 = (E2-EF)*Hartree
                W2 = ufbands[j][4][i,:]
                for e1,w1 in zip(E1,W1):
                    if (cut1 <= e1 <= cut2):
                        outstr = "%.6f %.6f %.7f \n"%(kout,e1,w1)
                        f1.write(outstr)
                for e2,w2 in zip(E2,W2):
                    if (cut1 <= e2 <= cut2):
                        outstr = "%.6f %.6f %.7f \n"%(kout,e2,w2)
                        f2.write(outstr)
            elif (SP==4):
                E = ufbands[j][1][i,:]
                E = (E-EF)*Hartree
                W = ufbands[j][2][i,:]
                Sx = ufbands[j][3][i,:]
                Sy = ufbands[j][4][i,:]
                Sz = ufbands[j][5][i,:]
                for e,w,sx,sy,sz in zip(E,W,Sx,Sy,Sz):
                    if (cut1 <= e <= cut2):
                        outstr = "%.6f %.6f %.7f \n"%(kout,e,w)
                        f1.write(outstr)
                        outstr = "%.6f %.6f %.7f \n"%(kout,e,sx)
                        f2.write(outstr)
                        outstr = "%.6f %.6f %.7f \n"%(kout,e,sy)
                        f3.write(outstr)
                        outstr = "%.6f %.6f %.7f \n"%(kout,e,sz)
                        f4.write(outstr)
        k0 = kout
        ticks.append(k0)
    if (SP==0):
        f.close()
    elif (SP==1):
        f1.close()
        f2.close()
    elif (SP==4):
        f1.close()
        f2.close()
        f3.close()
        f4.close()
    f = open(fname+".kpt",'w')
    if (labels is None):
        for k in ticks:
            f.write(" %.8f \n"%k)
    else:
        for k,l in zip(ticks,labels):
            f.write((" %.8f "%k) +l+" \n")
    f.close()

class unfolding_pymx():
    '''unfolding package based on pymx'''

    def __init__(self, pm, uf_cell, uf_map, origin=Otxt, check=True):
        #initailizing pymx
        self.pm = pm
        if ((pm.H_R1 is None)and(pm.MS_H_R1 is None)):
            pm.read_file()
            pm.del_rawdata()

        # Unfolding.ReferenceVectors
        if type(uf_cell) is not str:
            raise Exception("Error: uf_cell should be str") 
        uc_str = uf_cell.splitlines()
        uc_list = []
        for line in uc_str:
            spl = line.split()
            if (len(spl)==3):
                uc_list.append([float(spl[0]),float(spl[1]),float(spl[2])])
        if (len(uc_list)!=3):
            raise Exception("Error: wrong uf_cell format") 
        self.ufa1 = np.array([uc_list[0][0],uc_list[0][1],uc_list[0][2]])/Bohr
        self.ufa2 = np.array([uc_list[1][0],uc_list[1][1],uc_list[1][2]])/Bohr
        self.ufa3 = np.array([uc_list[2][0],uc_list[2][1],uc_list[2][2]])/Bohr

        # origin setting
        if type(origin) is not str:
            raise Exception("Error: origin should be str") 
        spl = origin.split()
        if (len(spl)!=3):
            raise Exception("Error: wrong origin format") 
        O = self.ufa1*float(spl[0]) + self.ufa2*float(spl[1]) + \
            self.ufa3*float(spl[2])
        self.origin = O/Bohr

        # Unfolding.Map
        if type(uf_map) is not str:
            raise Exception("Error: uf_map should be str") 
        um_str = uf_map.splitlines()
        um_list = []
        for line in um_str:
            spl = line.split()
            if (len(spl)==2):
                um_list.append([int(spl[0]),int(spl[1])])
        self.uf_map = np.array(um_list)
        self.Ntot = self.uf_map.shape[0]
        self.Ncon = np.max(self.uf_map[:,1])  # number of atoms in conceptual cell

        # lattice information of the systems
        self.ufV = abs(np.dot(self.ufa1,np.cross(self.ufa2,self.ufa3)))
        self.a1 = pm.a1
        self.a2 = pm.a2
        self.a3 = pm.a3 
        self.V = abs(np.dot(self.a1,np.cross(self.a2,self.a3)))
        self.multi = np.rint(self.V/self.ufV)
        self.Nm = int(self.multi)
        if check:
            print("multiplicity of the supercell: rint(%.3f) = %d"\
                 %(self.multi,self.Nm))
        self.b1 = pm.b1
        self.b2 = pm.b2
        self.b3 = pm.b3 
        self.ufb1 = 2.*np.pi*np.cross(self.ufa2,self.ufa3)/self.ufV
        self.ufb2 = 2.*np.pi*np.cross(self.ufa3,self.ufa1)/self.ufV
        self.ufb3 = 2.*np.pi*np.cross(self.ufa1,self.ufa2)/self.ufV

        def findcell(r1,r2,maxcell):
            dist = []
            ints = []
            r0 = r2-r1 #r1:atom in conceptual cell/ r2: atom in supercell
            for i in range(-maxcell,maxcell+1):
                for j in range(-maxcell,maxcell+1):
                    for k in range(-maxcell,maxcell+1):
                        r = i*self.ufa1 + j*self.ufa2 + k*self.ufa3
                        d = np.linalg.norm(r-r0)
                        dist.append(d)
                        ints.append([i,j,k])
            imin = np.argmin(dist)
            return ints[imin]

        atom_con = [] #atom numbers corresponding to the conceptual cell
        uf_map2 = []
        for i in range(self.Ncon):
            ncon = i+1
            tfv = self.uf_map[:,1]==ncon
            atom_ncon = self.uf_map[tfv,0]
            dist = []
            for j in atom_ncon:
                dist.append(np.linalg.norm(pm.tau[j]-self.origin))
            imin = np.argmin(dist)
            jncon = atom_ncon[imin]
            atom_con.append(atom_ncon[imin])
            for j in atom_ncon:
                uf_map2.append([j,jncon])
        self.uf_map2 = np.array(uf_map2)
        if check:
            print("unfolding mapping in the supercell: ")
            print(self.uf_map2)

        refL = np.linalg.norm(self.a1)+np.linalg.norm(self.a2)+np.linalg.norm(self.a3)
        refl = min(np.linalg.norm(self.ufa1),np.linalg.norm(self.ufa2),np.linalg.norm(self.ufa3))
        Nl = int(refL*1.01/refl)
        maxfind = min(self.Nm+1,Nl)

        if check:
            print("check M, m(M), r'(M): ")
        self.rlist = [] # r'(M)
        for i in range(self.Ntot):
            M,m = self.uf_map2[i,:]
            rm = pm.tau[m]
            rM = pm.tau[M]
            ijk = findcell(rm,rM,maxfind)
            self.rlist.append(np.array(ijk))
            if check:
                print(M,m,ijk)

        self.rset = []
        for v in self.rlist:
            tf = True
            for vv in self.rset:
                if np.all(vv==v):
                    tf = False
            if tf:
                self.rset.append(v)
        if check:
            print("r set: ")
            print(self.rset)
        self.Nr = len(self.rset)
        if check:
            print("rset size: ")
            print(self.Nr)
        
        #if check:
        #    print("correction can be applied here")

        self.M2 = np.zeros((self.Nr,self.Ntot),dtype=int)
        for i in range(self.Nr):
            r = self.rset[i]
            for j in range(self.Ntot):
                M,m = self.uf_map2[j,:]
                for k,rr in enumerate(self.rlist):
                    if np.all(r==rr) and (m==self.uf_map2[k,1]):
                        self.M2[i,M-1] = self.uf_map2[k,0] 
        if check:
            print("M2: ")
            print(self.M2)
                

        self.ms = pm.mat_size
        self.delta = []
        for i in range(self.Nr):
            rr = []
            cc = []
            for j in range(self.Ntot):
                ca1,ca2 = pm.At_range[j+1] # column atom index
                # for a given column (M), find non-zero row
                ra1,ra2 = pm.At_range[self.M2[i,j]] # row atom index
                if ((ca2-ca1!=0)and(ra2-ra1!=0)):
                    for ic in range(ca1,ca2):
                        cc.append(ic)
                    for ir in range(ra1,ra2):
                        rr.append(ir)
            one = np.ones(len(cc), dtype=float)
            self.delta.append(spr.csr_matrix((one, (rr, cc)),\
                         dtype=float, shape=(self.ms,self.ms)))

    def eikr0(self, k):
        out = []
        for ijl in self.rset:
            i,j,l = ijl
            r0 = i*self.ufa1 + j*self.ufa2 + l*self.ufa3
            out.append(np.exp(1.j*np.dot(k,r0)))
        return np.array(out) 

    def eikrM(self, k):
        mat = np.identity(self.ms,dtype=complex)
        for M in range(self.Ntot):
            i,j,l = self.rlist[M] 
            rM = i*self.ufa1 + j*self.ufa2 + l*self.ufa3
            i1,i2 = self.pm.At_range[M+1] 
            mat[i1:i2,i1:i2] *= np.exp(-1.j*np.dot(k,rM))
        return mat

    def Unfold_mat(self, Sk,k):
        eikr0vec = self.eikr0(k)/self.multi
        eikr0delta = np.zeros((self.ms,self.ms), dtype=complex)
        for phase, spmat in zip(eikr0vec,self.delta): 
            eikr0delta += phase*spmat.todense()
        return np.linalg.multi_dot([Sk,eikr0delta,self.eikrM(k)])

    def Unfolding(self, k, eV=False, shift=False):
        SP = self.pm.SpinP_switch
        if (SP==0)or(SP==3): 
            h = self.pm.Hk_kvec(k)
            s = self.pm.Sk_kvec(k, small=True)
            ufmat = self.Unfold_mat(s,k)
            if (SP==3):
                I = np.identity(2,dtype=complex)
                ufmat = np.kron(I,ufmat)
                s = np.kron(I,s)
            w,v = scipylinalg.eigh(h, s,\
                  overwrite_a=True, overwrite_b=True)
            if shift:
                w = w-self.pm.ChemP
            if eV:
                w *= Hartree
            uf = np.linalg.multi_dot([v.conjugate().transpose(),ufmat,v])
            uf = np.abs(uf.diagonal().real)
            return [w,uf]
        elif (SP==1):
            h = self.pm.Hk_kvec(k, spin=1)
            s = self.pm.Sk_kvec(k, small=True)
            ufmat = self.Unfold_mat(s,k)
            w1,v = scipylinalg.eigh(h, s.copy(),\
                  overwrite_a=True, overwrite_b=True)
            uf1 = np.linalg.multi_dot([v.conjugate().transpose(),ufmat,v])
            uf1 = np.abs(uf1.diagonal().real)
            h = self.pm.Hk_kvec(k, spin=-1)
            w2,v = scipylinalg.eigh(h, s.copy(),\
                  overwrite_a=True, overwrite_b=True)
            uf2 = np.linalg.multi_dot([v.conjugate().transpose(),ufmat,v])
            uf2 = np.abs(uf2.diagonal().real)
            if shift:
                w1 = w1-self.pm.ChemP
                w2 = w2-self.pm.ChemP
            if eV:
                w1 *= Hartree
                w2 *= Hartree
            return [w1,uf1,w2,uf2]
        else:
            raise Exception("error : spin of Unfolding")

    def Unfolding_spintexture(self, k, eV=False, shift=False):
        SP = self.pm.SpinP_switch
        if (SP!=3):
            raise Exception("error: only spin non-collinear case is considered \
for Unfolding_spintexture")
        h = self.pm.Hk_kvec(k)
        s = self.pm.Sk_kvec(k, small=True)
        ufmat0 = self.Unfold_mat(s,k)
        I = np.identity(2,dtype=complex)
        ufmat = np.kron(I,ufmat0)
        s = np.kron(I,s)
        w,v = scipylinalg.eigh(h, s,\
              overwrite_a=True, overwrite_b=True)
        if shift:
            w = w-self.pm.ChemP
        if eV:
            w *= Hartree
        uf = np.linalg.multi_dot([v.conjugate().transpose(),ufmat,v])
        uf = np.abs(uf.diagonal().real)
        Sx = np.kron(0.5*s_x,ufmat0)
        Sx = np.linalg.multi_dot([v.conjugate().transpose(),Sx,v])
        Sx = Sx.diagonal().real
        Sy = np.kron(0.5*s_y,ufmat0)
        Sy = np.linalg.multi_dot([v.conjugate().transpose(),Sy,v])
        Sy = Sy.diagonal().real
        Sz = np.kron(0.5*s_z,ufmat0)
        Sz = np.linalg.multi_dot([v.conjugate().transpose(),Sz,v])
        Sz = Sz.diagonal().real
        return [w,uf,Sx,Sy,Sz]

    def Unfolding_band(self, k1, k2, n, num_print=False, eV=False, shift=False):
        ni = 1
        SP = self.pm.SpinP_switch
        path = kpath(k1,k2,n)
        k = 0.0
        klist = []
        if (SP==0)or(SP==3): 
            Elists = []
            Wlists = []
        elif (SP==1):
            Elists1 = []
            Wlists1 = []
            Elists2 = []
            Wlists2 = []
        else:
            raise Exception("error : spin of Unfolding_band")
        kbefore = path[0]
        for kvec in path:
            k += np.linalg.norm(kvec-kbefore)
            klist.append(k)
            if (SP==0)or(SP==3): 
                if num_print:
                    print("band %d/%d "%(ni,n))
                    ni += 1
                e,w = self.Unfolding(kvec, eV=eV, shift=shift)
                Elists.append(e)
                Wlists.append(w)
            elif (SP==1):
                if num_print:
                    print("band %d/%d "%(ni,n))
                    ni += 1
                e1,w1,e2,w2 = self.Unfolding(kvec, eV=eV, shift=shift)
                Elists1.append(e1)
                Wlists1.append(w1)
                Elists2.append(e2)
                Wlists2.append(w2)
            kbefore = kvec
        klist = np.array(klist)
        if (SP==0)or(SP==3): 
            Elists = np.array(Elists) 
            Wlists = np.array(Wlists) 
            return [klist, Elists, Wlists]
        elif (SP==1):
            Elists1 = np.array(Elists1) 
            Wlists1 = np.array(Wlists1) 
            Elists2 = np.array(Elists2) 
            Wlists2 = np.array(Wlists2) 
            return [klist, Elists1, Wlists1, Elists2, Wlists2]

    def Unfolding_spintexture_band(self, k1, k2, n, num_print=False, eV=False, shift=False):
        SP = self.pm.SpinP_switch
        if (SP!=3):
            raise Exception("error: only spin non-collinear case is considered \
for Unfolding_spintexture_band")
        ni = 1
        path = kpath(k1,k2,n)
        k = 0.0
        klist = []
        Elists = []
        Wlists = []
        Sxlists = []
        Sylists = []
        Szlists = []
        kbefore = path[0]
        for kvec in path:
            k += np.linalg.norm(kvec-kbefore)
            klist.append(k)
            if num_print:
                print("band %d/%d "%(ni,n))
                ni += 1
            ST = self.Unfolding_spintexture(kvec, eV=eV, shift=shift)
            Elists.append(ST[0])
            Wlists.append(ST[1])
            Sxlists.append(ST[2])
            Sylists.append(ST[3])
            Szlists.append(ST[4])
            kbefore = kvec
        klist = np.array(klist)
        Elists = np.array(Elists) 
        Wlists = np.array(Wlists) 
        Sxlists = np.array(Sxlists) 
        Sylists = np.array(Sylists) 
        Szlists = np.array(Szlists) 
        return [klist, Elists, Wlists, Sxlists, Sylists, Szlists]
    
    def save_band(self, ufband, fname='unfolded_band'):
        ln = len(ufband)
        foutname = './'+fname+'.npz'
        if (ln==3):
            klist = ufband[0]
            Elists = ufband[1]
            Wlists = ufband[2]
            np.savez(foutname, klist=klist, Elists=Elists, Wlists=Wlists,\
                     EF=self.pm.ChemP)
        elif (ln==5):
            klist = ufband[0]
            Elists1 = ufband[1]
            Wlists1 = ufband[2]
            Elists2 = ufband[3]
            Wlists2 = ufband[4]
            np.savez(foutname, klist=klist, Elists1=Elists1, Wlists1=Wlists1,\
                                            Elists2=Elists2, Wlists2=Wlists2,\
                                            EF=self.pm.ChemP)
        elif (ln==6):
            klist = ufband[0]
            Elists = ufband[1]
            Wlists = ufband[2]
            Sxlists = ufband[3]
            Sylists = ufband[4]
            Szlists = ufband[5]
            np.savez(foutname, klist=klist, Elists=Elists, Wlists=Wlists,\
                     Sxlists=Sxlists, Sylists=Sylists, Szlists=Szlists,\
                     EF=self.pm.ChemP)
        else:
            raise Exception("error : invalid input of save_band.")
    
    def load_band(self, fname):
        load_band(fname)

    def print_openmxstyle(self, ufbands, cut1, cut2, fname='unfolded_bands_omx',\
                          labels=None):
        print_openmxstyle(ufbands, cut1, cut2, EF=self.pm.ChemP, fname=fname,\
                          labels=labels)
    
    def mat_print(self, X, l=0, delimiter=''):
        mat_print(X, l=l, delimiter=delimiter)
    

def intmap_sokhotski(banddat, kptfile=None, emin=-6., emax=6., NE=401, eta=0.02,\
                     outfile='unfold_intmap_sokhotski'): 
    '''intensity map by Sokhotski formula'''
    data = np.loadtxt(banddat)
    k0 = np.around(data[:,0], decimals=6)
    E0 = np.around(data[:,1], decimals=6)
    W0 = np.around(data[:,2], decimals=6)
    klist = []
    Elist = []
    Wlist = []
    Ndata = E0.shape[0]
    d = 0.000001
    i0 = 0
    while (i0<Ndata):
        k1 = k0[i0]
        idv = np.where(np.abs(k0-k1)<d)[0]
        klist.append(k1)
        Elist.append(E0[idv])
        Wlist.append(W0[idv])
        i0 = idv[-1]+1
    klist = np.array(klist)
    
    evec = np.linspace(emin,emax,NE)
    W = []
    for ee,ww in zip(Elist,Wlist):
        wout = np.zeros(NE,dtype=float)
        for e,w in zip(ee,ww):
            wout += w*(1.j/np.pi/(evec-e+1.j*eta)).real
        W.append(wout)
    W = np.array(W).transpose()
    
    kticks = []
    labels = []
    if kptfile is not None:
        f = open(kptfile,'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            spl = line.split()
            if (len(spl)==1):
                kticks.append(float(spl[0]))
            elif (len(spl)==2):
                kticks.append(float(spl[0]))
                labels.append(spl[1])
            else:
                raise Exception("Error: wrong kpt file format in intmap_sokhotski")
    np.savez(outfile+'.npz',\
             k=klist, E=evec, W=W, kticks=kticks, labels=labels)
    
def intmap_lorentzian(banddat, kptfile=None, emin=-6., emax=6., NE=401, dE=0.02,\
                     dk=0.0, pad=2, g=1., outfile='unfold_intmap_lorentzian'): 
    '''intensity map by Lorentzian function'''
    if (dE<=0.):
        raise Exception("Error: dE in intmap_lorentzian must be positive")
    if (dk<0.):
        raise Exception("Error: dk in intmap_lorentzian must be positive or zero")
    if (g<=0.):
        raise Exception("Error: g in intmap_lorentzian must be positive")
    data = np.loadtxt(banddat)
    k0 = np.around(data[:,0], decimals=6)
    E0 = np.around(data[:,1], decimals=6)
    W0 = np.around(data[:,2], decimals=6)
    klist = []
    Elist = []
    Wlist = []
    Ndata = E0.shape[0]
    d = 0.000001
    i0 = 0
    while (i0<Ndata):
        k1 = k0[i0]
        idv = np.where(np.abs(k0-k1)<d)[0]
        klist.append(k1)
        Elist.append(E0[idv])
        Wlist.append(W0[idv])
        i0 = idv[-1]+1
    klist = np.array(klist)
    Nk = klist.shape[0]
    evec = np.linspace(emin,emax,NE)
    
    kticks = []
    labels = []
    if kptfile is not None:
        f = open(kptfile,'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            spl = line.split()
            if (len(spl)==1):
                kticks.append(float(spl[0]))
            elif (len(spl)==2):
                kticks.append(float(spl[0]))
                labels.append(spl[1])
            else:
                raise Exception("Error: wrong kpt file format in intmap_sokhotski")

    if (float(dk)==0.):
        W = []
        scale = g/np.pi/dE
        for ee,ww in zip(Elist,Wlist):
            wout = np.zeros(NE,dtype=float)
            for e,w in zip(ee,ww):
                func = np.power((evec-e)/dE,2) + g*g
                wout += w*scale/func
            W.append(wout)
        W = np.array(W).transpose()
        np.savez(outfile+'.npz',\
                 k=klist, E=evec, W=W, kticks=kticks, labels=labels)

    else:
        if (pad==0):
            klist2 = klist
            Nk2 = Nk
        elif (pad<0):
            raise Exception("Error: pad in intmap_lorentzian must be positive or zero")
        else:
            Nk2 = (Nk-1)*(pad+1)+1
            klist2 = []
            for i in range(Nk-1):
                ki = klist[i]
                kf = klist[i+1]
                for k in np.linspace(ki,kf,pad+2)[:-1]:
                    klist2.append(k)
            klist2.append(klist[-1])
        klist2 = np.array(klist2)
        W = np.zeros((NE,Nk2),dtype=float)
        kgrid = np.array([klist2]*NE)
        egrid = np.array([evec]*Nk2).transpose()

        scale = 0.5*g/np.pi/dE/dk
        for k,ee,ww in zip(klist,Elist,Wlist):
            for e,w in zip(ee,ww):
                func1 = np.power((kgrid-k)/dk,2) +np.power((egrid-e)/dE,2) +g*g
                func2 = np.power(func1,1.5)
                W += w*scale/func2
        np.savez(outfile+'.npz',\
                 k=klist2, E=evec, W=W, kticks=kticks, labels=labels)

def plot_intmap(intdat, figsize=None, norm='log', vmin=1., vmax=None,\
                cmap='viridis', colorbar=False, cbar_ticks=[],\
                **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    f = np.load(intdat)
    k = f['k']  
    E = f['E']
    W = f['W'] 
    kticks = f['kticks']
    labels = f['labels']
    f.close()
    labels2 = [l.decode('utf-8') for l in labels]

    fig1 = plt.figure('unfolding_intensity_map', figsize=figsize)
    ax1 = fig1.add_subplot(111)
    if (type(norm)==str):
        if (norm=='log'):
            mynorm = colors.LogNorm(vmin=vmin, vmax=vmax)
        elif (norm=='linear'):
            mynorm = colors.Normalize(vmin=vmin, vmax=vmax)
        elif (norm=='symlog'):
            if vmax is None:
                vmin2 = None
            else:
                vmin2 = -vmax 
            mynorm = colors.SymLogNorm(linthresh=vmin,vmin=vmin2,vmax=vmax)
        else:
            mynorm = None
    else:
        mynorm = norm
    BC = ax1.pcolormesh(k,E,W,norm=mynorm,cmap=cmap, shading='gouraud', **kwargs)
    ax1.set_xticks(kticks)
    ax1.set_xticklabels(labels2)
    plt.grid(which='major', axis='x', linewidth=1.)
    plt.ylabel('Energy (eV)')
    if colorbar:
        plt.colorbar(BC, ax=ax1, ticks=cbar_ticks)
    plt.tight_layout()
    plt.show()
    
def spin_rotation(ufband, X, Z):    
    if len(ufband)!=6:
        raise Exception('error: input of spin_rotation should be \
the output of Unfolding_spintexture_band')
    klist, Elists, Wlists, Sxlists, Sylists, Szlists = ufband
    cri = 1.e-9
    if (np.abs(np.dot(X,Z))>cri):
        raise Exception('error: X and Z are not orthogonal in spin_rotation')
    x = X/np.linalg.norm(X)
    z = Z-x*np.dot(x,Z)
    z = z/np.linalg.norm(z)
    y = np.cross(z,x)
    y = y/np.linalg.norm(y)
    Sx = Sxlists*x[0] + Sylists*x[1] + Szlists*x[2]
    Sy = Sxlists*y[0] + Sylists*y[1] + Szlists*y[2]
    Sz = Sxlists*z[0] + Sylists*z[1] + Szlists*z[2]
    return [klist, Elists, Wlists, Sx, Sy, Sz]
    
