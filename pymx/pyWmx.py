from .pymx_common import *
import numpy as np
import sys


class PyWMX():
    
    def __init__(self, HWR_file):
        self.HWR_file = HWR_file

        self.a1,self.a2,self.a3 = None,None,None
        self.b1,self.b2,self.b3 = None,None,None
        self.WFNum,self.SCNum,self.Vcell = None,None,None
        self.spinPol,self.EF = None,None
        self.tau, self.projector = None,None
        self.R_num_mat,self.R_mat,self.R_dege = None,None,None
        self.H_R1,self.H_R2 = None,None

    basis_num = {'s':1, 'p':3, 'px':1, 'py':1, 'pz':1, 'd':5,\
                 'dz2':1, 'dx2-y2':1, 'dxy':1, 'dxz':1, 'dyz':1, 'f':7,\
                 'fz3':1, 'fxz2':1, 'fyz2':1, 'fzx2':1, 'fxyz':1,\
                 'fx3-3xy2':1, 'f3yx2-y3':1, 'sp':2, 'sp2':3, 'sp3':4,\
                 'sp3dz2':5, 'sp3deg':6}

    def read_file(self):
        fHWR = open(self.HWR_file,'r')
        lines = fHWR.readlines()
        fHWR.close()

        self.WFNum = int(lines[1].split()[-1])
        self.SCNum = int(lines[2].split()[-1])
        vec = lines[4].split()
        self.a1 = np.array([float(vec[0]),float(vec[1]),float(vec[2])])
        vec = lines[5].split()
        self.a2 = np.array([float(vec[0]),float(vec[1]),float(vec[2])])
        vec = lines[6].split()
        self.a3 = np.array([float(vec[0]),float(vec[1]),float(vec[2])])
        self.Vcell = np.dot(np.cross(self.a1,self.a2),self.a3)
        self.b1 = 2.*np.pi*np.cross(self.a2,self.a3)/self.Vcell
        self.b2 = 2.*np.pi*np.cross(self.a3,self.a1)/self.Vcell
        self.b3 = 2.*np.pi*np.cross(self.a1,self.a2)/self.Vcell
        self.spinPol = int(lines[7].split()[-1]) 
        self.EF = float(lines[8].split()[-1]) 
        SP = self.spinPol

        self.R_num_mat = np.empty((3,self.SCNum),dtype=int)
        self.R_mat = np.empty((3,self.SCNum),dtype=float)
        self.H_R1 = np.empty((self.SCNum,self.WFNum,self.WFNum),dtype=complex)
        if (SP==2):
            self.H_R2 = np.empty((self.SCNum,self.WFNum,self.WFNum),dtype=complex)
        self.R_dege = np.empty(self.SCNum,dtype=int)
        i_R = -1
        line_spin = self.SCNum*(self.WFNum*self.WFNum+1)
        for line in lines[9:9+line_spin]:
            spl = line.split()
            if (spl[0] == 'R'):
                i_R += 1
                self.R_num_mat[:,i_R] = np.array([int(spl[2]),int(spl[3]),int(spl[4])])
                self.R_dege[i_R] = int(spl[6])
                dege = float(spl[6])
            elif (len(spl) == 4):
                self.H_R1[i_R,int(spl[0])-1,int(spl[1])-1] = \
                    (float(spl[2]) + 1.j*float(spl[3])) / dege
        if (SP==2):
            i_R = -1
            for line in lines[9+line_spin:9+2*line_spin]:
                spl = line.split()
                if (spl[0] == 'R'):
                    i_R += 1
                    dege = float(spl[6])
                elif (len(spl) == 4):
                    self.H_R2[i_R,int(spl[0])-1,int(spl[1])-1] = \
                        (float(spl[2]) + 1.j*float(spl[3])) / dege

        for i_R in range(self.SCNum):
            Ns = self.R_num_mat[:,i_R].astype(float)
            R = Ns[0]*self.a1 + Ns[1]*self.a2 + Ns[2]*self.a3
            self.R_mat[:,i_R] = R

    def read_tau_proj(self, datfile):
        f = open(datfile,'r')
        lines = f.readlines()
        f.close()
        unit_txt = 'Wannier.Initial.Projectors.Unit'.upper()
        tau_txt1 = '<Wannier.Initial.Projectors'.upper()
        tau_txt2 = 'Wannier.Initial.Projectors>'.upper()
        tau_switch = False
        tau_line = []
        for line in lines:
            spl = line.split()
            l = len(spl)
            if ((l!=0) and (spl[0].upper() == unit_txt)):
                unit = spl[1][:2].upper()
            if ((l!=0) and (spl[0].upper() == tau_txt2)):
                tau_switch = False
            if (tau_switch):
                tau_line.append(spl)
            if ((l!=0) and (spl[0].upper() == tau_txt1)):
                tau_switch = True
        tau, projector = [],[]
        for tline in tau_line:
            pr, p1i, p2i, p3i = tline[:4]
            proj = pr.split('-')[-1]
            p1, p2, p3 = float(p1i), float(p2i), float(p3i)
            for i in range(self.basis_num[proj]):
                projector.append(proj)
                if ((unit == 'AU') or (unit == 'BO')):
                    tau.append(np.array([p1,p2,p3]))
                elif (unit == 'AN'):
                    tau.append(np.array([p1,p2,p3])*(1./Bohr))
                elif (unit == 'FR'):
                    t = p1*self.a1 + p2*self.a2 + p3*self.a3
                    tau.append(t)
                else:
                    raise Exception('wrong unit of tau')
        self.tau = tau
        self.projector = projector

    def read_tau_center(self, stdfile, WFNum=None):
        if (WFNum is None):
            wn = self.WFNum
        else:
            wn = int(WFNum)
        f = open(stdfile,'r')
        lines = f.readlines()
        f.close()
        ctxt1,ctxt2,ctxt3,ctxt4 = 'CENTER','OF','WANNIER','FUNCTION' 
        lnum = 0
        for i,line in enumerate(lines):
            spl = line.split()
            if (len(spl) > 6):
                s1 = spl[0].upper()
                s2 = spl[1].upper()
                s3 = spl[2].upper()
                s4 = spl[3].upper()
                if(s1==ctxt1 and s2==ctxt2 and s1==ctxt1 and s1==ctxt1):
                    lnum = i
        if (lnum==0):
            raise Exception('wrong input file for read_rau_center')
        tau = []
        for line in lines[lnum+1:lnum+1+wn]:
            r1,r2,r3 = ((line.split('(')[1]).split(')')[0]).split(',')
            t = np.array([float(r1),float(r2),float(r3)])*(1./Bohr)
            tau.append(t)
        self.tau = tau

    def tau_frac(self, frac):
        if (len(frac)!=self.WFNum):
            raise Exception('wrong input for tau_frac')
        tau = []
        for i in range(len(frac)):
            p1,p2,p3 = frac[i]
            t = p1*self.a1 + p2*self.a2 + p3*self.a3
            tau.append(t)
        self.tau = tau

    #phase factor exp(ik dot tau)matrix from the basis tau
    def phase_tau(self, k, mat_size=1):
        N = self.WFNum
        ekt = np.zeros((N,N),dtype=complex)
        for i in range(N):
            t = self.tau[i] 
            kdott = np.dot(k,t)
            ekt[i,i] = np.exp(1.j*kdott)
        if (mat_size != 1):
            I = np.identity(2,dtype=complex)
            ekt = np.kron(I,ekt)
        return ekt

    def kdotR_vec(self, k):
        return np.dot(k,self.R_mat)

    def exp_vec(self, k):
        return np.exp(1.j*np.dot(k,self.R_mat))

    def Hk(self, exp_vec, spin=False):
        SP = self.spinPol
        if (SP==1):
            h = np.tensordot(exp_vec,self.H_R1,axes=((0),(0)))
            return h
        elif (SP==2):
            if (spin==False):
                h1 = np.tensordot(exp_vec,self.H_R1,axes=((0),(0)))
                h2 = np.tensordot(exp_vec,self.H_R2,axes=((0),(0)))
                zero = np.zeros(h1.shape,dtype=complex)
                return np.block([[h1,zero],[zero,h2]])
            else:
                if (spin>0):
                    h1 = np.tensordot(exp_vec,self.H_R1,axes=((0),(0)))
                    return h1
                elif (spin<0):
                    h2 = np.tensordot(exp_vec,self.H_R2,axes=((0),(0)))
                    return h2
                else:
                    raise Exception("error : spin of Hk")

    def Hk_kvec(self, k, spin=False):
        exp_vec = np.exp(1.j*np.dot(k,self.R_mat))
        sp = spin
        return self.Hk(exp_vec, spin=sp)

    def Hk_cellperiodic(self, k, spin=False):
        if (self.tau != None):
            SP = self.spinPol
            exp_vec = np.exp(1.j*np.dot(k,self.R_mat))
            sp = spin
            H = self.Hk(exp_vec, spin=sp)
            if ((SP==1)or(spin==False)):
                ms = 1
            else:
                ms = 2
            U = self.phase_tau(k, mat_size=ms)
            return np.linalg.multi_dot([U.conjugate(),H,U])
        else:
            raise Exception("tau is not defined\n")

    def find_cell(self, n1, n2, n3):
        for i_R in range(self.SCNum):
            m1,m2,m3 = self.R_num_mat[:,i_R]
            if (n1==m1 and n2==m2 and n3==m3):
                return i_R
        else:
            return None

    def cell_i_info(self, i_R, spin=False):
        Ns = self.R_num_mat[:,i_R]
        print("R indices : %d  %d  %d"%(Ns[0],Ns[1],Ns[2]))
        R = self.R_mat[:,i_R]
        print("R : %.10f  %.10f  %.10f"%(R[0],R[1],R[2]))
        print("H(R) spin1")
        H = self.H_R1[i_R,:,:]
        for i in range(self.WFNum):
            for j in range(self.WFNum):
                v = H[i,j]
                if (j < self.WFNum-1):
                    print("%.10f+j%.10f"%(v.real,v.imag)),
                else:
                    print("%.10f+j%.10f"%(v.real,v.imag))
        SP = self.spinPol
        if (self.spinPol==2):
            print("H(R) spin2")
            H = self.H_R1[i_R,:,:]
            for i in range(self.WFNum):
                for j in range(self.WFNum):
                    v = H[i,j]
                    if (j < self.WFNum-1):
                        print("%.10f+j%.10f"%(v.real,v.imag)),
                    else:
                        print("%.10f+j%.10f"%(v.real,v.imag))

    def cell_info(self, n1, n2, n3):
        i_R = self.find_cell(n1,n2,n3)
        if (type(i_R) is int):
            self.cell_i_info(i_R)

    def Band(self, k1, k2, n, spin=False):
        path = kpath(k1,k2,n)
        k = 0.0
        klist = []
        Elists = []
        kbefore = path[0]
        for kvec in path:
            k += np.linalg.norm(kvec-kbefore)
            klist.append(k)
            exp_vec = self.exp_vec(kvec)
            h = self.Hk(exp_vec, spin=spin)
            w = np.linalg.eigvalsh(h)
            Elists.append(w)
            kbefore = kvec
        E = np.transpose(np.array(Elists))
        out = [np.array(klist)]
        for i in range(E.shape[0]):
            out.append(E[i,:])
        #out[0]:k-axis value from 0, out[1:]:energy eigenvalues
        return out

    def PlotBand(self, Band_list, kticks_label=None, yrange=None,
                 shift=False, eV=False, EF=None, highlight=None, save=False,
                 fname=None, c1='b', c2='r'):
        PlotBand(Band_list, kticks_label= kticks_label,
                 shift=shift, yrange=yrange, eV=eV, EF=EF, highlight=highlight,
                 save=save, fname=fname, c1=c1, c2=c2)

    def Zak_phase(self, k0, G, n, bands, total=True):
        dk = G/float(n-1)
        PI_det = 1.+0.j
        klist = kpath(k0,k0+G,n)
        eigvecs = []
        for k in klist[:-1]:
            if (total):
                h = self.Hk_cellperiodic(k)
            else:
                h = self.Hk_kvec(k)
            w,v = np.linalg.eigh(h)
            eigvecs.append(v[:,bands[0]:bands[1]+1])
        v_end = np.dot(self.phase_tau(G).conjugate(),eigvecs[0])
        eigvecs.append(v_end)
        for i in range(n-1):
            Ck1h = eigvecs[i].conjugate().transpose()
            Ck2 = eigvecs[i+1]
            unum_mat = np.dot(Ck1h,Ck2)
            PI_det *= np.linalg.det(unum_mat) 
        return -1.*np.imag(np.log(PI_det))

