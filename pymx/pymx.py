
# from 2018/07/23

from .pyread_scfout import *
from .pymx_common import *
import numpy as np
import sys
import copy

####### silence cython warnings ########
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
########################################
import scipy.linalg as scipylinalg


class PyMX(ReadScfout):
    ''' openmx python tool '''
    def __init__(self, scfout_file):
        ReadScfout.__init__(self)
        self.input_file(scfout_file)
        #variables of ReadScfout
        self.atomnum, self.SpinP_switch = None, None
        self.Catomnum, self.Latomnum, self.Ratomnum = None, None, None
        self.TCpyCell = None
        self.atv, self.atv_ijk = None, None
        self.Total_NumOrbs = None    
        self.FNAN, self.natn = None, None
        self.ncn, self.tv, self.rtv = None, None, None
        self.Gxyz = None  
        self.Hks, self.iHks, self.OLP = None, None, None
        self.OLPpox, self.OLPpoy, self.OLPpoz = None, None, None
        self.DM = None
        self.Solver, self.ChemP, self.E_Temp = None, None, None
        self.dipole_moment_core = None
        self.dipole_moment_background = None
        self.Valence_Electrons = None
        self.Total_SpinS = None
        self.temporal_input = None
        #variables of PyMX
        self.a1, self.a2, self.a3 = None, None, None
        self.b1, self.b2, self.b3 = None, None, None
        self.R_mat, self.R_size, self.tau = None, None, None
        self.mat_size, self.AtOrb, self.At_range = None, None, None
        self.Rn_reduced, self.R_size_reduced = None, None
        self.R_mapping, self.R_mat_reduced = None, None
        self.H_R1, self.H_R2, self.H_R3, self.H_R4 =None,None,None,None
        self.iH_R1, self.iH_R2, self.iH_R3 = None,None,None
        self.OLP_R = None
        self.OLPpox_R, self.OLPpoy_R, self.OLPpoz_R= None, None, None
        self.PAO_list, self.atom_list = None, None
        self.PAO_dict, self.basis_list = None, None

    def read_file(self):
        self.read_scfout(self.scfout_file)

        #lattice unit cell vector
        self.a1 = np.array(self.tv[1][1:])
        self.a2 = np.array(self.tv[2][1:])
        self.a3 = np.array(self.tv[3][1:])

        #reciprocal lattice unit cell vector
        self.b1 = np.array(self.rtv[1][1:])
        self.b2 = np.array(self.rtv[2][1:])
        self.b3 = np.array(self.rtv[3][1:])

        #lattice translation vectors in matrix form
        # (k dot R array) = numpy.dot(k,R_mat)
        self.R_mat = np.transpose(self.atv[:,1:])
        self.R_size = self.TCpyCell +1

        #about lattice vector of lattice that has non-zero Hks
        Rset = set()
        for l in self.ncn[1:]:
            Rset.update(l)
        self.Rn_reduced = list(Rset)
        self.Rn_reduced.sort()
        self.R_size_reduced = len(self.Rn_reduced)

        self.R_mapping = [0]*self.R_size
        for i in range(self.R_size_reduced):
            self.R_mapping[self.Rn_reduced[i]] = i

        self.R_mat_reduced = np.empty((3,self.R_size_reduced))
        for i in range(self.R_size_reduced):
            self.R_mat_reduced[:,i] = self.R_mat[:,self.Rn_reduced[i]]

        #atomic position coordinates in Bohr
        #tau[0] is dummy
        self.tau = []
        for i in range(self.atomnum+1):
            self.tau.append(np.array(self.Gxyz[i,1:4]))

        #matrix size (number of basis)
        self.mat_size = np.sum(self.Total_NumOrbs[1:])

        ms = self.mat_size
        Rs = self.R_size_reduced

        #AtOrb[basis index] = [atom index, orbital index]
        AtOrb = []
        for i in range(self.atomnum):
            for j in range(self.Total_NumOrbs[i+1]):
                AtOrb.append([i+1,j+1])
        self.AtOrb = np.array(AtOrb)

        #(index range of Atom i) = [At_range[i][0]:At_range[i][1]]
        At_range = [[0,0]] #dummy
        before = 0
        for i in range(self.atomnum):
            orb = self.Total_NumOrbs[i+1]
            At_range.append([before,before+orb])
            before += orb
        self.At_range = np.array(At_range)

        #hopping matrix component
        # array of R matrices
        H_R_list = []
        for spin in range(self.SpinP_switch+1):
            H_R = np.zeros((Rs,ms,ms),dtype=float)
            for ct_AN in range(self.atomnum+1)[1:]: #i
                for h_AN in range(self.FNAN[ct_AN]+1): #j
                    Rn = self.R_mapping[self.ncn[ct_AN][h_AN]]
                    Gh_AN = self.natn[ct_AN][h_AN]
                    ct_i,ct_f = self.At_range[ct_AN]
                    h_i,h_f = self.At_range[Gh_AN]
                    H_R[Rn,ct_i:ct_f,h_i:h_f] = self.Hks[spin][ct_AN][h_AN]
            H_R_list.append(H_R)
        SP = self.SpinP_switch
        self.H_R1 = H_R_list[0]
        if (SP==1):
            self.H_R2 = H_R_list[1]
        elif (SP==3):
            self.H_R2 = H_R_list[1]
            self.H_R3 = H_R_list[2]
            self.H_R4 = H_R_list[3]
        del(H_R_list)

        #imagianry part of hopping matrix component
        # array of R matrices
        iH_R_list = []
        for spin in range(3):
            iH_R = np.zeros((Rs,ms,ms),dtype=float)
            for ct_AN in range(self.atomnum+1)[1:]: #i
                for h_AN in range(self.FNAN[ct_AN]+1): #j
                    Rn = self.R_mapping[self.ncn[ct_AN][h_AN]]
                    Gh_AN = self.natn[ct_AN][h_AN]
                    ct_i,ct_f = self.At_range[ct_AN]
                    h_i,h_f = self.At_range[Gh_AN]
                    iH_R[Rn,ct_i:ct_f,h_i:h_f] = self.iHks[spin][ct_AN][h_AN]
            iH_R_list.append(iH_R)
        SP = self.SpinP_switch
        self.iH_R1 = iH_R_list[0]
        if (SP==1):
            self.iH_R2 = iH_R_list[1]
        elif (SP==3):
            self.iH_R2 = iH_R_list[1]
            self.iH_R3 = iH_R_list[2]
        del(iH_R_list)

        #Overlap matrix component
        # array of R matrices
        self.OLP_R = np.zeros((Rs,ms,ms),dtype=float)
        for ct_AN in range(self.atomnum+1)[1:]: #i
            for h_AN in range(self.FNAN[ct_AN]+1): #j
                Rn = self.R_mapping[self.ncn[ct_AN][h_AN]]
                Gh_AN = self.natn[ct_AN][h_AN]
                ct_i,ct_f = self.At_range[ct_AN]
                h_i,h_f = self.At_range[Gh_AN]
                self.OLP_R[Rn,ct_i:ct_f,h_i:h_f] = self.OLP[ct_AN][h_AN]

        #order of basis orbital in matrix
        self.PAO_list = []
        self.atom_list = ['dummy']
        PAO_ikwd = '<Definition.of.Atomic.Species'.upper()
        PAO_fkwd = 'Definition.of.Atomic.Species>'.upper()
        atom_ikwd = '<Atoms.SpeciesAndCoordinates'.upper()
        atom_fkwd = 'Atoms.SpeciesAndCoordinates>'.upper()
        PAO_TF = False
        atom_TF = False
        for line in self.temporal_input:
            spl = line.split()
            if (len(spl) != 0):
                if (spl[0].upper() == PAO_fkwd): PAO_TF = False
                if (spl[0].upper() == atom_fkwd): atom_TF = False
            if (PAO_TF == True): self.PAO_list.append([spl[0],spl[1]])
            if (atom_TF == True): self.atom_list.append(spl[1])
            if (len(spl) != 0):
                if (spl[0].upper() == PAO_ikwd): PAO_TF = True
                if (spl[0].upper() == atom_ikwd): atom_TF = True
        self.PAO_dict = dict()
        m_dict = {'s':1, 'p':3, 'd':5, 'f':7}
        for pao in self.PAO_list:
            orbs = pao[1].split('-')[1]
            self.PAO_dict[pao[0]] = orbs
        self.basis_list = []
        for A in self.atom_list[1:]:
            orbs = self.PAO_dict[A]
            for i in range(int(len(orbs)/2)):
                l, N = orbs[2*i], int(orbs[2*i+1])
                for j in range(N):
                    for m in range(m_dict[l]):
                        mnum = str(m+1)
                        self.basis_list.append(A+'-'+l+mnum)

    def del_rawdata(self):
        del(self.Hks)
        del(self.iHks)
        del(self.OLP)
        del(self.OLPpox)
        del(self.OLPpoy)
        del(self.OLPpoz)
        del(self.DM)
        self.Hks, self.iHks, self.OLP = None, None, None
        self.OLPpox, self.OLPpoy, self.OLPpoz = None, None, None
        self.DM = None

    #get Overlap with position vector matrix component
    # array of R matrices
    def get_XYZ(self):
        ms = self.mat_size
        Rs = self.R_size_reduced

        self.OLPpox_R = np.zeros((Rs,ms,ms),dtype=float)
        for ct_AN in range(self.atomnum+1)[1:]: #i
            for h_AN in range(self.FNAN[ct_AN]+1): #j
                Rn = self.R_mapping[self.ncn[ct_AN][h_AN]]
                Gh_AN = self.natn[ct_AN][h_AN]
                ct_i,ct_f = self.At_range[ct_AN]
                h_i,h_f = self.At_range[Gh_AN]
                self.OLPpox_R[Rn,ct_i:ct_f,h_i:h_f] = self.OLPpox[ct_AN][h_AN]

        self.OLPpoy_R = np.zeros((Rs,ms,ms),dtype=float)
        for ct_AN in range(self.atomnum+1)[1:]: #i
            for h_AN in range(self.FNAN[ct_AN]+1): #j
                Rn = self.R_mapping[self.ncn[ct_AN][h_AN]]
                Gh_AN = self.natn[ct_AN][h_AN]
                ct_i,ct_f = self.At_range[ct_AN]
                h_i,h_f = self.At_range[Gh_AN]
                self.OLPpoy_R[Rn,ct_i:ct_f,h_i:h_f] = self.OLPpoy[ct_AN][h_AN]

        self.OLPpoz_R = np.zeros((Rs,ms,ms),dtype=float)
        for ct_AN in range(self.atomnum+1)[1:]: #i
            for h_AN in range(self.FNAN[ct_AN]+1): #j
                Rn = self.R_mapping[self.ncn[ct_AN][h_AN]]
                Gh_AN = self.natn[ct_AN][h_AN]
                ct_i,ct_f = self.At_range[ct_AN]
                h_i,h_f = self.At_range[Gh_AN]
                self.OLPpoz_R[Rn,ct_i:ct_f,h_i:h_f] = self.OLPpoz[ct_AN][h_AN]

    def default_setting(self):
        self.read_file()
        self.get_XYZ()
        self.del_rawdata()

    def print_basis(self):
        N = self.mat_size
        SP = self.SpinP_switch
        if (SP == 0):
            for i in range(N): print(" %d "%i +self.basis_list[i])
        else:
            for i in range(N): print(" %d "%i +self.basis_list[i])
            for i in range(N): print(" %d "%(i+N) +self.basis_list[i])

    def delete_orbital(self, del_list):
        if (type(del_list)==int):
            del_list = [del_list]
        del_len = len(del_list)
        self.mat_size = self.mat_size - del_len
        AtOrb = np.delete(self.AtOrb,del_list,0)
        TNO = np.zeros(1+self.atomnum,dtype=int)
        I,A = 1,1
        for i in range(len(AtOrb)):
            if (A != AtOrb[i,0]):
                A = AtOrb[i,0]
                I = 1
            AtOrb[i,1] = I
            TNO[A] = I
            I += 1
        self.AtOrb = AtOrb
        self.Total_NumOrbs = TNO
        At_range = [[0,0]] #dummy
        before = 0
        for i in range(self.atomnum):
            orb = self.Total_NumOrbs[i+1]
            At_range.append([before,before+orb])
            before += orb
        self.At_range = np.array(At_range)
        del_l2 = copy.deepcopy(del_list)
        del_l2.sort()
        del_l2.reverse()
        for i in del_l2:
            del(self.basis_list[i])   
        MAT_R_list = [self.H_R1, self.H_R2, self.H_R3, self.H_R4, \
                      self.iH_R1, self.iH_R2, self.iH_R3, self.OLP_R, \
                      self.OLPpox_R, self.OLPpoy_R, self.OLPpoz_R]
        Mdel = [None]*11
        for i,MAT in enumerate(MAT_R_list):
            if (MAT is not None):
                MAT1 = np.delete(MAT, del_list, 1)
                Mdel[i] = np.delete(MAT1, del_list, 2)
        self.H_R1, self.H_R2, self.H_R3, self.H_R4 = Mdel[:4]
        self.iH_R1, self.iH_R2, self.iH_R3, self.OLP_R = Mdel[4:8]
        self.OLPpox_R, self.OLPpoy_R, self.OLPpoz_R = Mdel[8:]

    def delete_atom(self, del_list):
        if (type(del_list)==int):
            del_list = [del_list]
        del_list.sort()
        del_list.reverse()
        for i in del_list:
            i1,i2 = self.At_range[i,:]
            orb_list = list(range(i1,i2))
            self.delete_orbital(orb_list)

    def translate_tau(self, T):
        tau1 = copy.deepcopy(self.tau)
        tau2 = [np.array([0.,0.,0.])]
        for t in tau1[1:]:
            tau2.append(t+T)
        self.tau = tau2

    #list of k dot R
    def kdotR_vec(self, k):
        return np.dot(k,self.R_mat_reduced)

    #list of exp(k dot R)
    def exp_vec(self, k):
        return np.exp(1.j*np.dot(k,self.R_mat_reduced))

    #list of R_i*exp(k dot R)
    def Ri_exp_vec(self, exp_vec, i):
        return exp_vec*self.R_mat_reduced[i,:]

    #Hamiltonian matrix at k. input is exp_vec(k)
    def Hk(self, exp_vec, spin=False):
        SP = self.SpinP_switch
        if (SP==0):
            h = np.tensordot(exp_vec,self.H_R1,axes=((0),(0)))
            return h
        elif (SP==1):
            if (spin == False):
                h1 = np.tensordot(exp_vec,self.H_R1,axes=((0),(0)))
                h2 = np.tensordot(exp_vec,self.H_R2,axes=((0),(0)))
                zero = np.zeros(h1.shape)
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
        elif (SP==3):
            h11r = np.tensordot(exp_vec,self.H_R1,axes=((0),(0)))
            h11i = np.tensordot(exp_vec,self.iH_R1,axes=((0),(0)))
            h11 = h11r + 1.j*h11i
            h22r = np.tensordot(exp_vec,self.H_R2,axes=((0),(0)))
            h22i = np.tensordot(exp_vec,self.iH_R2,axes=((0),(0)))
            h22 = h22r + 1.j*h22i
            h12r = np.tensordot(exp_vec,self.H_R3,axes=((0),(0)))
            h12i = np.tensordot(exp_vec,self.iH_R3,axes=((0),(0)))+\
                   np.tensordot(exp_vec,self.H_R4,axes=((0),(0)))
            h12 = h12r + 1.j*h12i
            h21 = h12.conjugate().transpose()
            return np.block([[h11,h12],[h21,h22]])

    #Hamiltonian matrix at k. input is k
    def Hk_kvec(self, k, spin=False):
        exp_vec = np.exp(1.j*np.dot(k,self.R_mat_reduced))
        sp = spin
        return self.Hk(exp_vec, spin=sp)

    #k_i-derivative of Hk
    def dHk_dki(self, exp_vec, i, spin=False):
        Rexp = self.Ri_exp_vec(exp_vec, i)
        sp = spin
        return 1.j*self.Hk(Rexp, spin=sp)

    #Overlap matrix at k. input is exp_vec(k)
    def Sk(self, exp_vec, spin=False, small=False):
        SP = self.SpinP_switch
        s = np.tensordot(exp_vec,self.OLP_R,axes=((0),(0)))
        if (SP==0 or small):
            return s
        elif (SP==1):
            if (spin == False):
                I = np.identity(2,dtype=complex)
                return np.kron(I,s)
            else:
                return s
        elif (SP==3):
            I = np.identity(2,dtype=complex)
            return np.kron(I,s)

    #Overlap matrix at k. input is k
    def Sk_kvec(self, k, spin=False, small=False):
        exp_vec = np.exp(1.j*np.dot(k,self.R_mat_reduced))
        sp = spin
        sm = small
        return self.Sk(exp_vec, spin=sp, small=sm)

    #k_i-derivative of Sk
    def dSk_dki(self, exp_vec, i, spin=False, small=False):
        Rexp = self.Ri_exp_vec(exp_vec, i)
        sp = spin
        sm = small
        return 1.j*self.Sk(Rexp, spin=sp, small=sm)

    #Overlap with position x matrix at k. input is exp_vec(k)
    def Xk(self, exp_vec, spin=False, small=False):
        SP = self.SpinP_switch
        x = np.tensordot(exp_vec,self.OLPpox_R,axes=((0),(0)))
        if (SP==0 or small):
            return x
        elif (SP==1):
            if (spin == False):
                I = np.identity(2,dtype=complex)
                return np.kron(I,x)
            else:
                return x
        elif (SP==3):
            I = np.identity(2,dtype=complex)
            return np.kron(I,x)

    #Overlap with position x matrix at k. input is k
    def Xk_kvec(self, k, spin=False, small=False):
        exp_vec = np.exp(1.j*np.dot(k,self.R_mat_reduced))
        sp = spin
        sm = small
        return self.Xk(exp_vec, spin=sp, small=sm)

    #Overlap with position y matrix at k. input is exp_vec(k)
    def Yk(self, exp_vec, spin=False, small=False):
        SP = self.SpinP_switch
        y = np.tensordot(exp_vec,self.OLPpoy_R,axes=((0),(0)))
        if (SP==0 or small):
            return y
        elif (SP==1):
            if (spin == False):
                I = np.identity(2,dtype=complex)
                return np.kron(I,y)
            else:
                return y
        elif (SP==3):
            I = np.identity(2,dtype=complex)
            return np.kron(I,y)

    #Overlap with position y matrix at k. input is k
    def Yk_kvec(self, k, spin=False, small=False):
        exp_vec = np.exp(1.j*np.dot(k,self.R_mat_reduced))
        sp = spin
        sm = small
        return self.Yk(exp_vec, spin=sp, small=sm)

    #Overlap with position z matrix at k. input is exp_vec(k)
    def Zk(self, exp_vec, spin=False, small=False):
        SP = self.SpinP_switch
        z = np.tensordot(exp_vec,self.OLPpoz_R,axes=((0),(0)))
        if (SP==0 or small):
            return z
        elif (SP==1):
            if (spin == False):
                I = np.identity(2,dtype=complex)
                return np.kron(I,z)
            else:
                return z
        elif (SP==3):
            I = np.identity(2,dtype=complex)
            return np.kron(I,z)

    #Overlap with position z matrix at k. input is k
    def Zk_kvec(self, k, spin=False, small=False):
        exp_vec = np.exp(1.j*np.dot(k,self.R_mat_reduced))
        sp = spin
        sm = small
        return self.Zk(exp_vec, spin=sp, small=sm)

    def TB_eigen(self, k, spin=False):
        exp_vec = self.exp_vec(k)
        h = self.Hk(exp_vec, spin=spin)
        s = self.Sk(exp_vec, spin=spin)
        eig = scipylinalg.eigh(h, s, \
           overwrite_a=True,overwrite_b=True,turbo=False)
        return eig

    #phase factor exp(ik dot tau)matrix from the basis tau
    def phase_tau(self, k, mat_size=1):
        N = self.mat_size
        ekt = np.zeros((N,N),dtype=complex)
        for i in range(N):
            t = self.tau[self.AtOrb[i,0]] 
            kdott = np.dot(k,t)
            ekt[i,i] = np.exp(1.j*kdott)
        if (mat_size != 1):
            I = np.identity(2,dtype=complex)
            ekt = np.kron(I,ekt)
        return ekt

    def Crystal_symmetry_pair(self, R, center=np.array([0.,0.,0.]),\
                              t=np.array([0.,0.,0.]), cri=0.001):
        ddet = abs(abs(np.linalg.det(R))-1.)
        if (ddet > 0.0001):
            raise Exception("error: R is not orthogonal in Crystal_symmetry_pair")
        Rnear = []
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    Rnear.append(i*self.a1 + j*self.a2 + k*self.a3)
        ctterm = np.dot((np.identity(3,dtype=float)-R),center)+t
        pairs = []
        for i in range(self.atomnum+1)[1:]:
            t = self.tau[i]
            Rt = np.dot(R,t) + ctterm
            p, pR = None, None
            for j in range(self.atomnum+1)[1:]:
                for r in Rnear:
                    Rtj = self.tau[j] + r
                    diff = np.linalg.norm(Rt-Rtj)
                    if (diff < cri):
                        p = j
                        pR = r
            pairs.append([i,p,pR])
        none_list = []
        for pair in pairs:
            i,p,pR = pair
            if (p==None):
                none_list.append(i)
        if (len(none_list)!=0):
            Rnear = []
            for i in range(-3,4):
                for j in range(-3,4):
                    for k in range(-3,4):
                        Rnear.append(i*self.a1 + j*self.a2 + k*self.a3)
            for i in none_list:
                t = self.tau[i]
                Rt = np.dot(R,t) + ctterm
                p, pR = None, None
                for j in range(self.atomnum+1)[1:]:
                    for r in Rnear:
                        Rtj = self.tau[j] + r
                        diff = np.linalg.norm(Rt-Rtj)
                        if (diff < cri):
                            p = j
                            pR = r
                pairs[i-1] = [i,p,pR]
        return pairs

    def Crystal_symmetry_mat(self, R, k, SU2=None,\
                     center=np.array([0.,0.,0.]), t=np.array([0.,0.,0.]),\
                     cri=0.001, spin=False, no_pair_ignore=False,\
                     overlap=True):
        N = self.mat_size
        O = Rl(R)
        R0,R1,R2,R3 = O.get_R()
        pair = self.Crystal_symmetry_pair(R, center=center, t=t, cri=cri)
        if not no_pair_ignore:
            for pp in pair:
                i,p,pR = pp
                if (p==None):
                    raise Exception("error: some pairs cannot be found in Crystal_symmetry_mat")
        basis_list = copy.deepcopy(self.basis_list)
        orb_len = {'s':1, 'p':3, 'd':5, 'f':7}
        Rmat = np.zeros((N,N),dtype=complex)
        for p in pair:
            p1,p2,RR2 = p
            kR = np.dot(k,RR2)
            m0,n0 = self.At_range[p1][0],self.At_range[p2][0]
            lng = self.Total_NumOrbs[p1]
            i = 0
            while(i<lng):
                orb = basis_list[i+m0].split('-')[-1][0]
                ol = orb_len[orb]
                if (orb=='s'):
                    Rmat[i+m0:i+m0+ol,i+n0:i+n0+ol] = R0
                elif (orb=='p'):
                    Rmat[i+m0:i+m0+ol,i+n0:i+n0+ol] = R1
                elif (orb=='d'):
                    Rmat[i+m0:i+m0+ol,i+n0:i+n0+ol] = R2
                elif (orb=='f'):
                    Rmat[i+m0:i+m0+ol,i+n0:i+n0+ol] = R3
                else:
                    raise Exception("error in basis_list of Crystal_symmetry_mat")
                i += ol
            Rmat[:,n0:n0+lng] *= np.exp(-1.j*kR)
        SP = self.SpinP_switch
        if (SU2 is None):
            nvec,theta = axis_angle_crystal(R)
            SU2 = SU2_mat(nvec,theta)
        if (SP==1):
            if (spin == False):
                Rmat = np.kron(SU2,Rmat)
        elif (SP==3):
            Rmat = np.kron(SU2,Rmat)
        if (overlap):
            Sk = self.Sk_kvec(k, spin=spin)
            mat = np.dot(Sk,Rmat)  
        else:
            mat = Rmat
        return mat


    def Band(self, k1, k2, n, eigvals = None, spin=False):
        path = kpath(k1,k2,n)
        egvs = eigvals
        k = 0.0
        klist = []
        Elists = []
        kbefore = path[0]
        for kvec in path:
            k += np.linalg.norm(kvec-kbefore)
            klist.append(k)
            exp_vec = self.exp_vec(kvec)
            h = self.Hk(exp_vec,spin=spin)
            s = self.Sk(exp_vec,spin=spin)
            w = scipylinalg.eigh(h, s, eigvals_only = True, \
                overwrite_a=True,overwrite_b=True,turbo=False, \
                eigvals=egvs)
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
                 fname=None, c1='b', c2='r', figsize=None):
        PlotBand(Band_list, kticks_label= kticks_label,
                 yrange=yrange, shift=shift, eV=eV, EF=EF, highlight=highlight,
                 save=save, fname=fname, c1=c1, c2=c2, figsize=figsize)

    def Orbital_weight(self, k, band_idx, orbitals):
        expvec = self.exp_vec(k)
        h = self.Hk(expvec)
        s = self.Sk(expvec)
        w,v = scipylinalg.eigh(h, s, overwrite_a=True,turbo=False)
        result = []
        for n in band_idx:
            for i in orbitals:
                c = v[:,n]
                cs = np.dot(c.conjugate(),s[:,i])
                result.append([n,i,cs*c[i]])
        return result

    def Band_weight(self, k1, k2, n, orbitals, eigvals = None):
        path = kpath(k1,k2,n)
        egvs = eigvals
        k = 0.0
        klist = []
        Elists = []
        Wlists = []
        kbefore = path[0]
        for kvec in path:
            k += np.linalg.norm(kvec-kbefore)
            klist.append(k)
            exp_vec = self.exp_vec(kvec)
            h = self.Hk(exp_vec)
            s = self.Sk(exp_vec)
            w,v = scipylinalg.eigh(h, s, \
                overwrite_a=True, turbo=False, eigvals=egvs)
            Elists.append(w)

            weights = []
            cs = np.dot(v.conjugate().transpose(),s)
            for m in range(v.shape[1]):
                wei = 0.
                vm = v[:,m]
                for i in orbitals:
                    wei += cs[m,i]*vm[i]
                weights.append(wei.real)
            Wlists.append(weights)

            kbefore = kvec
        E = np.transpose(np.array(Elists))
        W = np.transpose(np.array(Wlists))
        out = [np.array(klist),E,W]
        #out[0]:k-axis value from 0, out[1]:energy eigenvalues
        #out[2]:weight of selected orbitals
        return out

    def PlotBand_weight(self, Band_list, kticks_label=None, yrange=None,
                 eV=False, EF=None, save=False):
        if (save):
            #del(sys.modules['matplotlib'])
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cut = 0.9
        fig = plt.figure()
        ax = plt.subplot()
        kticks = []
        kstart = 0.
        for Bands in Band_list:
            klist = Bands[0]+kstart
            E = Bands[1]
            W = Bands[2]
            kticks.append(klist[0])
            for i in range(E.shape[0]):
                e = E[i,:]
                if (eV == True): e *= 27.2113860217
                col = []
                for w in W[i,:]:
                    if (w >= cut): rw = 1.
                    else: rw = abs(w/cut)
                    col.append([rw,0.,1.-rw])
                plt.scatter(klist,e,s=0.2,edgecolors=None,c=col)
            kstart = klist[-1]
        kticks.append(kstart)
        ax.set_xticks(kticks, minor=False)
        ax.xaxis.grid(True, which='major')
        if (kticks_label != None):
            ax.set_xticklabels(kticks_label)
        if (EF != None):
            if (eV == True): EF *= 27.2113860217
            plt.plot([0.,kstart],[EF,EF],lw=0.25,color='gray',ls='--')
        plt.xlim(0,kstart)
        if (yrange != None):
            plt.ylim([yrange[0],yrange[1]])
        if(save): plt.savefig('./pymx_band_weight.png')
        else: plt.show()

    def Zak_phase(self, k0, G, n, bands, spin=False):
        dk = G/float(n-1)
        SP = self.SpinP_switch
        if ((SP==0)or ((SP==1) and bool(spin))):
            edkt = self.phase_tau(dk,mat_size=1).conjugate()
        else:
            edkt = self.phase_tau(dk,mat_size=2).conjugate()
        egvs = (bands[0],bands[1])
        PI_det = 1.+0.j
        klist = kpath(k0,k0+G,n)
        for i in range(n):
            k = klist[i]
            if (i==0):
                exp_vec = self.exp_vec(k)
                h = self.Hk(exp_vec,spin=spin)
                s = self.Sk(exp_vec,spin=spin)
                w,v = scipylinalg.eigh(h, s, eigvals=egvs, \
                    overwrite_a=True,overwrite_b=True,turbo=False)
                Ckn = v.copy()
                Ck2 = v.copy()
                xn = self.Xk(exp_vec,spin=spin)
                yn = self.Yk(exp_vec,spin=spin)
                zn = self.Zk(exp_vec,spin=spin)
            elif (i==n-1):
                Ck1h = Ck2.conjugate().transpose()
                dkX = dk[0]*xn + dk[1]*yn + dk[2]*zn
                MAT = s - 1.j*dkX
                unum_mat = np.linalg.multi_dot([Ck1h,edkt,MAT,Ckn])
                PI_det *= np.linalg.det(unum_mat) 
            else:
                Ck1h = Ck2.conjugate().transpose()
                exp_vec = self.exp_vec(k)
                h = self.Hk(exp_vec,spin=spin)
                s = self.Sk(exp_vec,spin=spin)
                x = self.Xk(exp_vec,spin=spin)
                y = self.Yk(exp_vec,spin=spin)
                z = self.Zk(exp_vec,spin=spin)
                s_in = s.copy()
                w,Ck2 = scipylinalg.eigh(h, s_in, eigvals=egvs, \
                    overwrite_a=True,overwrite_b=True,turbo=False)
                dkX = dk[0]*x + dk[1]*y + dk[2]*z
                MAT = s - 1.j*dkX
                unum_mat = np.linalg.multi_dot([Ck1h,edkt,MAT,Ck2])
                PI_det *= np.linalg.det(unum_mat) 
        return -1.*np.imag(np.log(PI_det))

    def Berry_phase_from_list(self, klist, Clist, Slist, Xlist, Ylist, Zlist,
                            mat_size=1):
        PI_det = 1.+0.j
        klist.append(klist[0])
        Clist.append(Clist[0])
        Slist.append(Slist[0])
        Xlist.append(Xlist[0])
        Ylist.append(Ylist[0])
        Zlist.append(Zlist[0])
        ms = mat_size
        for i in range(len(klist)-1):
            dk = klist[i+1] - klist[i]
            edkt = self.phase_tau(dk,mat_size=ms).conjugate()
            Ck1h = Clist[i].conjugate().transpose()
            Ck2 = Clist[i+1]
            dkX = dk[0]*Xlist[i+1] + \
                  dk[1]*Ylist[i+1] + dk[2]*Zlist[i+1]
            MAT = Slist[i+1] - 1.j*dkX
            unum_mat = np.linalg.multi_dot([Ck1h,edkt,MAT,Ck2])
            PI_det *= np.linalg.det(unum_mat) 
        return -1.*np.imag(np.log(PI_det))

    def Zak_phase2(self, k0, G, n, bands, spin=False):
        dk = G/float(n-1)
        N = self.mat_size
        SP = self.SpinP_switch

        dktau = np.zeros((N,N),dtype=float)
        for i in range(N):
            t = self.tau[self.AtOrb[i,0]] 
            dktau[i,i] = np.dot(dk,t)
        if not((SP==0)or ((SP==1) and bool(spin))):
            dktau = np.kron(np.identity(2,dtype=complex),dktau)

        egvs = (bands[0],bands[1])
        sum_intra = 0.+0.j
        sum_dSk = 0.+0.j
        PI_det = 1.+0.j
        klist = kpath(k0,k0+G,n)
        for i in range(n-1):
            k = klist[i]
            exp_vec = self.exp_vec(k)
            h = self.Hk(exp_vec,spin=spin)
            s = self.Sk(exp_vec,spin=spin)
            x = self.Xk(exp_vec,spin=spin)
            y = self.Yk(exp_vec,spin=spin)
            z = self.Zk(exp_vec,spin=spin)
            sx = self.dSk_dki(exp_vec,0,spin=spin)
            sy = self.dSk_dki(exp_vec,1,spin=spin)
            sz = self.dSk_dki(exp_vec,2,spin=spin)
            s_in = s.copy()
            w,v = scipylinalg.eigh(h, s_in, eigvals=egvs, \
                overwrite_a=True,overwrite_b=True,turbo=False)
            Ck1 = v.copy()
            Ck1h = Ck1.conjugate().transpose()
            #intra
            dkX = dk[0]*x + dk[1]*y + dk[2]*z
            tauS = np.dot(dktau,s)
            idkdkS = 1.j*(dk[0]*sx + dk[1]*sy + dk[2]*sz)
            MAT1 = dkX + tauS + idkdkS
            sum_intra += np.trace(np.linalg.multi_dot([Ck1h,MAT1,Ck1]))
            #inter
            if (i==0):
                Ckn = v.copy()
            else:
                MAT2 = s2
                cncm_mat = np.linalg.multi_dot([Ck0h,MAT2,Ck1])
                PI_det *= np.linalg.det(cncm_mat) 
            exp_vec2 = self.exp_vec(k+dk/2.)
            s2 = self.Sk(exp_vec2,spin=spin)
            Ck0h = Ck1h
        MAT2 = s2
        cncm_mat = np.linalg.multi_dot([Ck0h,MAT2,Ckn])
        PI_det *= np.linalg.det(cncm_mat) 

        intraZak = sum_intra.real
        interZak =  -1.*np.imag(np.log(PI_det))
        return [intraZak, interZak]

    def Wilson_loop(self, k0, G, n, bands, spin=False):
        nband = bands[1]-bands[0]+1
        dk = G/float(n-1)
        SP = self.SpinP_switch
        if ((SP==0)or ((SP==1) and bool(spin))):
            edkt = self.phase_tau(dk,mat_size=1).conjugate()
        else:
            edkt = self.phase_tau(dk,mat_size=2).conjugate()
        egvs = (bands[0],bands[1])
        klist = kpath(k0,k0+G,n)
        PI_mat = np.identity(nband,dtype=complex)
        eigvecs = []
        Slist,Xlist,Ylist,Zlist = [],[],[],[]
        for k in klist[:-1]:
            exp_vec = self.exp_vec(k)
            h = self.Hk(exp_vec,spin=spin)
            s = self.Sk(exp_vec,spin=spin)
            x = self.Xk(exp_vec,spin=spin)
            y = self.Yk(exp_vec,spin=spin)
            z = self.Zk(exp_vec,spin=spin)
            Slist.append(s)
            Xlist.append(x)
            Ylist.append(y)
            Zlist.append(z)
            w,v = scipylinalg.eigh(h, s, eigvals=egvs, \
                overwrite_a=True,overwrite_b=True,turbo=False)
            eigvecs.append(v)
        eigvecs.append(eigvecs[0])
        Slist.append(Slist[0])
        Xlist.append(Xlist[0])
        Ylist.append(Ylist[0])
        Zlist.append(Zlist[0])
        for ii in range(n-1):
            Ck1h = eigvecs[ii].conjugate().transpose()
            Ck2 = eigvecs[ii+1]
            dkX = dk[0]*Xlist[ii+1] + \
                  dk[1]*Ylist[ii+1] + dk[2]*Zlist[ii+1]
            MAT = Slist[ii+1] - 1.j*dkX
            unum_mat = np.linalg.multi_dot([Ck1h,edkt,MAT,Ck2])
            PI_mat = np.dot(PI_mat,unum_mat)
        return PI_mat

    def WCC(self, G, k0, k1, nG, nk, bands, spin=False):
        nband = bands[1]-bands[0]+1
        WCCout = np.empty((nband+1,nk),dtype=float)
        kaxis = kpath(k0,k1,nk)
        kaxis_num = np.linspace(0.,np.linalg.norm(k1-k0),nk)
        WCCout[0,:] = kaxis_num
        dk = G/float(nG-1)
        SP = self.SpinP_switch
        if ((SP==0)or ((SP==1) and bool(spin))):
            edkt = self.phase_tau(dk,mat_size=1).conjugate()
        else:
            edkt = self.phase_tau(dk,mat_size=2).conjugate()
        egvs = (bands[0],bands[1])
        for i,k00 in enumerate(kaxis):
            PI_mat = np.identity(nband,dtype=complex)
            klist = kpath(k00,k00+G,nG)
            eigvecs = []
            Slist,Xlist,Ylist,Zlist = [],[],[],[]
            for k in klist[:-1]:
                exp_vec = self.exp_vec(k)
                h = self.Hk(exp_vec,spin=spin)
                s = self.Sk(exp_vec,spin=spin)
                x = self.Xk(exp_vec,spin=spin)
                y = self.Yk(exp_vec,spin=spin)
                z = self.Zk(exp_vec,spin=spin)
                Slist.append(s)
                Xlist.append(x)
                Ylist.append(y)
                Zlist.append(z)
                w,v = scipylinalg.eigh(h, s, eigvals=egvs, \
                    overwrite_a=True,overwrite_b=True,turbo=False)
                eigvecs.append(v)
            eigvecs.append(eigvecs[0])
            Slist.append(Slist[0])
            Xlist.append(Xlist[0])
            Ylist.append(Ylist[0])
            Zlist.append(Zlist[0])
            for ii in range(nG-1):
                Ck1h = eigvecs[ii].conjugate().transpose()
                Ck2 = eigvecs[ii+1]
                dkX = dk[0]*Xlist[ii+1] + \
                      dk[1]*Ylist[ii+1] + dk[2]*Zlist[ii+1]
                MAT = Slist[ii+1] - 1.j*dkX
                unum_mat = np.linalg.multi_dot([Ck1h,edkt,MAT,Ck2])
                PI_mat = np.dot(PI_mat,unum_mat)
            wcc = np.angle(np.linalg.eigvals(PI_mat))
            wcc.sort()
            WCCout[1:,i] = wcc
        return WCCout

    def PlotWCC(self, WCC, save=False, fname=None, figsize=None):
        PlotWCC(WCC, save=save, fname=fname, figsize=figsize)

    def Spintexture(self, k, bands, overlap=True):
        N = bands[1] - bands[0] + 1
        ms = self.mat_size
        expvec = self.exp_vec(k)
        h = self.Hk(expvec)
        s0 = self.Sk(expvec, small=True)
        if (overlap):
            ss = s0.copy()
        else:
            ss = np.identity(ms,dtype=float)
        Sx = np.kron(s_x,ss)
        Sy = np.kron(s_y,ss)
        Sz = np.kron(s_z,ss)
        zero = np.zeros((ms,ms))
        s = np.block([[s0,zero],[zero,s0]])
        w,V = scipylinalg.eigh(h, s, eigvals=bands, \
              overwrite_a=True,overwrite_b=True,turbo=False)
        Vh = V.conjugate().transpose()
        spin = []
        for i in range(N):
            v = V[:,i]
            vh = Vh[i,:]
            SX = np.linalg.multi_dot([vh,Sx,v]).item(0).real 
            SY = np.linalg.multi_dot([vh,Sy,v]).item(0).real
            SZ = np.linalg.multi_dot([vh,Sz,v]).item(0).real
            spin.append(np.array([SX,SY,SZ]))
        return [w,spin]

    def Orbital_angular_momentum_onsite_mat(self, spin=False):
        N = self.mat_size
        matx = np.zeros((N,N),dtype=complex)
        maty = np.zeros((N,N),dtype=complex)
        matz = np.zeros((N,N),dtype=complex)
        basis_list = copy.deepcopy(self.basis_list)
        orb_len = {'s':1, 'p':3, 'd':5, 'f':7}
        i = 0
        while(i<N):
            orb = basis_list.pop(0).split('-')[-1][0]
            ol = orb_len[orb]
            if (orb=='s'):
                pass
            elif (orb=='p'):
                matx[i:i+ol,i:i+ol] = Lpxr
                maty[i:i+ol,i:i+ol] = Lpyr
                matz[i:i+ol,i:i+ol] = Lpzr
            elif (orb=='d'):
                matx[i:i+ol,i:i+ol] = Ldxr
                maty[i:i+ol,i:i+ol] = Ldyr
                matz[i:i+ol,i:i+ol] = Ldzr
            elif (orb=='f'):
                matx[i:i+ol,i:i+ol] = Lfxr
                maty[i:i+ol,i:i+ol] = Lfyr
                matz[i:i+ol,i:i+ol] = Lfzr
            else:
                raise Exception("error in basis_list of Orbital_ang_mom_onsite_mat")
            i += 1
            for j in range(ol-1):
                basis_list.pop(0)
                i += 1
        SP = self.SpinP_switch
        if (SP==1):
            if (spin == False):
                iden = np.identity(2,dtype=complex)
                matx = np.kron(iden,matx)
                maty = np.kron(iden,maty)
                matz = np.kron(iden,matz)
        elif (SP==3):
            iden = np.identity(2,dtype=complex)
            matx = np.kron(iden,matx)
            maty = np.kron(iden,maty)
            matz = np.kron(iden,matz)
        return [matx,maty,matz]

    def velocity_mat(self, k, bands, energy=False):
        expvec = self.exp_vec(k)
        H = self.Hk(expvec)
        S = self.Sk(expvec)
        dHdkx = self.dHk_dki(expvec,0)
        dHdky = self.dHk_dki(expvec,1)
        dHdkz = self.dHk_dki(expvec,2)
        dSdkx = self.dSk_dki(expvec,0)
        dSdky = self.dSk_dki(expvec,1)
        dSdkz = self.dSk_dki(expvec,2)
        M = self.mat_size
        tx = np.zeros((M,M),dtype=float)
        ty = np.zeros((M,M),dtype=float)
        tz = np.zeros((M,M),dtype=float)
        for i in range(M):
            t = self.tau[self.AtOrb[i,0]] 
            tx[i,i] = t[0]
            ty[i,i] = t[1]
            tz[i,i] = t[2]
        SP = self.SpinP_switch
        if (SP==1 or SP==3):
            zero = np.zeros((M,M),dtype=float) 
            tx = np.block([[tx,zero],[zero,tx]])
            ty = np.block([[ty,zero],[zero,ty]])
            tz = np.block([[tz,zero],[zero,tz]])
        X = self.Xk(expvec) + np.dot(tx,S)
        Y = self.Yk(expvec) + np.dot(ty,S)
        Z = self.Zk(expvec) + np.dot(tz,S)
        w,v = scipylinalg.eigh(H, S, eigvals=bands, \
            overwrite_a=True,overwrite_b=False,turbo=False)
        vh = v.transpose().conjugate()
        ev = w*v
        evh = ev.transpose().conjugate()
        vx1 = np.linalg.multi_dot([vh,dHdkx,v])
        vy1 = np.linalg.multi_dot([vh,dHdky,v])
        vz1 = np.linalg.multi_dot([vh,dHdkz,v])
        vx2 = np.linalg.multi_dot([evh,1.j*X-dSdkx,v])
        vy2 = np.linalg.multi_dot([evh,1.j*Y-dSdky,v])
        vz2 = np.linalg.multi_dot([evh,1.j*Z-dSdkz,v])
        vx3 = np.linalg.multi_dot([vh,-1.j*X,ev])
        vy3 = np.linalg.multi_dot([vh,-1.j*Y,ev])
        vz3 = np.linalg.multi_dot([vh,-1.j*Z,ev])
        vx = vx1+vx2+vx3
        vy = vy1+vy2+vy3
        vz = vz1+vz2+vz3
        if(energy):
            return [vx,vy,vz,w]
        else:
            return [vx,vy,vz]

    def BerryCurvature(self, k):
        vx,vy,vz,w = self.velocity_mat(k, None, energy=True)
        e1 = np.array([w]*len(w))
        e2 = e1.copy().transpose()
        E = 1./(e1 - e2 + np.identity(len(w))) - np.identity(len(w))
        X = 2.*np.dot(vy*E,vz*E).imag.diagonal()
        Y = 2.*np.dot(vz*E,vx*E).imag.diagonal()
        Z = 2.*np.dot(vx*E,vy*E).imag.diagonal() 
        return [X,Y,Z]
 

