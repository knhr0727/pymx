
# coding start from 2018/07/03 at ISS2018
# pyread_scfout for openmx3.9 version (major coding 2021/01/26)

import sys
import struct
import copy
import numpy as np

#
# Endianness control 
#
# If you run pymx in the same machine that produce your scfout file,
# you don't have to consider the endianness issue.
# If you know the endianness of your scfout binary file and it does not
# match your machine, please control the endianness by the following
# parameter "edn". 
# edn = '<' : scfout written in little endian / '>' : big endian
edn = ''   

class ReadScfout:
    '''read_scfout'''

    def __init__(self):
        pass

    #read_scfout for openmx3.8
    def read_scfout38(self, scfout_file):
        f = open(scfout_file,'rb')
        
        MAX_LINE_SIZE = 256
        sizeof_int = 4 #i
        sizeof_double = 8 #d
        sizeof_char = 1 #c
        
        def fmtstr(i,f): #integer and format string
             return edn+str(i)+f
        
        #  /****************************************************
        # int atomnun;
        #  the number of total atoms
        # int SpinP_switch;
        #  0: non-spin polarized 
        #  1: spin polarized
        # int Catomnun;
        #  the number of atoms in the central region
        # int Latomnun;
        #  the number of atoms in the left lead
        # int Ratomnun;
        #  the number of atoms in the left lead
        # int TCpyCell;
        #  the total number of periodic cells
        #
        #     grobal index of atom runs
        #     Catomnum -> Catomnum + Latomnum -> Catomnum + Latomnum + Ratomnum
        #  ****************************************************/
        i_vec = struct.unpack(edn+'6i',f.read(sizeof_int*6))
        atomnum      = i_vec[0]
        SpinP_switch = i_vec[1]
        Catomnum =     i_vec[2]
        Latomnum =     i_vec[3]
        Ratomnum =     i_vec[4]
        TCpyCell =     i_vec[5]
        
        self.atomnum      = atomnum     
        self.SpinP_switch = SpinP_switch
        self.Catomnum =     Catomnum 
        self.Latomnum =     Latomnum 
        self.Ratomnum =     Ratomnum 
        self.TCpyCell =     TCpyCell 
        if (self.SpinP_switch>=4):
            print("Error: version mismatch. ver=='3.8' now")
            quit()

        #  /****************************************************
        #    double atv[TCpyCell+1][4];
        #  x,y,and z-components of translation vector of  
        #  periodically copied cells
        #  ****************************************************/
        atv = []
        for Rn in range(TCpyCell+1):
            un_pack = struct.unpack(edn+'4d',f.read(sizeof_double*4))
            atv.append(list(un_pack))
        atv = np.array(atv)
        self.atv = atv
        
        #  /****************************************************
        #    int atv_ijk[TCpyCell+1][4];
        #  i,j,and j number of periodically copied cells
        #  ****************************************************/
        atv_ijk = []
        for Rn in range(TCpyCell+1):
            un_pack = struct.unpack(edn+'4i',f.read(sizeof_int*4))
            atv_ijk.append(list(un_pack))
        atv_ijk = np.array(atv_ijk)
        self.atv_ijk = atv_ijk     
        
        #  /****************************************************
        #    int Total_NumOrbs[atomnum+1];
        # the number of atomic orbitals in each atom
        #    int FNAN[atomnum+1];
        # the number of first neighboring atoms of each atom
        #  ****************************************************/
        fmt = fmtstr(atomnum,'i')
        p_vec = struct.unpack(fmt,f.read(sizeof_int*atomnum))
        Total_NumOrbs = [1] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            Total_NumOrbs.append(p_vec[ct_AN-1])
        Total_NumOrbs = np.array(Total_NumOrbs)
        del(p_vec)
        self.Total_NumOrbs = Total_NumOrbs     
        
        fmt = fmtstr(atomnum,'i')
        p_vec = struct.unpack(fmt,f.read(sizeof_int*atomnum))
        FNAN = [0] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            FNAN.append(p_vec[ct_AN-1])
        FNAN = np.array(FNAN)
        del(p_vec)
        self.FNAN = FNAN     
        
        #  /****************************************************
        #    int natn[atomnum+1][FNAN[ct_AN]+1];
        #  grobal index of neighboring atoms of an atom ct_AN
        #    int ncn[atomnum+1][FNAN[ct_AN]+1];
        #  grobal index for cell of neighboring atoms
        #  of an atom ct_AN
        #  note: natn[i][0] and ncn[i][0] mean the global index of atom i itself
        #  ****************************************************/
        natn = [0] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            fmt = fmtstr(FNAN[ct_AN]+1,'i')
            un_pack = struct.unpack(fmt,f.read(sizeof_int*(FNAN[ct_AN]+1)))
            natn.append(list(un_pack))
        # stored in the form of list
        self.natn = natn     
        
        ncn = [0] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            fmt = fmtstr(FNAN[ct_AN]+1,'i')
            un_pack = struct.unpack(fmt,f.read(sizeof_int*(FNAN[ct_AN]+1)))
            ncn.append(list(un_pack))
        # stored in the form of list
        self.ncn = ncn     
        
        #  /****************************************************
        #  double  tv[4][4]:
        #    unit cell vectors in Bohr
        #  ****************************************************/
        tv = [[0.,0.,0.,0.]] #dummy
        for i in range(3):
            un_pack = struct.unpack(edn+'4d',f.read(sizeof_double*4))
            tv.append(list(un_pack))
        tv = np.array(tv)
        self.tv = tv     
        
        #  /****************************************************
        #  double  rtv[4][4]:
        #  reciprocal unit cell vectors in Bohr^{-1}
        #  note:
        #   tv_i \dot rtv_j = 2PI * Kronecker's delta_{ij}
        #  ****************************************************/
        rtv = [[0.,0.,0.,0.]] #dummy
        for i in range(3):
            un_pack = struct.unpack(edn+'4d',f.read(sizeof_double*4))
            rtv.append(list(un_pack))
        rtv = np.array(rtv)
        self.rtv = rtv     
        
        #  /****************************************************
        # double Gxyz[atomnum+1][60];
        #  atomic coordinates in Bohr
        #  ****************************************************/
        Gxyz = [[0.]*60] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            un_pack = struct.unpack(edn+'4d',f.read(sizeof_double*4))
            Gxyz.append(list(un_pack)+[0.]*56)
        Gxyz = np.array(Gxyz)
        self.Gxyz = Gxyz     
        
        #  /****************************************************
        #    Kohn-Sham Hamiltonian
        #
        #     double Hks[SpinP_switch+1]
        #               [atomnum+1]
        #               [FNAN[ct_AN]+1]
        #               [Total_NumOrbs[ct_AN]]
        #               [Total_NumOrbs[h_AN]];
        #
        ########################################################
        #
        #      Hks[spin][ct_AN][h_AN][i][j]
        #
        #      spin:  spin=0, up
        #             spin=1, down
        #
        #      ct_AN: global index of atoms
        #      h_AN   local index of neighbouring atoms for the atom ct_AN
        #      i:     orbital index in the atom ct_AN
        #      j:     orbital index in the atom h_AN
        #
        #   NOTE: 
        #
        #      For instance, if the basis specification of the atom ct_AN is s2p2,
        #      then the obital index runs in order of
        #                    s, s', px, py, pz, px', py', pz'
        #
        #      Transformation of the local index h_AN to the grobal index Gh_AN
        #      is made as
        #
        #                       Gh_AN = natn[ct_AN][h_AN];
        #
        #      Also, the cell index is given by
        #
        #                       Rn = ncn[ct_AN][h_AN];
        #      
        #      Each component l, m, or n (Rn = l*a + m*b + n*c) are given by
        #   
        #                       l = atv_ijk[Rn][1];
        #                       m = atv_ijk[Rn][2];
        #                       n = atv_ijk[Rn][3];
        #  ****************************************************/
        Hks = [] #H(ia)(jb)
        for spin in range(SpinP_switch+1):
            list_spin = [0]  #dummy
            for ct_AN in range(atomnum+1)[1:]: #i
                list_ctAN = []
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN]+1): #j
                    list_hAN = []
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1): #a
                        fmt = fmtstr(TNO2,'d')
                        un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                        list_hAN.append(list(un_pack))
                    list_ctAN.append(np.array(list_hAN))
                list_spin.append(list_ctAN)
            Hks.append(list_spin)
        self.Hks = Hks     
        
        #  /****************************************************
        #  iHks:
        #  imaginary Kohn-Sham matrix elements of basis orbitals
        #  for alpha-alpha, beta-beta, and alpha-beta spin matrices
        #  of which contributions come from spin-orbit coupling 
        #  and Hubbard U effective potential.
        #  ****************************************************/
        iHks = [] #iH(ia)(jb)
        for spin in range(3):
            list_spin = [0]  #dummy
            for ct_AN in range(atomnum+1)[1:]: #i
                list_ctAN = []
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN]+1): #j
                    list_hAN = []
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1): #a
                        if (SpinP_switch==3):
                            fmt = fmtstr(TNO2,'d')
                            un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                            list_hAN.append(list(un_pack))
                        else:
                            list_hAN.append([0.0]*TNO2)
                    list_ctAN.append(np.array(list_hAN))
                list_spin.append(list_ctAN)
            iHks.append(list_spin)
        self.iHks = iHks     
        
        #  /****************************************************
        #    Overlap matrix
        #
        #     double OLP[atomnum+1]
        #               [FNAN[ct_AN]+1]
        #               [Total_NumOrbs[ct_AN]]
        #               [Total_NumOrbs[h_AN]]; 
        #  ****************************************************/
        OLP = [0] #OLP(ia)(jb) + dummy
        for ct_AN in range(atomnum+1)[1:]: #i
            list_ctAN = []
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN]+1): #j
                list_hAN = []
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                for i in range(TNO1): #a
                    fmt = fmtstr(TNO2,'d')
                    un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                    list_hAN.append(list(un_pack))
                list_ctAN.append(np.array(list_hAN))
            OLP.append(list_ctAN)
        self.OLP = OLP     
        
        #  /****************************************************
        #    Overlap matrix with position operator x, y, z
        #
        #     dooble OLPpox,y,z
        #                 [atomnum+1]
        #                 [FNAN[ct_AN]+1]
        #                 [Total_NumOrbs[ct_AN]]
        #                 [Total_NumOrbs[h_AN]]; 
        #  ****************************************************/
        OLPpox = [0] #OLPpox(ia)(jb) + dummy
        for ct_AN in range(atomnum+1)[1:]: #i
            list_ctAN = []
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN]+1): #j
                list_hAN = []
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                for i in range(TNO1): #a
                    fmt = fmtstr(TNO2,'d')
                    un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                    list_hAN.append(list(un_pack))
                list_ctAN.append(np.array(list_hAN))
            OLPpox.append(list_ctAN)
        self.OLPpox = OLPpox     
        
        OLPpoy = [0] #OLPpoy(ia)(jb) + dummy
        for ct_AN in range(atomnum+1)[1:]: #i
            list_ctAN = []
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN]+1): #j
                list_hAN = []
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                for i in range(TNO1): #a
                    fmt = fmtstr(TNO2,'d')
                    un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                    list_hAN.append(list(un_pack))
                list_ctAN.append(np.array(list_hAN))
            OLPpoy.append(list_ctAN)
        self.OLPpoy = OLPpoy     
        
        OLPpoz = [0] #OLPpoz(ia)(jb) + dummy
        for ct_AN in range(atomnum+1)[1:]: #i
            list_ctAN = []
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN]+1): #j
                list_hAN = []
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                for i in range(TNO1): #a
                    fmt = fmtstr(TNO2,'d')
                    un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                    list_hAN.append(list(un_pack))
                list_ctAN.append(np.array(list_hAN))
            OLPpoz.append(list_ctAN)
        self.OLPpoz = OLPpoz     
        
        #  /****************************************************
        #    Density matrix
        #
        #     dooble DM[SpinP_switch+1]
        #              [atomnum+1]
        #              [FNAN[ct_AN]+1]
        #              [Total_NumOrbs[ct_AN]]
        #              [Total_NumOrbs[h_AN]];
        #  ****************************************************/
        DM = [] #DM(ia)(jb)
        for spin in range(SpinP_switch+1):
            list_spin = [0]  #dummy
            for ct_AN in range(atomnum+1)[1:]: #i
                list_ctAN = []
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN]+1): #j
                    list_hAN = []
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1): #a
                        fmt = fmtstr(TNO2,'d')
                        un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                        list_hAN.append(list(un_pack))
                    list_ctAN.append(np.array(list_hAN))
                list_spin.append(list_ctAN)
            DM.append(list_spin)
        self.DM = DM     
        
        #  /****************************************************
        # int Solver;
        #  method for solving eigenvalue problem
        #  ****************************************************/
        i_vec = struct.unpack(edn+'1i',f.read(sizeof_int*1))
        Solver = i_vec[0]
        self.Solver = Solver
        
        #  /****************************************************
        # double ChemP;
        #  chemical potential
        # double E_Temp;
        #  electronic temperature
        # double dipole_moment_core[4];
        # double dipole_moment_background[4];
        # int Valence_Electrons;
        #  total number of valence electrons
        # double Total_SpinS;
        #  total value of Spin (2*Total_SpinS = muB)
        #  ****************************************************/
        d_vec = struct.unpack(edn+'10d',f.read(sizeof_double*10))
        ChemP  = d_vec[0]
        E_Temp = d_vec[1]
        dipole_moment_core = np.array([0.,d_vec[2],d_vec[3],d_vec[4]])
        dipole_moment_background = np.array([0.,d_vec[5],d_vec[6],d_vec[7]])
        Valence_Electrons = d_vec[8]
        Total_SpinS = d_vec[9]
        
        self.ChemP  = ChemP 
        self.E_Temp = E_Temp
        self.dipole_moment_core = dipole_moment_core
        self.dipole_moment_background = dipole_moment_background
        self.Valence_Electrons = Valence_Electrons
        self.Total_SpinS =       Total_SpinS 
        
        #  /****************************************************
        #      input file 
        #  ****************************************************/
        i_vec = struct.unpack(edn+'1i',f.read(sizeof_int*1))
        num_lines = i_vec[0]
        temporal_input = []
        for i in range(num_lines+1)[1:]:
            fmt = fmtstr(MAX_LINE_SIZE,'c')
            un_pack = struct.unpack(fmt,f.read(sizeof_char*MAX_LINE_SIZE))
            l = list(un_pack)
            if (sys.version_info[0]==3):
                new_line = ""
            else:
                new_line = "".decode('utf-8')
            nTF = True
            while(nTF):
                ch =  l.pop(0)
                ch = ch.decode('utf-8','ignore')
                if (ch=='\n'):
                    nTF = False
                new_line += ch
            temporal_input.append(new_line)
        self.temporal_input = temporal_input     

        f.close()

    #read_scfout for openmx3.9
    def read_scfout39(self, scfout_file):
        f = open(scfout_file,'rb')
        
        MAX_LINE_SIZE = 256
        sizeof_int = 4 #i
        sizeof_double = 8 #d
        sizeof_char = 1 #c
        
        def fmtstr(i,f): #integer and format string
             return edn+str(i)+f

        #For version 3.9
        SCFOUT_VERSION = 3
        LATEST_VERSION = 3
        
        #  /****************************************************
        # int atomnun;
        #  the number of total atoms
        # int SpinP_switch;
        #  0: non-spin polarized 
        #  1: spin polarized
        # int Catomnun;
        #  the number of atoms in the central region
        # int Latomnun;
        #  the number of atoms in the left lead
        # int Ratomnun;
        #  the number of atoms in the left lead
        # int TCpyCell;
        #  the total number of periodic cells
        #
        #     grobal index of atom runs
        #     Catomnum -> Catomnum + Latomnum -> Catomnum + Latomnum + Ratomnum
        #  ****************************************************/
        i_vec = struct.unpack(edn+'6i',f.read(sizeof_int*6))

        atomnum      = i_vec[0]
        SpinP_switch = i_vec[1]%4
        version  =     i_vec[1]/4
        Catomnum =     i_vec[2]
        Latomnum =     i_vec[3]
        Ratomnum =     i_vec[4]
        TCpyCell =     i_vec[5]
        
        self.atomnum      = atomnum     
        self.SpinP_switch = SpinP_switch
        self.version  =     version
        self.Catomnum =     Catomnum 
        self.Latomnum =     Latomnum 
        self.Ratomnum =     Ratomnum 
        self.TCpyCell =     TCpyCell 

        if (version==0):
            openmxVersion="3.7, 3.8 or an older distribution"
        elif (version==1):
            openmxVersion="3.7.x (for development of HWC)"
        elif (version==2):
            openmxVersion="3.7.x (for development of HWF)"
        elif (version==3):
            openmxVersion="3.9"
        else:
            openmxVersion=" INVALID VALUE"
        
        if (version!=SCFOUT_VERSION):
            print("The file format of the SCFOUT file:   %d ("%version +openmxVersion+")")
            print("The vesion is not supported by the current read_scfout")
            print("Or the endianness mismatch occurs")
            quit()
      
        version_text = ""
        version_text +="***\n"
        version_text +="The file format of the SCFOUT file:  %d\n"%version
        version_text +="And it supports the following functions:\n"
        version_text +="- jx\n"
        version_text +="- polB\n"
        version_text +="- kSpin\n"
        version_text +="- Z2FH\n"
        version_text +="- calB\n"
        version_text +="***\n"
        self.version_text = version_text
      
        #  /****************************************************
        #    order_max (added by N. Yamaguchi for HWC)
        #   ****************************************************/
        i_vec = struct.unpack(edn+'1i',f.read(sizeof_int*1))
        order_max = list(i_vec)[0]
        self.order_max = order_max

        #  /****************************************************
        #    double atv[TCpyCell+1][4];
        #  x,y,and z-components of translation vector of  
        #  periodically copied cells
        #  ****************************************************/
        atv = []
        for Rn in range(TCpyCell+1):
            un_pack = struct.unpack(edn+'4d',f.read(sizeof_double*4))
            atv.append(list(un_pack))
        atv = np.array(atv)
        self.atv = atv
        
        #  /****************************************************
        #    int atv_ijk[TCpyCell+1][4];
        #  i,j,and j number of periodically copied cells
        #  ****************************************************/
        atv_ijk = []
        for Rn in range(TCpyCell+1):
            un_pack = struct.unpack(edn+'4i',f.read(sizeof_int*4))
            atv_ijk.append(list(un_pack))
        atv_ijk = np.array(atv_ijk)
        self.atv_ijk = atv_ijk     
        
        #  /****************************************************
        #    int Total_NumOrbs[atomnum+1];
        # the number of atomic orbitals in each atom
        #    int FNAN[atomnum+1];
        # the number of first neighboring atoms of each atom
        #  ****************************************************/
        fmt = fmtstr(atomnum,'i')
        p_vec = struct.unpack(fmt,f.read(sizeof_int*atomnum))
        Total_NumOrbs = [1] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            Total_NumOrbs.append(p_vec[ct_AN-1])
        Total_NumOrbs = np.array(Total_NumOrbs)
        del(p_vec)
        self.Total_NumOrbs = Total_NumOrbs     
        
        fmt = fmtstr(atomnum,'i')
        p_vec = struct.unpack(fmt,f.read(sizeof_int*atomnum))
        FNAN = [0] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            FNAN.append(p_vec[ct_AN-1])
        FNAN = np.array(FNAN)
        del(p_vec)
        self.FNAN = FNAN     
        
        #  /****************************************************
        #    int natn[atomnum+1][FNAN[ct_AN]+1];
        #  grobal index of neighboring atoms of an atom ct_AN
        #    int ncn[atomnum+1][FNAN[ct_AN]+1];
        #  grobal index for cell of neighboring atoms
        #  of an atom ct_AN
        #  note: natn[i][0] and ncn[i][0] mean the global index of atom i itself
        #  ****************************************************/
        natn = [0] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            fmt = fmtstr(FNAN[ct_AN]+1,'i')
            un_pack = struct.unpack(fmt,f.read(sizeof_int*(FNAN[ct_AN]+1)))
            natn.append(list(un_pack))
        # stored in the form of list
        self.natn = natn     
        
        ncn = [0] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            fmt = fmtstr(FNAN[ct_AN]+1,'i')
            un_pack = struct.unpack(fmt,f.read(sizeof_int*(FNAN[ct_AN]+1)))
            ncn.append(list(un_pack))
        # stored in the form of list
        self.ncn = ncn     
        
        #  /****************************************************
        #  double  tv[4][4]:
        #    unit cell vectors in Bohr
        #  ****************************************************/
        tv = [[0.,0.,0.,0.]] #dummy
        for i in range(3):
            un_pack = struct.unpack(edn+'4d',f.read(sizeof_double*4))
            tv.append(list(un_pack))
        tv = np.array(tv)
        self.tv = tv     
        
        #  /****************************************************
        #  double  rtv[4][4]:
        #  reciprocal unit cell vectors in Bohr^{-1}
        #  note:
        #   tv_i \dot rtv_j = 2PI * Kronecker's delta_{ij}
        #  ****************************************************/
        rtv = [[0.,0.,0.,0.]] #dummy
        for i in range(3):
            un_pack = struct.unpack(edn+'4d',f.read(sizeof_double*4))
            rtv.append(list(un_pack))
        rtv = np.array(rtv)
        self.rtv = rtv     
        
        #  /****************************************************
        # double Gxyz[atomnum+1][60];
        #  atomic coordinates in Bohr
        #  ****************************************************/
        Gxyz = [[0.]*60] #dummy
        for ct_AN in range(atomnum+1)[1:]:
            un_pack = struct.unpack(edn+'4d',f.read(sizeof_double*4))
            Gxyz.append(list(un_pack)+[0.]*56)
        Gxyz = np.array(Gxyz)
        self.Gxyz = Gxyz     
        
        #  /****************************************************
        #    Kohn-Sham Hamiltonian
        #
        #     double Hks[SpinP_switch+1]
        #               [atomnum+1]
        #               [FNAN[ct_AN]+1]
        #               [Total_NumOrbs[ct_AN]]
        #               [Total_NumOrbs[h_AN]];
        #
        ########################################################
        #
        #      Hks[spin][ct_AN][h_AN][i][j]
        #
        #      spin:  spin=0, up
        #             spin=1, down
        #
        #      ct_AN: global index of atoms
        #      h_AN   local index of neighbouring atoms for the atom ct_AN
        #      i:     orbital index in the atom ct_AN
        #      j:     orbital index in the atom h_AN
        #
        #   NOTE: 
        #
        #      For instance, if the basis specification of the atom ct_AN is s2p2,
        #      then the obital index runs in order of
        #                    s, s', px, py, pz, px', py', pz'
        #
        #      Transformation of the local index h_AN to the grobal index Gh_AN
        #      is made as
        #
        #                       Gh_AN = natn[ct_AN][h_AN];
        #
        #      Also, the cell index is given by
        #
        #                       Rn = ncn[ct_AN][h_AN];
        #      
        #      Each component l, m, or n (Rn = l*a + m*b + n*c) are given by
        #   
        #                       l = atv_ijk[Rn][1];
        #                       m = atv_ijk[Rn][2];
        #                       n = atv_ijk[Rn][3];
        #  ****************************************************/
        Hks = [] #H(ia)(jb)
        for spin in range(SpinP_switch+1):
            list_spin = [0]  #dummy
            for ct_AN in range(atomnum+1)[1:]: #i
                list_ctAN = []
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN]+1): #j
                    list_hAN = []
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1): #a
                        fmt = fmtstr(TNO2,'d')
                        un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                        list_hAN.append(list(un_pack))
                    list_ctAN.append(np.array(list_hAN))
                list_spin.append(list_ctAN)
            Hks.append(list_spin)
        self.Hks = Hks     
        
        #  /****************************************************
        #  iHks:
        #  imaginary Kohn-Sham matrix elements of basis orbitals
        #  for alpha-alpha, beta-beta, and alpha-beta spin matrices
        #  of which contributions come from spin-orbit coupling 
        #  and Hubbard U effective potential.
        #  ****************************************************/
        iHks = [] #iH(ia)(jb)
        for spin in range(3):
            list_spin = [0]  #dummy
            for ct_AN in range(atomnum+1)[1:]: #i
                list_ctAN = []
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN]+1): #j
                    list_hAN = []
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1): #a
                        if (SpinP_switch==3):
                            fmt = fmtstr(TNO2,'d')
                            un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                            list_hAN.append(list(un_pack))
                        else:
                            list_hAN.append([0.0]*TNO2)
                    list_ctAN.append(np.array(list_hAN))
                list_spin.append(list_ctAN)
            iHks.append(list_spin)
        self.iHks = iHks     
        
        #  /****************************************************
        #    Overlap matrix
        #
        #     double OLP[atomnum+1]
        #               [FNAN[ct_AN]+1]
        #               [Total_NumOrbs[ct_AN]]
        #               [Total_NumOrbs[h_AN]]; 
        #  ****************************************************/
        OLP = [0] #OLP(ia)(jb) + dummy
        for ct_AN in range(atomnum+1)[1:]: #i
            list_ctAN = []
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN]+1): #j
                list_hAN = []
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                for i in range(TNO1): #a
                    fmt = fmtstr(TNO2,'d')
                    un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                    list_hAN.append(list(un_pack))
                list_ctAN.append(np.array(list_hAN))
            OLP.append(list_ctAN)
        self.OLP = OLP     
        
        #  /****************************************************
        #    Overlap matrix with position operator x, y, z
        #
        #    double ******OLPpo;
        #    [3]
        #    [1]
        #    [atomnum+1]
        #    [FNAN[ct_AN]+1]
        #    [Total_NumOrbs[ct_AN]]
        #    [Total_NumOrbs[h_AN]]; 
        #  ****************************************************/
        OLPpo = [] #OLPpo[[OLPpox],[OLPpoy],[OLPpoz]]
        for direction in range(3):
            list_direction = []
            for order in range(order_max): 
                OLPpoi = [0] #OLPpoi(ia)(jb) + dummy
                for ct_AN in range(atomnum+1)[1:]: #i
                    list_ctAN = []
                    TNO1 = Total_NumOrbs[ct_AN]
                    for h_AN in range(FNAN[ct_AN]+1): #j
                        list_hAN = []
                        Gh_AN = natn[ct_AN][h_AN]
                        TNO2 = Total_NumOrbs[Gh_AN]
                        for i in range(TNO1): #a
                            fmt = fmtstr(TNO2,'d')
                            un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                            list_hAN.append(list(un_pack))
                        list_ctAN.append(np.array(list_hAN))
                    OLPpoi.append(list_ctAN)
                list_direction.append(copy.deepcopy(OLPpoi))
            OLPpo.append(copy.deepcopy(list_direction))
        self.OLPpo = OLPpo     

        #  /****************************************************
        #    Overlap matrix with momentum operator px, py, pz
        #
        #    double *****OLPmo;
        #    [3]
        #    [atomnum+1]
        #    [FNAN[ct_AN]+1]
        #    [Total_NumOrbs[ct_AN]]
        #    [Total_NumOrbs[h_AN]]; 
        #   ****************************************************/
        OLPmo = [] #OLPmo[OLPmox,OLPmoy,OLPmoz]
        for direction in range(3):
            OLPmoi = [0] #OLPmoi(ia)(jb) + dummy
            for ct_AN in range(atomnum+1)[1:]: #i
                list_ctAN = []
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN]+1): #j
                    list_hAN = []
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1): #a
                        fmt = fmtstr(TNO2,'d')
                        un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                        list_hAN.append(list(un_pack))
                    list_ctAN.append(np.array(list_hAN))
                OLPmoi.append(list_ctAN)
            OLPmo.append(copy.deepcopy(OLPmoi))
        self.OLPmo = OLPmo     

        #  /****************************************************
        #    Density matrix
        #
        #     double DM[SpinP_switch+1]
        #              [atomnum+1]
        #              [FNAN[ct_AN]+1]
        #              [Total_NumOrbs[ct_AN]]
        #              [Total_NumOrbs[h_AN]];
        #  ****************************************************/
        DM = [] #DM(ia)(jb)
        for spin in range(SpinP_switch+1):
            list_spin = [0]  #dummy
            for ct_AN in range(atomnum+1)[1:]: #i
                list_ctAN = []
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN]+1): #j
                    list_hAN = []
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1): #a
                        fmt = fmtstr(TNO2,'d')
                        un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                        list_hAN.append(list(un_pack))
                    list_ctAN.append(np.array(list_hAN))
                list_spin.append(list_ctAN)
            DM.append(list_spin)
        self.DM = DM     

        #  /*******************************************************
        #    Density matrix iDM
        #                      
        #    double *****iDM;
        #    density matrix
        #  size: iDM[2]
        #  [atomnum+1]
        #  [FNAN[ct_AN]+1]
        #  [Total_NumOrbs[ct_AN]]
        #  [Total_NumOrbs[h_AN]]
        #   *******************************************************/
        iDM = [] #iDM(ia)(jb)
        for spin in range(2):
            list_spin = [0]  #dummy
            for ct_AN in range(atomnum+1)[1:]: #i
                list_ctAN = []
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN]+1): #j
                    list_hAN = []
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1): #a
                        fmt = fmtstr(TNO2,'d')
                        un_pack = struct.unpack(fmt,f.read(sizeof_double*TNO2)) #b
                        list_hAN.append(list(un_pack))
                    list_ctAN.append(np.array(list_hAN))
                list_spin.append(list_ctAN)
            iDM.append(list_spin)
        self.iDM = iDM     

        #  /****************************************************
        # int Solver;
        #  method for solving eigenvalue problem
        #  ****************************************************/
        i_vec = struct.unpack(edn+'1i',f.read(sizeof_int*1))
        Solver = i_vec[0]
        self.Solver = Solver
        
        #  /****************************************************
        # double ChemP;
        #  chemical potential
        # double E_Temp;
        #  electronic temperature
        # double dipole_moment_core[4];
        # double dipole_moment_background[4];
        # int Valence_Electrons;
        #  total number of valence electrons
        # double Total_SpinS;
        #  total value of Spin (2*Total_SpinS = muB)
        #  ****************************************************/
        d_vec = struct.unpack(edn+'10d',f.read(sizeof_double*10))
        ChemP  = d_vec[0]
        E_Temp = d_vec[1]
        dipole_moment_core = np.array([0.,d_vec[2],d_vec[3],d_vec[4]])
        dipole_moment_background = np.array([0.,d_vec[5],d_vec[6],d_vec[7]])
        Valence_Electrons = d_vec[8]
        Total_SpinS = d_vec[9]
        
        self.ChemP  = ChemP 
        self.E_Temp = E_Temp
        self.dipole_moment_core = dipole_moment_core
        self.dipole_moment_background = dipole_moment_background
        self.Valence_Electrons = Valence_Electrons
        self.Total_SpinS =       Total_SpinS 
        
        #  /****************************************************
        #      input file 
        #  ****************************************************/
        i_vec = struct.unpack(edn+'1i',f.read(sizeof_int*1))
        num_lines = i_vec[0]
        temporal_input = []
        for i in range(num_lines+1)[1:]:
            fmt = fmtstr(MAX_LINE_SIZE,'c')
            un_pack = struct.unpack(fmt,f.read(sizeof_char*MAX_LINE_SIZE))
            l = list(un_pack)
            if (sys.version_info[0]==3):
                new_line = ""
            else:
                new_line = "".decode('utf-8')
            nTF = True
            while(nTF):
                ch =  l.pop(0)
                ch = ch.decode('utf-8','ignore')
                if (ch=='\n'):
                    nTF = False
                new_line += ch
            temporal_input.append(new_line)
        self.temporal_input = temporal_input     

        f.close()
    
    def input_file(self, scfout_file): 
        self.scfout_file = scfout_file

    def print_Hks(self):
        for spin in range(self.SpinP_switch+1):
            print("\n\nKohn-Sham Hamiltonian spin=%i\n"%(spin))
            for ct_AN in range(self.atomnum+1)[1:]: #i
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1): #j
                    Gh_AN = self.natn[ct_AN][h_AN]
                    Rn = self.ncn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    print("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n"%(ct_AN,h_AN,Gh_AN,Rn))
                    for i in range(TNO1): #a
                        for j in range(TNO2): #b
                            print("%10.7f "%(self.Hks[spin][ct_AN][h_AN][i][j]))
                        print("\n")
    
    def print_iHks(self):
        for spin in range(3):
            print("\n\nimaginary Kohn-Sham Hamiltonian spin=%i\n"%(spin))
            for ct_AN in range(self.atomnum+1)[1:]: #i
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1): #j
                    Gh_AN = self.natn[ct_AN][h_AN]
                    Rn = self.ncn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    print("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n"%(ct_AN,h_AN,Gh_AN,Rn))
                    for i in range(TNO1): #a
                        for j in range(TNO2): #b
                            print("%10.7f "%(self.iHks[spin][ct_AN][h_AN][i][j]))
                        print("\n")

    def print_OLP(self):
        print("\n\nOverlap matrix\n")
        for ct_AN in range(self.atomnum+1)[1:]: #i
            TNO1 = self.Total_NumOrbs[ct_AN]
            for h_AN in range(self.FNAN[ct_AN]+1): #j
                Gh_AN = self.natn[ct_AN][h_AN]
                Rn = self.ncn[ct_AN][h_AN]
                TNO2 = self.Total_NumOrbs[Gh_AN]
                print("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n"%(ct_AN,h_AN,Gh_AN,Rn))
                for i in range(TNO1): #a
                    for j in range(TNO2): #b
                        print("%10.7f "%(self.OLP[ct_AN][h_AN][i][j]))
                    print("\n")

    def print_OLPpox(self):
        try:
            OLPpox = self.OLPpo[0][0]
        except:
            OLPpox = self.OLPpox
        print("\n\nOverlap matrix with position operator x\n")
        for ct_AN in range(self.atomnum+1)[1:]: #i
            TNO1 = self.Total_NumOrbs[ct_AN]
            for h_AN in range(self.FNAN[ct_AN]+1): #j
                Gh_AN = self.natn[ct_AN][h_AN]
                Rn = self.ncn[ct_AN][h_AN]
                TNO2 = self.Total_NumOrbs[Gh_AN]
                print("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n"%(ct_AN,h_AN,Gh_AN,Rn))
                for i in range(TNO1): #a
                    for j in range(TNO2): #b
                        print("%10.7f "%(OLPpox[ct_AN][h_AN][i][j]))
                    print("\n")

    def print_OLPpoy(self):
        try:
            OLPpoy = self.OLPpo[1][0]
        except:
            OLPpoy = self.OLPpoy
        print("\n\nOverlap matrix with position operator y\n")
        for ct_AN in range(self.atomnum+1)[1:]: #i
            TNO1 = self.Total_NumOrbs[ct_AN]
            for h_AN in range(self.FNAN[ct_AN]+1): #j
                Gh_AN = self.natn[ct_AN][h_AN]
                Rn = self.ncn[ct_AN][h_AN]
                TNO2 = self.Total_NumOrbs[Gh_AN]
                print("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n"%(ct_AN,h_AN,Gh_AN,Rn))
                for i in range(TNO1): #a
                    for j in range(TNO2): #b
                        print("%10.7f "%(OLPpoy[ct_AN][h_AN][i][j]))
                    print("\n")

    def print_OLPpoz(self):
        try:
            OLPpoz = self.OLPpo[2][0]
        except:
            OLPpoz = self.OLPpoz
        print("\n\nOverlap matrix with position operator z\n")
        for ct_AN in range(self.atomnum+1)[1:]: #i
            TNO1 = self.Total_NumOrbs[ct_AN]
            for h_AN in range(self.FNAN[ct_AN]+1): #j
                Gh_AN = self.natn[ct_AN][h_AN]
                Rn = self.ncn[ct_AN][h_AN]
                TNO2 = self.Total_NumOrbs[Gh_AN]
                print("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n"%(ct_AN,h_AN,Gh_AN,Rn))
                for i in range(TNO1): #a
                    for j in range(TNO2): #b
                        print("%10.7f "%(self.OLPpoz[ct_AN][h_AN][i][j]))
                    print("\n")

    def print_DM(self):
        for spin in range(self.SpinP_switch+1):
            print("\n\nDensity matrix spin=%i\n"%(spin))
            for ct_AN in range(self.atomnum+1)[1:]: #i
                TNO1 = self.Total_NumOrbs[ct_AN]
                for h_AN in range(self.FNAN[ct_AN]+1): #j
                    Gh_AN = self.natn[ct_AN][h_AN]
                    Rn = self.ncn[ct_AN][h_AN]
                    TNO2 = self.Total_NumOrbs[Gh_AN]
                    print("glbal index=%i  local index=%i (grobal=%i, Rn=%i)\n"%(ct_AN,h_AN,Gh_AN,Rn))
                    for i in range(TNO1): #a
                        for j in range(TNO2): #b
                            print("%10.7f "%(self.DM[spin][ct_AN][h_AN][i][j]))
                        print("\n")





