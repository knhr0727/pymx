import numpy as np

# Pauli matrices
s_x = np.array([[  0.,  1.],[  1.,  0.]])
s_y = np.array([[  0.,-1.j],[ 1.j,  0.]])
s_z = np.array([[  1.,  0.],[  0., -1.]])

def SU2_mat(n, theta, angle='radian'):
    u = n/np.linalg.norm(n)
    ux,uy,uz = u[0],u[1],u[2]
    if (angle[:3].lower()=='rad'):
        theta2 = theta
    elif (angle[:3].lower()=='deg'):
        theta2 = theta*np.pi/180.
    else:
        raise Exception("wrong angle in SU2")
    a = theta2/2.
    I = np.identity(2, dtype=complex)
    return I*np.cos(a)-1.j*(ux*s_x+uy*s_y+uz*s_z)*np.sin(a)

# (Euler-)Rodrigues formula
def rotation_mat(n, theta, angle='radian', rotation='active'):
    u = n/np.linalg.norm(n)
    ux,uy,uz = u[0],u[1],u[2]
    if (angle[:3].lower()=='rad'):
        theta2 = theta
    elif (angle[:3].lower()=='deg'):
        theta2 = theta*np.pi/180.
    else:
        raise Exception("wrong angle in rotation_mat")
    if (rotation[:3].lower()=='act'):
        theta2 = theta2
    elif (rotation[:3].lower()=='pas'):
        theta2 = -theta2
    else:
        raise Exception("wrong rotation in rotation_mat")
    cos = np.cos(theta2)
    sin = np.sin(theta2)
    I = np.identity(3,dtype=float)
    ucross = np.array([[0.,-uz,uy],[uz,0.,-ux],[-uy,ux,0.]])
    uuT = np.array([[ux*ux,ux*uy,ux*uz],[ux*uy,uy*uy,uy*uz],[ux*uz,uy*uz,uz*uz]])
    return cos*I + sin*ucross + (1.-cos)*uuT

def axis_angle(R_in):
    cri = 0.000002
    det = np.linalg.det(R_in)
    if (det < 0):
        R = -1.*R_in
        det = -det
    else:
        R = R_in
    if (abs(det-1.) < cri):
        w,v = np.linalg.eig(R)
        D = np.abs(w-np.ones(3,dtype=float))
        i = np.where(D==D.min())[0].item(0)
        n = v[:,i].real.flatten()
        TrR = np.trace(R)
    else:
        raise Exception("wrong R in axis_angle")
    arg = (TrR-1.)/2.
    if (arg>1.):
        theta = 0.
    elif (arg<-1.):
        theta = np.pi
    else:
        theta = np.arccos(arg)
    R1 = rotation_mat(n,theta)
    R2 = rotation_mat(n,-theta)
    sum1 = np.abs(R-R1).sum()
    sum2 = np.abs(R-R2).sum()
    if (sum1 > sum2):
        theta = -theta
    if (abs(theta)<cri):
        n = np.array([0.,0.,1.])
    return [n,theta]
    
def axis_angle_crystal(R_in):
    cri = 0.000002
    det = np.linalg.det(R_in)
    if (det < 0):
        R = -1.*R_in
        det = -det
    else:
        R = R_in
    if (abs(det-1.) < cri):
        TrR = np.trace(R)
    else:
        raise Exception("wrong R in axis_angle_crystal")
    diff_list = []
    for i in range(-1,4):
        diff_list.append(abs(TrR-float(i)))
    m = min(diff_list)
    im = diff_list.index(m)
    if (m>cri):
        raise Exception("wrong R in axis_angle_crystal")
    else:
        tr = im-1
    if  (tr==-1):
        theta = np.pi
        order = 2
    elif(tr== 0):
        theta = 2.*np.pi/3.
        order = 3
    elif(tr== 1):
        theta = np.pi/2.
        order = 4
    elif(tr== 2):
        theta = np.pi/3.
        order = 6
    else:
        theta = 0.
        order = 1
    Y = np.identity(3,dtype=float)
    Rn = np.identity(3,dtype=float)
    for i in range(order-1):
        Rn = np.dot(Rn,R)
        Y += Rn
    Yy = [np.abs(Y[:,0]).sum(),np.abs(Y[:,1]).sum(),np.abs(Y[:,2]).sum()]
    yi = Yy.index(max(Yy))  
    n = Y[:,yi].flatten()
    n = n/np.linalg.norm(n)
    R1 = rotation_mat(n,theta)
    R2 = rotation_mat(-n,theta)
    sum1 = np.abs(R-R1).sum()
    sum2 = np.abs(R-R2).sum()
    if (sum1 > sum2):
        n = -n
    if (abs(theta)<cri):
        n = np.array([0.,0.,1.])
    return [n,theta]
    
def identity_mat():
    return np.identity(3)

def inversion_mat():
    return -1.*np.identity(3)

def mirror_mat(n):
    return -1.*rotation_mat(n, np.pi)

# Rotation matices for real spherical harmonics according to ang.mom. quantum number l
# "Direct Determination by Recursion" J. Phys. Chem. 1996, 100, 6342-6347
# modified for rotation of wave function (not of coordinate)
# R^{l}_{mm'} = <Y_{lm}(r)|Y_{lm'}(R^{-1}r)>

def kdelta(i,j):
    I = int(i)
    J = int(j)
    if (I==J):
        return 1
    else:
        return 0

# matrices permute the order of orbitals from real spherical harmonics to openMX
# l=1
r2o1 = np.zeros((3,3),dtype=float)
r2o1[0,2],r2o1[1,0],r2o1[2,1] = 1.,1.,1.
# l=2
r2o2 = np.zeros((5,5),dtype=float)
r2o2[0,2],r2o2[1,4],r2o2[2,0],r2o2[3,3],r2o2[4,1] = 1.,1.,1.,1.,1.
# l=3
r2o3 = np.zeros((7,7),dtype=float)
r2o3[0,3],r2o3[1,4],r2o3[2,2],r2o3[3,5],r2o3[4,1],r2o3[5,6],r2o3[6,0] = \
1.,1.,1.,1.,1.,1.,1.
r2o1t = r2o1.transpose().copy()
r2o2t = r2o2.transpose().copy()
r2o3t = r2o3.transpose().copy()

class Rl:

    def u(self, l,m1,m2):
        L,M1,M2 = int(l),int(m1),int(m2)
        lf,m1f,m2f = float(l),float(m1),float(m2)
        if ((abs(M1)>L)or(abs(M2)>L)):
            raise Exception("|m1| or |m2| is larger than l in u")
        if (abs(M2)==L):
            return np.sqrt((lf+m1f)*(lf-m1f)/(2.*lf*(2.*lf-1.)))
        else:
            return np.sqrt((lf+m1f)*(lf-m1f)/((lf+m2f)*(lf-m2f)))
    
    def v(self, l,m1,m2):
        L,M1,M2 = int(l),int(m1),int(m2)
        lf,m1f,m2f = float(l),float(m1),float(m2)
        if ((abs(M1)>L)or(abs(M2)>L)):
            raise Exception("|m1| or |m2| is larger than l in v")
        numer = (1.+kdelta(M1,0))*(lf+abs(m1f)-1.)*(lf+abs(m1f))
        if (abs(M2)==L):
            deno = 2.*lf*(2.*lf-1.)
        else:
            deno = (lf+m2f)*(lf-m2f)
        return 0.5*np.sqrt(numer/deno)*(1.-2.*kdelta(M1,0))
    
    def w(self, l,m1,m2):
        L,M1,M2 = int(l),int(m1),int(m2)
        lf,m1f,m2f = float(l),float(m1),float(m2)
        if ((abs(M1)>L)or(abs(M2)>L)):
            raise Exception("|m1| or |m2| is larger than l in w")
        numer = (lf-abs(m1f)-1.)*(lf-abs(m1f))
        if (abs(M2)==L):
            deno = 2.*lf*(2.*lf-1.)
        else:
            deno = (lf+m2f)*(lf-m2f)
        return -0.5*np.sqrt(numer/deno)*(1.-kdelta(M1,0))

    def P(self, i,l,m1,m2):
        I,L,M1,M2 = int(i),int(l),int(m1),int(m2)
        if ((abs(M1)>L)or(abs(M2)>L)):
            raise Exception("wrong m1 or m2 in P")
        R = self.R1rsh
        d1 = 1
        if (L==2):
            Rb = self.R1rsh
            d2 = 1
        elif (L==3):
            Rb = self.R2rsh
            d2 = 2
        else:
            raise Exception("wrong l in P")
        if (M2==L):
            #return R[I+d1,1+d1]*Rb[M1+d2,M2+d2]-R[I+d1,-1+d1]*Rb[M1+d2,-M2+d2]
            return R[I+d1,1+d1]*Rb[M1+d2,M2-1+d2]-R[I+d1,-1+d1]*Rb[M1+d2,-M2+1+d2]
        elif (M2==-L):
            #return R[I+d1,1+d1]*Rb[M1+d2,M2+d2]+R[I+d1,-1+d1]*Rb[M1+d2,-M2+d2]
            return R[I+d1,1+d1]*Rb[M1+d2,M2+1+d2]+R[I+d1,-1+d1]*Rb[M1+d2,-M2-1+d2]
        else:
            return R[I+d1,0+d1]*Rb[M1+d2,M2+d2]

    def U(self, l,m1,m2):
        L,M1,M2 = int(l),int(m1),int(m2)
        if ((abs(M1)>L)or(abs(M2)>L)):
            raise Exception("|m1| or |m2| is larger than l in U")
        if (abs(M1)==L):
            return 0.
        else:
            return self.P(0,L,M1,M2)

    def V(self, l,m1,m2):
        L,M1,M2 = int(l),int(m1),int(m2)
        if ((abs(M1)>L)or(abs(M2)>L)):
            raise Exception("|m1| or |m2| is larger than l in V")
        if (M1==0):
            return self.P(1,L,1,M2)+self.P(-1,L,-1,M2)
        elif (M1>0):
            term1 = self.P(1,L,M1-1,M2)*np.sqrt(1.+kdelta(M1,1))
            if (M1==1):
                term2 = 0.
            else:
                term2 = self.P(-1,L,-M1+1,M2)
            return term1 - term2
        elif (M1<0):
            if (M1==-1):
                term1 = 0.
            else:
                term1 = self.P(1,L,M1+1,M2)
            term2 = self.P(-1,L,-M1-1,M2)*np.sqrt(1.+kdelta(M1,-1))
            return term1 + term2

    def W(self, l,m1,m2):
        L,M1,M2 = int(l),int(m1),int(m2)
        if ((abs(M1)>L)or(abs(M2)>L)):
            raise Exception("|m1| or |m2| is larger than l in W")
        if ((M1==0)or(abs(M1)==L)or(L-abs(M1)-1==0)):
            return 0.
        elif (M1>0):
            return self.P(1,L,M1+1,M2) + self.P(-1,L,-M1-1,M2)
        elif (M1<0):
            return self.P(1,L,M1-1,M2) - self.P(-1,L,-M1+1,M2)

    def calc_R(self, l):
        L = int(l)
        if ((L!=2)and(L!=3)):
            raise Exception("l is out of range for this program in get_R")
        if ((L==2)and(self.R1rsh is None)):
            raise Exception("R1 is not defined in get_R")
        elif ((L==3)and(self.R2rsh is None)):
            raise Exception("R2 is not defined in get_R")
        R = np.empty((2*L+1,2*L+1),dtype=float)
        r = range(-L,L+1,1)
        d = L
        for m1 in r:
            for m2 in r:
                term1 = self.u(L,m1,m2)*self.U(L,m1,m2)
                term2 = self.v(L,m1,m2)*self.V(L,m1,m2)
                term3 = self.w(L,m1,m2)*self.W(L,m1,m2)
                R[m1+d,m2+d] = term1 + term2 + term3
        if (L==2):
            self.R2rsh = R.copy()
            self.R2 = np.dot(r2o2,np.dot(R,r2o2t))
        elif (L==3):
            self.R3rsh = R.copy()
            self.R3 = np.dot(r2o3,np.dot(R,r2o3t))

    def calc_R_all(self):
        self.calc_R(2)
        self.calc_R(3)

    def get_R(self, *args, **kwargs):
        if ('R1' in kwargs):
            t1,t2,t3 = type(R1),type(np.empty(1)),type(np.mat(0))
            if ((t1==t2)or(t1==t3)):
                if ((R1.shape==(3,3))and(R1.ndim==2)):
                    self.R1 = R1
                    self.R1rsh = np.dot(r2o1t,np.dot(self.R1,r2o1))
                    self.calc_R_all()
                else:
                    raise Exception("wrong shape of R1 in Rl")
            else:
                raise Exception("input of Rl is 3x3 array or matrix")
        if (self.R2rsh is None):
            if (self.R1rsh is not None):
                self.calc_R_all()
            else:
                raise Exception("R1 is not defined")
        if (len(args)==0):
            return [self.R0,self.R1,self.R2,self.R3]
        elif (len(args)>1):
            raise Exception("too many arguments in get_R")
        else:
            L = int(args[0])
            if (L==0):
                return self.R0
            elif (L==1):
                return self.R1
            elif (L==2):
                return self.R2
            elif (L==3):
                return self.R3
            else:
                raise Exception("l is out of range in get_R")

    def __init__(self, *args):
        self.R0 = np.array([[1.]])
        if (args):
            R1 = args[0]
            t1,t2,t3 = type(R1),type(np.empty(1)),type(np.mat(0))
            if ((t1==t2)or(t1==t3)):
                if ((R1.shape==(3,3))and(R1.ndim==2)):
                    self.R1 = R1.copy()
                    self.R1rsh = np.dot(r2o1t,np.dot(self.R1,r2o1))
                    self.calc_R_all()
                else:
                    raise Exception("wrong shape of R1 in Rl")
            else:
                raise Exception("input of Rl is 3x3 array or matrix")
        else:
            self.R1 = None
            self.R1rsh = None
            self.R2 = None
            self.R2rsh = None
            self.R3 = None
            self.R3rsh = None

    def check_det(self):
        if (self.R0 is not None):
            print('det(R0) = '+str(np.linalg.det(self.R0)))
        if (self.R1 is not None):
            print('det(R1) = '+str(np.linalg.det(self.R1)))
        if (self.R2 is not None):
            print('det(R2) = '+str(np.linalg.det(self.R2)))
        if (self.R3 is not None):
            print('det(R3) = '+str(np.linalg.det(self.R3)))

    def identity(self):
        self.R1 = np.identity(3)
        self.R1rsh = np.identity(3)

    def rotation(self, n, theta, angle='radian', rotation='active'):
        self.R1 = rotation_mat(n, theta, angle=angle, rotation=rotation)
        self.R1rsh = np.dot(r2o1t,np.dot(self.R1,r2o1))

    def inversion(self):
        try:
            self.R1 *= -1.
            self.R1rsh *= -1.
        except:
            self.R1 = -1.*np.identity(3)
            self.R1rsh = -1.*np.identity(3)

    def mirror(self, n):
        self.R1 = -1.*rotation_mat(n, np.pi)
        self.R1rsh = np.dot(r2o1t,np.dot(self.R1,r2o1))
