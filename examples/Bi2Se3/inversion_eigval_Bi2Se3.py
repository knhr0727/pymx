import pymx
import numpy as np

scfoutfile = 'Bi2Se3.scfout'
pm = pymx.PyMX(scfoutfile)
pm.default_setting()

center = (pm.a1+pm.a2+pm.a3)/2.
I_mat = pymx.inversion_mat()

gamma = 0.*pm.b1
L  = 0.5*pm.b1
X = 0.5*pm.b1+0.5*pm.b2

print('inversion pair')
print(pm.Crystal_symmetry_pair(I_mat,center=center))

print('gamma')
INV = pm.Crystal_symmetry_mat(I_mat,gamma,center=center)
w,v = pm.TB_eigen(gamma)
for i in range(0,48,2):
    c = v[:,i]
    print np.linalg.multi_dot([c.conjugate(),INV,c])

print('L')
INV = pm.Crystal_symmetry_mat(I_mat,L,center=center)
w,v = pm.TB_eigen(L)
for i in range(0,48,2):
    c = v[:,i]
    print np.linalg.multi_dot([c.conjugate(),INV,c])

print('X')
INV = pm.Crystal_symmetry_mat(I_mat,X,center=center)
w,v = pm.TB_eigen(X)
for i in range(0,48,2):
    c = v[:,i]
    print np.linalg.multi_dot([c.conjugate(),INV,c])

print('\nparity of Bi2Se3 at Gamma point from DOI:10.1038/NPHYS1270 :')
print('+-+-+-+-+-+--+(HOMO);-(LUMO)')
