import pymx
import numpy as np

scfoutfile = 'Bi2Se3.scfout'
pm = pymx.PyMX(scfoutfile,ver='3.8')
pm.default_setting()

g  = 0.000000000000*pm.b1 + 0.000000000000*pm.b2 + 0.000000000000*pm.b3 
L  = 0.500000000000*pm.b1 + 0.000000000000*pm.b2 + 0.000000000000*pm.b3 
B1 = 0.500000000000*pm.b1 + 0.176157051841*pm.b2 - 0.176157051841*pm.b3
B  = 0.823842948159*pm.b1 + 0.500000000000*pm.b2 + 0.176157051841*pm.b3
Z  = 0.500000000000*pm.b1 + 0.500000000000*pm.b2 + 0.500000000000*pm.b3
X  = 0.338078525921*pm.b1 + 0.000000000000*pm.b2 - 0.338078525921*pm.b3
Q  = 0.661921474079*pm.b1 + 0.338078525921*pm.b2 + 0.000000000000*pm.b3
F  = 0.500000000000*pm.b1 + 0.500000000000*pm.b2 + 0.000000000000*pm.b3
P1 = 0.661921474079*pm.b1 + 0.661921474079*pm.b2 + 0.176157051841*pm.b3 
P  = 0.823842948159*pm.b1 + 0.338078525921*pm.b2 + 0.338078525921*pm.b3

EF = pm.ChemP

band1 = pm.Band(g,L ,30)
band2 = pm.Band(L,B1,30)
band3 = pm.Band(B,Z ,30)
band4 = pm.Band(Z,g ,30)
band5 = pm.Band(g,X ,30)
band6 = pm.Band(Q,F ,30)
band7 = pm.Band(F,P1,30)
band8 = pm.Band(P1,Z,30)
band9 = pm.Band(L,P ,30)

print('number of valence elecetrons : %d'%(pm.Valence_Electrons))

pm.PlotBand([band1,band2,band3,band4,band5,band6,band7,band8,band9],
            eV=True,EF=EF,shift=True,
            kticks_label=[r'$\Gamma$','L','B1|B','Z',r'$\Gamma$','X|Q','F','P1','Z|L','P'],
            yrange=[-5,5],highlight=[30,47])

