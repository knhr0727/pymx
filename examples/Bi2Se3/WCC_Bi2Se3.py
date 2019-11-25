import pymx
import numpy as np

scfoutfile = 'Bi2Se3.scfout'
pm = pymx.PyMX(scfoutfile)
pm.default_setting()

gamma = 0.*pm.b1
 

wcc1 = pm.WCC(pm.b3,gamma,0.5*pm.b2,50,50,(30,47))
pm.PlotWCC(wcc1)

wcc2 = pm.WCC(pm.b3,gamma+0.5*pm.b1,0.5*pm.b2+0.5*pm.b1,50,50,(30,47))
pm.PlotWCC(wcc2)

wcc3 = pm.WCC(pm.b3,gamma,0.5*pm.b1,50,10,(30,47))
pm.PlotWCC(wcc3)

wcc4 = pm.WCC(pm.b3,gamma+0.5*pm.b2,0.5*pm.b1+0.5*pm.b2,50,10,(30,47))
pm.PlotWCC(wcc4)

wcc5 = pm.WCC(pm.b1,gamma,0.5*pm.b2,50,10,(30,47))
pm.PlotWCC(wcc5)

wcc6 = pm.WCC(pm.b1,gamma+0.5*pm.b3,0.5*pm.b2+0.5*pm.b3,50,10,(30,47))
pm.PlotWCC(wcc6)

