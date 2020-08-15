import numpy as np
from CMsubmit import CMsolver
from CAsubmit import CAsolver
from SGDsubmit import SGDsolver
from matplotlib import pyplot as plt

Z = np.loadtxt( "data" )

y = Z[:,0]
X = Z[:,1:]
C = 1
horizon = 1500000

(primal_CM, time_CM) = CMsolver( X, y, C, horizon )
(primal_CA, time_CA) = CAsolver( X, y, C, horizon )
(primal_SGD, time_SGD) = SGDsolver( X, y, C, horizon )

plt.figure(figsize=(10, 10))
plt.title("PrimalObjVal v/s Time")
plt.plot( time_CM, primal_CM, color = 'b', linestyle = '-', label = "CM Primal" )
plt.plot( time_CA, primal_CA, color = 'g', linestyle = ':', label = "CA Primal" )
plt.plot( time_SGD, primal_SGD, color = 'r', linestyle = '-', label = "SGD Primal" )
plt.legend()
plt.xlabel( "Elapsed time (sec)" )
plt.ylabel( "SVM Objective value" )
plt.ylim( 0, 10000 )
#plt.savefig('plot.jpg')
plt.show()

