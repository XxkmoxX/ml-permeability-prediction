# Numpy file creation (.npy)
import os
import numpy as np

cwd = os.getcwd()
rdir = ['/ics_ref_lc_bin_test/', '/hcs_ref_lc/']

realizations = 10

X = np.zeros((realizations, 256, 256, 1), dtype=np.float64)
Y = np.zeros((realizations, 1), dtype=np.float64)

print('X shape:', X.shape,'\n''Y shape:', Y.shape)
print(cwd + rdir[0])

for ir in range(realizations):
    x = np.load(cwd + rdir[0] + str(ir) + '/k.npy')
    y = np.loadtxt(cwd + rdir[0] + str(ir) + '/SolverRes.txt')[0]
    X[ir] = x
    Y[ir] = y

np.save(f'X_256L_bin_ics_test.npy', X)
np.save(f'Y_256L_bin_ics_test.npy', Y)
