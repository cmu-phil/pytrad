# Test kci python implementation, using the example in the orginal Matlab implementation
import numpy as np
from kci.KCI import KCI_CInd, KCI_UInd
from kernel.GaussianKernel import GaussianKernel
from scipy.io import loadmat

# test_data = loadmat('test.mat')
# X = test_data['X']
# Y = test_data['Y']
# Z = test_data['Z']

X = np.random.randn(300, 1)
X1 = np.random.randn(300, 1)
Y = np.concatenate((X,X), axis=1) + 0.5*np.random.randn(300, 2)
Z = Y + 0.5*np.random.randn(300, 2)

kernelX = GaussianKernel()
kernelY = GaussianKernel()
kernelZ = GaussianKernel()

hsic = KCI_UInd(300, kernelX, kernelX)
pvalue = hsic.compute_pvalue(X, X1)
print(pvalue)

hsic = KCI_UInd(300, kernelX, kernelZ)
pvalue = hsic.compute_pvalue(X, Z)
print(pvalue)

kci = KCI_CInd(300, kernelX, kernelZ, kernelY)
pvalue = kci.compute_pvalue(X, Z, Y)
print(pvalue)