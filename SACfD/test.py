import numpy as np
C=200*250/1.5
K = np.array([[1.5e6,-1.5e6,],[-1.5e6, 1.5e6]])
u = np.array([[0],[2.193e-3]])
#np.linalg.solve(A, b) 解 Ax = b
# u = np.linalg.solve(C*K, F)
print(np.dot(K,u))