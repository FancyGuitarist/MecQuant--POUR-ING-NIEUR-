import numpy as np
from scipy.constants import hbar
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

def build_Hmatrix(eigenvalue: float, v: np.ndarray, n: int):
  H = np.pad(-np.identity(n - 1), ((0,1),(1,0)), 'constant') + np.identity(n) * (2 + v) + np.pad(-np.identity(n - 1), ((1,0),(0,1)), 'constant')
  return H

def lambd(m, delta_x):
  return hbar**2 / (2*m*delta_x**2)

A = 1
N = 100
deltx = A / (N + 1)
M = 1

def Energie(M, A, N):
    Energie = (N**2 * np.pi**2 * hbar**2) / (2 * M * A**2)
    return Energie

Hmatrix = build_Hmatrix(lambd(M, deltx), np.zeros(N), N)
#print(Hmatrix)

eigenVals, eigenVecs = eigs(Hmatrix + np.eye(N) * Energie(M, A, N))
#print(eigenVals)
y = eigenVecs[:, 2].real
#print(y)
x = np.linspace(0, A, N)
plt.plot(x, y)
plt.show()

sorted_eigenVals = sorted(eigenVals)
"""for i in sorted_eigenVals[:5]:
  print(i*lambd(M, deltx))"""

