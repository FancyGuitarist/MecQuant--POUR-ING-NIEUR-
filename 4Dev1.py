import numpy as np
from scipy.constants import hbar

def build_Hmatrix(eigenvalue: float, v: np.ndarray, n: int):
  H = np.pad(-np.identity(n - 1), ((0,1),(1,0)), 'constant') + np.identity(n) * (2 + v) + np.pad(-np.identity(n - 1), ((1,0),(0,1)), 'constant')
  return H

def lambd(m, delta_x):
  return 1 / (2*m*delta_x**2)

A = 1
N = 3
deltx = A / (N + 1)
M = 1

def Energie(M, A, N):
    Energie = (N**2 * np.pi**2 * hbar**2) / (2 * M * A**2)
    return Energie

Hmatrix = build_Hmatrix(lambd(M, deltx), np.zeros(N), N)
print(Hmatrix)

eigenVals = np.linalg.eig(Hmatrix - np.eye(N) * Energie(M, A, N))[0][:]
print(eigenVals)

sorted_eigenVals = sorted(eigenVals)
for i in sorted_eigenVals[:5]:
  print(i*lambd(M, deltx))