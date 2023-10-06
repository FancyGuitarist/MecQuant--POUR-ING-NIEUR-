import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt

pi = np.pi
a = 20
j = complex(0, 1)
m = 9.1093837015e-31
e = 1.602176634e-19
t = np.linspace(0, 12, 6)
x = np.linspace(0, a, 1000) # define our grid

def penice_a(x, t):
    Penis = (1/np.sqrt(a))*(2*np.sin((np.pi*x)/a) * e**(-j * t) + np.sin((np.pi * x * 2)/a) * e**(-j * 2 * t))
    return Penis


fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(len(t)):
    ax.plot(x , penice_a(x, t[i]), label=f'temps : {t[i]:.2f} sec') #color=colors[i], + abs(penice_a(x, t[i]).imag)

fig.tight_layout()
fig.subplots_adjust(top=0.8)
plt.xlabel("Distance [-]")
plt.ylabel("Fonction d'onde")
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.legend()
fig.savefig(f"Graph_A", bbox_inches='tight',dpi=600)


fig = plt.figure()
ax = fig.add_subplot(111)
def penice_b(x, t):
    Penis = (1/np.sqrt(a))*(2*np.sin((np.pi*x)/a) * e**(-j * t) + np.sin((np.pi * x * 3)/a) * e**(-j * 9 * t))
    return Penis

for i in range(len(t)):
    ax.plot(x , penice_b(x, t[i]), label=f'temps : {t[i]:.2f} sec') # color=colors[i], + abs(penice_b(x, t[i]).imag)

fig.tight_layout()
fig.subplots_adjust(top=0.8)
plt.xlabel("Distance [-]")
plt.ylabel("Fonction d'onde")
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
ax.legend()
fig.savefig(f"Graph_B", bbox_inches='tight',dpi=600)

def PENICE(x,t):
    PENIS = (1/np.sqrt(a))*(2*np.sin((np.pi*x)/a) * e**(-j * t) + np.sin((np.pi * x * 2)/a) * e**(-j * 2 * t)) + (1/np.sqrt(a))*(2*np.sin((np.pi*x)/a) * e**(-j * t) + np.sin((np.pi * x * 3)/a) * e**(-j * 9 * t))
    return PENIS

for i in range(len(t)):
    ax.plot(x, PENICE(x, t[i]), label=f'temps : {t[i]:.2f} sec') #+ abs(PENICE(x, t[i]).imag)

fig.tight_layout()
fig.subplots_adjust(top=0.8)
plt.xlabel("Distance [-]")
plt.ylabel("Fonction d'onde")
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
#ax.legend()
fig.savefig(f"GraphThot", bbox_inches='tight',dpi=600)