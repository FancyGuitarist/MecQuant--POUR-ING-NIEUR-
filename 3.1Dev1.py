import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt
#from scipy import pi, hbar, planck
a = 20
j = complex(0, 1)


m = 1
e = 2.7182818284590452353602874713527
t = np.linspace(0, np.pi, 1000)

x = np.linspace(0, a, 1000) # define our grid

def penice_a(x, t):
    Penis = (1/np.sqrt(a))*(2*np.sin((np.pi*x)/a) * e**(-j * t) + np.sin((np.pi * x * 2)/a) * e**(-j * 2 * t))
    return Penis

colors = ['blue', 'black', 'orange', 'red', 'purple', 'green']
fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(len(t)):
    ax.plot(x , penice_a(x, t[i]).real**2 + penice_a(x, t[i]).imag**2 , label=f'temps : {t[i]:.2f} sec') #color=colors[i],

def penice_b(x, t):
    Penis = (1/np.sqrt(a))*(2*np.sin((np.pi*x)/a) * e**(-j * t) + np.sin((np.pi * x * 3)/a) * e**(-j * 9 * t))
    return Penis

for i in range(len(t)):
    ax.plot(x , penice_b(x, t[i]).real**2 + penice_b(x, t[i]).imag**2 , label=f'temps : {t[i]:.2f} sec') # color=colors[i],

fig.tight_layout()
fig.subplots_adjust(top=0.8)
plt.xlabel("Distance [-]")
plt.ylabel('Densité de Probabilité')
ax.tick_params(axis="y",direction="in")
ax.tick_params(axis="x",direction="in")
fig.savefig(f"Graph", bbox_inches='tight',dpi=600)