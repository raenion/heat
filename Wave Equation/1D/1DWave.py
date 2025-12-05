import numpy as np
import matplotlib.pyplot as plt

T = 5
dt = 0.001
N = int(T/dt) + 1
c = 2

L = 10
nx = 500
x = np.linspace(0, L, nx)
dx = L/(nx-1)

t = np.zeros(N)
y = np.zeros((N, nx))


#y0 = np.sin(np.pi*x/L)
#y0 = np.exp(-10*(x-L/2)**2)
y0 = np.cos((np.pi/10*(x-L/2)))

y[0] = y0

y[1] = y0


def laplacian1D(vec):
    newvec = np.zeros_like(vec)

    newvec[1:-1] = (vec[:-2] - 2*vec[1:-1] + vec[2:])/dx**2

    return newvec

for i in range(1, N-1):
    t[i] = t[i-1] + dt
    y[i+1] = 2*y[i] - y[i-1] + (c**2)*laplacian1D(y[i])*(dt**2)

for i in range(0, N, int(N/5)):                                       # remove
    print(y[i, int(nx/2)])

fig, ax = plt.subplots()

plottimes = [0,1,2]

for time in plottimes:
    bestindex = np.argmin(np.abs(t - time))
    ax.plot(x, y[bestindex], label=f't = {time:.2f}')
    ax.legend()

plt.legend()

plt.show()
