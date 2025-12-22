import numpy as np
import matplotlib.pyplot as plt

T = 3
dt = 0.0001
N = int(T/dt) + 1
nx = 300
dx = 2/(nx-1)
nu = 0.1

x = np.linspace(-1, 1, nx)
t = np.zeros(N)
u = np.zeros((N, nx))

#u0 = 1/8*(x+1)**4
#u0 = np.sin(4*x)
u0 = np.exp(100*x)
#u0 = np.tan(1.5*x)
#u0 = x**10
#u0 = np.sin(np.exp(-3*x))

u[0] = u0

c1 = u0[0]
c2 = u0[-1]

def laplacian1D(vec):      # same function name as periodic 1d laplacian
    
    newvec = np.zeros_like(vec) 
    newvec[1:-1] = (vec[:-2] - 2*vec[1:-1] + vec[2:]) / dx**2
    
    return newvec

for i in range(N-1):
    t[i+1] = t[i] + dt
    u[i+1] = u[i] + nu*laplacian1D(u[i])*dt
    u[i+1,0] = c1
    u[i+1,-1] = c2



#### Animate

from matplotlib.animation import FuncAnimation 

plt.style.use('dark_background')

fig, ax = plt.subplots(figsize=(12,7))

plt.xlabel('x')
plt.ylabel('u')

ax.grid(True, color='grey', linestyle='-', linewidth=0.2)


line, = ax.plot(x,u[0])

ax.set_xlim(-1,1)
ax.set_ylim(np.min(u[0]), np.max(u[0]))


def update(frames):

    line.set_ydata(u[frames])

    return line,

steps = 100

frames = range(0, N, steps)

ani = FuncAnimation(fig, update, frames=frames, interval=30)

plt.show()