import numpy as np
import matplotlib.pyplot as plt

"""
1D Heat Equation
Periodic boundary conditions
Explicit Euler scheme
"""

# Space:

nx = 100
dx = 2/(nx-1)
x = np.linspace(-1, 1, nx)

# Time:

T = 0.5
dt = 0.0001
N = int(T/dt)+1
t = np.zeros(N)

# Physics:

nu = 0.1

# Stability check and conditional adjust:

if dt > dx**2/(2*nu):
    dt = dx**2/(2*nu)
    N = int(T/dt)+1
    t = np.zeros(N)
    print(f'Stability Warning: Timestep adjusted to dt = {dt}')

# Initialization:

#u0 = np.cos(2*np.pi*x) + 1/2*np.cos(4*np.pi*x)
u0 = np.sin(4*np.pi*x) + 1/3*np.sin(7*np.pi*x)

u = np.zeros((N,nx))

u[0] = u0


def laplacian(f):

    # Periodic boundary conditions via np.roll:
    
    return (np.roll(f,-1) - 2*f + np.roll(f, 1))/dx**2

# Explicit Euler scheme integrator loop:

for i in range(N-1):
    
    u[i+1] = u[i] + nu*laplacian(u[i])*dt
    t[i+1] = t[i] + dt


# Animation: 

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

step = 10 # Gap between frames is this many integrator time steps.

umin = np.min(u)
umax = np.max(u)

ax.set_ylim(umin, umax)
ax.set_xlim(-1, 1)

line, = ax.plot([], [])

def update(frame):
    line.set_data(x, u[frame])
    ax.set_title(f't = {t[frame]:.2f}')

    return line,

frames = range(0, len(u), step) # frames can be an iterable as well as int.

ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()