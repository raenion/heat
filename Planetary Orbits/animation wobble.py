import numpy as np
import matplotlib.pyplot as plt


# ToDo: clean, sort, make planet more massive, move closer to sun, rename variables

# we NO LONGER assume sun is fixed, i.e. we compute gravitational pull of Earth on Sun
# due to conservation of angular momentum orbit lies in a plane, say the xy plane

T = 365*24*3600*4
dt = 3600
N = int(T/dt) + 1

G = 6.6734810e-11 # m^3 / (kg s^2)
m_sun = 1.989e30 # kg
m_earth = 5.972e24 # kg
R = 1.496e11 # m   (average radius of earth's orbit around sun)

r0_earth = [R, 0]
v0_earth = [0, 29.8e3]
r0_sun = [0, 0]
v0_sun = [0, 0]

r_earth = np.zeros((N,2))
v_earth = np.zeros((N,2))

r_sun = np.zeros((N,2))
v_sun = np.zeros((N,2)) 

r_earth[0] = r0_earth
v_earth[0] = v0_earth

r_sun[0] = r0_sun
v_sun[0] = v0_sun

norm = np.linalg.norm

def gravitational_acceleration(r1, r2, m2):

    return G*m2*(r2 - r1)/(norm(r2 - r1)**3)

a = gravitational_acceleration

for i in range(N-1):

    r_earth[i+1] = r_earth[i] + v_earth[i] * dt + 1/2*a(r_earth[i], r_sun[i], m_sun) * dt**2    
    r_sun[i+1] = r_sun[i] + v_sun[i] * dt + 1/2*a(r_sun[i], r_earth[i], m_earth) * dt**2


    v_earth[i+1] = v_earth[i] + 1/2*(a(r_earth[i], r_sun[i], m_sun) + a(r_earth[i+1], r_sun[i+1], m_sun) ) * dt
    v_sun[i+1] = v_sun[i] + 1/2*( a(r_sun[i], r_earth[i], m_earth) + a(r_sun[i+1], r_earth[i+1], m_earth) ) * dt

fig, ax = plt.subplots(figsize=(8,8))

#ax.set_aspect('equal', 'box')



from matplotlib.animation import FuncAnimation

(line_earth,) = ax.plot([], [])
(line_sun,) = ax.plot([], [])

ax.set_ylim(-1.5*R, 1.5*R)
ax.set_xlim(-1.5*R, 1.5*R)

steps = 10

frames = range(0, N, steps)


def update(frame):
    line_earth.set_data(r_earth[:frame, 0], r_earth[:frame, 1])
    line_sun.set_data(r_sun[:frame, 0], r_sun[:frame, 1])

    return line_earth, line_sun


ani = FuncAnimation(fig, update, frames=frames, interval=20)

plt.show()


