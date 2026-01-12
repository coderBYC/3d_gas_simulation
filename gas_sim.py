import matplotlib.pyplot as plt
import numpy as np 
from itertools import product, combinations 
from matplotlib.animation import FuncAnimation
fig = plt.figure(figsize=(6,6)) # Use subplot_kw for current versions of Matplotlib 
ax = fig.add_subplot(111, projection='3d') 
# Define cube range 
r = [-1, 1] 
# Get all combinations of coordinates 
points = np.array(list(product(r, r, r))) 
# Draw the edges 
for s, e in combinations(points, 2): 
    if np.sum(np.abs(s - e)) == r[1] - r[0]: 
        ax.plot(*zip(s, e), color="b") 
ax.set_xlim(r)
ax.set_ylim(r)
ax.set_zlim(r)
ax.set_xlabel('X axis') 
ax.set_ylabel('Y axis') 
ax.set_zlabel('Z axis') 
ax.set_title('Gas Particle Simulation in a Cube') 

# Number of particles
N = 50
# Mass of particles
m = 1.0
# Add particles
pos = np.random.uniform(-0.9, 0.9, (N, 3))
v = np.random.normal(0, 0.3, (N, 3))
dt = 0.05
def resolveCollision():
    global pos,v
    for i in range(N):
        for j in range(i+1, N):
            delta = pos[i] - pos[j]
            dist = np.linalg.norm(delta)
            if dist <= 0.06:
                print(f"Collision between particle {i} and {j}")
                n= delta / dist
                vn = np.dot(v[i] - v[j], n)
                if vn > 0:
                    continue
                v1x = np.dot(n,v[i])
                v2x= np.dot(n,v[j])
                v1y = v[i] - v1x*n
                v2y = v[j] - v2x*n
                v1n = v2x*n
                v2n = v1x*n
                v[i] = v1n + v1y
                v[j] = v2n + v2y
                overlap = 0.06 - dist
                pos[i] -= n*(overlap/2)
                pos[j] += n*(overlap/2)

point, = ax.plot(pos[:,0], pos[:,1], pos[:,2], 'ro', markersize=8)
def update(frame):
    global pos,v
    pos += v *dt
    resolveCollision()
    for i in range(3):
        lower = pos[:, i] <= -1
        upper = pos[:, i] >= 1

        v[lower, i] *= -1
        v[upper, i] *= -1

        pos[lower, i] = -1
        pos[upper, i] = 1
    point.set_data(pos[:, 0], pos[:, 1])
    point.set_3d_properties(pos[:, 2])
    KE = 0.5 * m * np.sum(np.linalg.norm(v, axis=1)**2)  # total kinetic energy
    T = (2/3) * KE / N  # temperature proportional to average kinetic energy per particle
    ax.set_title(f'Gas Particle Simulation in a Cube\nTemperature: {T:.4f}')
    return point,
ani = FuncAnimation(fig, update, frames=300, interval=30)
plt.show()