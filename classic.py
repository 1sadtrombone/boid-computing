import numpy as np
from matplotlib import pyplot as plt

def new_flock(count, lower_lim, upper_lim):
    width =  upper_lim - lower_lim
    return (lower_lim[:,np.newaxis] + np.random.rand(2, count) * width[:,np.newaxis])

show = True

limits = np.array([2000,2000])
tmax = 200
nboids = 50

rng = 200
cohere_param = 0.01
avoid_param = 10
avoid_c = 100
align_param = 0.1

max_speed = 25

xs = new_flock(nboids, np.array([0,0]), limits)
ps = new_flock(nboids, np.array([-20,-20]), np.array([20,20]))
fs = new_flock(nboids, np.array([0,0]), np.array([0,0]))
# axes (dim, boid)

if show:
    plt.ion()
    ax =  plt.axes(xlim=(0,limits[0]), ylim=(0,limits[1]))
        
for i in range(tmax):

    if show:
        ax.scatter(xs[0], xs[1], marker='^', edgecolor='k', lw=0.5)
        plt.pause(0.0001)
        plt.cla()
        ax.set_xlim((0,limits[0]))
        ax.set_ylim((0,limits[1]))

    # calculate stuff
    norms = np.linalg.norm(xs, axis=0)
    speeds = np.linalg.norm(ps, axis=0)

    # interact (NOT OPTIMAL)
    for i in range(nboids):

        other_xs = np.delete(xs, i, axis=1)
        other_ps = np.delete(ps, i, axis=1)
        
        dists = np.linalg.norm(other_xs - xs[:,i][:,np.newaxis], axis=0)

        count = np.sum(dists<=rng)

        if count > 0:

            # cohere
            cohere = np.sum(other_xs[:,dists<=rng],axis=1)
            cohere /= count
            fs[:,i] += (cohere - xs[:,i]) * cohere_param

            # avoid (inv. force)
            diffs = other_xs - xs[:,i][:,np.newaxis]
            avoid = np.sum(1/diffs[:,dists<=rng],axis=1)
            avoid /= count
            fs[:,i] += (-avoid) * avoid_param

            # align
            align = np.sum(other_ps[:,dists<=rng], axis=1)
            align /= count
            fs[:,i] += (align - ps[:,i]) * align_param


    # update positions and momenta
    xs += ps
    ps += fs

    # enforce max speed
    too_fast = np.where(speeds > max_speed)
    ps[:,too_fast] = (ps[:,too_fast] / speeds[too_fast]) * max_speed
            
    # enforce periodic bounds
    xs = xs % limits.reshape((-1,1))

plt.show()

