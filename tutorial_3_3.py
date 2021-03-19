import numpy as np
import matplotlib.pyplot as plt

tau = 0.1
T = 100
t = np.arange(0, T, tau)
n_steps = t.shape[0]
n_cells = 100

# striatal projection neurons
C = 50
vr = -80
vt = -25
vpeak = 40
k = 1
a = 0.01
b = -20
c = -55
d = 150

# response of each spike on post synaptic membrane v
psp_amp = 1e5
psp_decay = 10

# allocate memory for each neuron
v = np.zeros((n_cells, n_steps))
u = np.zeros((n_cells, n_steps))
g = np.zeros((n_cells, n_steps))
spike = np.zeros((n_cells, n_steps))
v[:, 0] = vr + np.random.rand(n_cells) * 100

# connection weight matrix
w = np.random.rand(n_cells, n_cells) * 0.05

# input into cells from other cells
I_net = np.zeros((n_cells, n_steps))

# define input signal
I_in = np.zeros(n_steps)
I_in[100:900] = 1e3

for i in range(1, n_steps):

    dt = t[i] - t[i - 1]

    I_net = np.zeros((n_cells, n_steps))
    for jj in range(n_cells):
        for kk in range(n_cells):
            if jj != kk:
                I_net[jj, i - 1] += -w[jj, kk] * g[kk, i - 1]

        dvdt = (k * (v[jj, i - 1] - vr) * (v[jj, i - 1] - vt) - u[jj, i - 1] +
                I_net[jj, i - 1] + I_in[i - 1]) / C
        dudt = a * (b * (v[jj, i - 1] - vr) - u[jj, i - 1])
        dgdt = (-g[jj, i - 1] + psp_amp * spike[jj, i - 1]) / psp_decay

        v[jj, i] = v[jj, i - 1] + dvdt * dt
        u[jj, i] = u[jj, i - 1] + dudt * dt
        g[jj, i] = g[jj, i - 1] + dgdt * dt

        if v[jj, i] >= vpeak:
            v[jj, i - 1] = vpeak
            v[jj, i] = c
            u[jj, i] = u[jj, i] + d
            spike[jj, i] = 1

fig, ax = plt.subplots(2, 1, squeeze=False)
ax[0, 0].imshow(g)
ax[1, 0].plot(t, v[0, :])
plt.show()
