import numpy as np
import matplotlib.pyplot as plt

tau = 0.1
T = 2000
t = np.arange(0, T, tau)
n_steps = t.shape[0]
# NOTE: trials
n_trials = 10

# Cells: CTX, D1, D2, GPi, GPe, Thal, STN
# CTX -> D1 -> GPi -> Thal -> CTX
# CTX -> D2 -> GPe -> GPi -> Thal -> CTX
# CTX -> STN -> GPi
# STN <-> GPe

# # striatal projection neuron
# C = 50; vr = -80; vt = -25; vpeak = 40;
# a = 0.01; b = -20; c = -55; d = 150; k = 1;

# # regular spiking neuron
# C = 100; vr = -60; vt = -40; vpeak = 35;
# a = 0.03; b = -2; c = -50; d = 100; k = 0.7;

iz_params = np.array([
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # viusal ctx (rs) 0
    [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # d1 (spn) 1
    [50, -80, -25, 40, 0.01, -20, -55, 150, 1],  # d2 (spn) 2
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # gpi (rs) 3
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # gpe (rs) 4
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # thal (rs) 5
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7],  # stn (rs) 6
    [100, -60, -40, 35, 0.03, -2, -50, 100, 0.7]  # motor ctx (rs) 7
])

# NOTE: baseline firing
E_mu = np.array([0, 0, 0, 300, 300, 300, 0, 50])
E_sd = np.array([0, 0, 0, 0, 0, 0, 0, 100])

n_cells = iz_params.shape[0]

# response of each spike on post synaptic membrane v
psp_amp = 1e3
psp_decay = 100

# allocate memory for each neuron
v = np.zeros((n_cells, n_steps))
u = np.zeros((n_cells, n_steps))
g = np.zeros((n_cells, n_steps))
spike = np.zeros((n_cells, n_steps))
v[:, 0] = iz_params[:, 1] + np.random.rand(n_cells) * 100

# connection weight matrix
w = np.zeros((n_cells, n_cells))

# direct pathway
w[0, 1] = 100  # NOTE: start this off small to allow for learning
w[1, 3] = -1 * 100

# indirect pathway
# w[0, 2] = 1 * 100
# w[2, 4] = -1 * 100
# w[4, 3] = -1 * 25

# hyperdirect pathway
# w[0, 6] = 1 * 50 * 0
# w[6, 3] = 1

# stn-gpe feedback
# w[6, 4] = 1
# w[4, 6] = -1 * 50

# gpi-thal
w[3, 5] = -1 * 100

# thal-motor
w[5, 7] = 100

# fb from thal back to input ctx
# w[5, 0] = 1

# input into cells from other cells
I_net = np.zeros((n_cells, n_steps))

# define input signal (artificial input into ctx)
I_in = np.zeros(n_steps)
I_in[5000:] = 5e1

# NOTE: response threshold
resp_thresh = 25

# NOTE: predicted reward
pr = 0

# NOTE: records
w_rec = np.zeros(n_trials)
resp_rec = np.zeros(n_trials)
pr_rec = np.zeros(n_trials)

# NOTE: iterate over trials
for trl in range(n_trials):

    # NOTE: initialise response to zero
    resp = 0

    for i in range(1, n_steps):

        dt = t[i] - t[i - 1]

        I_net = np.zeros((n_cells, n_steps))
        for jj in range(n_cells):
            for kk in range(n_cells):
                if jj != kk:
                    I_net[jj, i - 1] += w[kk, jj] * g[kk, i - 1]
                if jj == 0:
                    I_net[jj, i - 1] += I_in[i - 1]

            C = iz_params[jj, 0]
            vr = iz_params[jj, 1]
            vt = iz_params[jj, 2]
            vpeak = iz_params[jj, 3]
            a = iz_params[jj, 4]
            b = iz_params[jj, 5]
            c = iz_params[jj, 6]
            d = iz_params[jj, 7]
            k = iz_params[jj, 8]

            # NOTE: The np.random.normal() below is new
            dvdt = (k * (v[jj, i - 1] - vr) *
                    (v[jj, i - 1] - vt) - u[jj, i - 1] + I_net[jj, i - 1] +
                    np.random.normal(E_mu[jj], E_sd[jj])) / C
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

        # NOTE: respond if motor unit crosses resp_thresh
        # NOTE: 5000 is Ugly hack to cut out the annoying initial spikes
        if (g[7, i] > resp_thresh) and i > 5000:
            resp = 1
            break

    # NOTE: force exploratory responses (e.g., epsilon greedy)
    if np.random.uniform() > 0.8:
        resp = 1

    # NOTE: compute rewards and prediction errors
    if resp == 1:
        r = 1
    else:
        r = 0

    rpe = r - pr
    pr += 0.1 * rpe # NOTE: this learning is a free parameter

    # NOTE: update weight (learning rate is another free parameter)
    pre = g[0, :].sum()
    post = g[1, :].sum()
    w[0, 1] += 0.1 * pre * post * rpe

    # NOTE: keep records
    w_rec[trl] = w[0, 1]
    resp_rec[trl] = resp
    pr_rec[trl] = pr


fig, ax = plt.subplots(1, 3, squeeze=False)
ax[0, 0].plot(pr_rec)
ax[0, 1].plot(w_rec)
ax[0, 2].plot(resp_rec)
plt.show()


fig, ax = plt.subplots(3, 5, squeeze=False)
# ctx
ax[1, 0].set_title('ctx')
ax1 = ax[1, 0]
ax2 = ax1.twinx()
ax1.plot(t, v[0, :], 'C0')
ax2.plot(t, g[0, :], 'C1')
# stn
ax[0, 1].set_title('stn')
ax1 = ax[0, 1]
ax2 = ax1.twinx()
ax1.plot(t, v[6, :], 'C0')  # stn
ax2.plot(t, g[6, :], 'C1')  # stn
# d1
ax[1, 1].set_title('d1')
ax1 = ax[1, 1]
ax2 = ax1.twinx()
ax1.plot(t, v[1, :], 'C0')  # d1
ax2.plot(t, g[1, :], 'C1')  # d1
# d2
ax[2, 1].set_title('d2')
ax1 = ax[2, 1]
ax2 = ax1.twinx()
ax1.plot(t, v[2, :], 'C0')  # d2
ax2.plot(t, g[2, :], 'C1')  # d2
# gpi
ax[1, 2].set_title('gpi')
ax1 = ax[1, 2]
ax2 = ax1.twinx()
ax1.plot(t, v[3, :], 'C0')  # gpi
ax2.plot(t, g[3, :], 'C1')  # gpi
# gpe
ax[2, 2].set_title('gpe')
ax1 = ax[2, 2]
ax2 = ax1.twinx()
ax1.plot(t, v[4, :], 'C0')  # gpe
ax2.plot(t, g[4, :], 'C1')  # gpe
# thal
ax[1, 3].set_title('thal')
ax1 = ax[1, 3]
ax2 = ax1.twinx()
ax1.plot(t, v[5, :], 'C0')  # thal
ax2.plot(t, g[5, :], 'C1')  # thal
# motor
ax[1, 4].set_title('motor')
ax1 = ax[1, 4]
ax2 = ax1.twinx()
ax1.plot(t, v[7, :], 'C0')  # motor
ax2.plot(t, g[7, :], 'C1')  # motor
# plt.tight_layout()
plt.show()
