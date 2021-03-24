import numpy as np
import matplotlib.pyplot as plt

n_trials = 1000

v_init = 0.5

p_reward = [0.0, 0.0, 0.0, 50, 100, 25, 150]

alpha = 0.01
gamma = 0.1
epsilon = 0.2

v = np.zeros((7, n_trials))
v[:, 0] = v_init


for i in range(0, n_trials - 1):

    s_trace = np.zeros(3)

    # step 0
    s = 0
    act = [1, 2]
    sm = np.exp(v[act, i]) / np.sum(np.exp(v[act, i]))
    a = (np.random.uniform() < sm[0]).astype(int)
    sprime = act[a]
    r = 0
    v[s, i + 1] = v[s, i] + alpha * (r + gamma * v[sprime, i] - v[s, i])
    s_trace[1] = sprime

    # step 1
    s = sprime
    act = [3, 4] if s == 1 else [5, 6]
    sm = np.exp(v[act, i]) / np.sum(np.exp(v[act, i]))
    a = (np.random.uniform() < sm[0]).astype(int)
    sprime = act[a]
    r = 0
    v[s, i + 1] = v[s, i] + alpha * (r + gamma * v[sprime, i] - v[s, i])
    s_trace[2] = sprime

    # step 2
    s = sprime
    r = np.random.normal(p_reward[sprime], 2)
    v[s, i + 1] = v[s, i] + alpha * (r - v[s, i])

    nots = np.setdiff1d(np.arange(0, 7, 1), s_trace)
    v[nots, i + 1] = v[nots, i]

fig, ax = plt.subplots(1, 3, squeeze=False)
ax[0, 0].plot(v[0, :], label='value 0')

ax[0, 1].plot(v[1, :], label='value 1')
ax[0, 1].plot(v[2, :], label='value 2')

ax[0, 2].plot(v[3, :], label='value 3')
ax[0, 2].plot(v[4, :], label='value 4')
ax[0, 2].plot(v[5, :], label='value 5')
ax[0, 2].plot(v[6, :], label='value 6')

ax[0, 0].legend()
ax[0, 1].legend()
ax[0, 2].legend()
plt.show()
