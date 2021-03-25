import numpy as np
import matplotlib.pyplot as plt

n_trials = 1000
n_steps = 100

v_init = 0.0

alpha = 0.1
gamma = 1

v = np.zeros((n_steps, n_trials))
v[:, 0] = v_init

for n in range(1, n_trials):
    for t in range(n_steps - 1):
        s = t
        sprime = t + 1
        r = 1 if s == (n_steps - 2) else 0
        v[s, n] = v[s, n - 1] + alpha * (r + gamma * v[sprime, n - 1] - v[s, n - 1])

fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0, 0].imshow(v, aspect='auto')
ax[0, 0].set_xlabel('trial')
ax[0, 0].set_ylabel('time step (state)')
plt.show()
