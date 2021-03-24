import numpy as np
import matplotlib.pyplot as plt

n_trials = 1000

v_init = 0.5
p_reward_1 = 0.6
p_reward_2 = 0.4
alpha = 0.01
epsilon = 0.2

v = np.zeros((2, n_trials))
v[:, 0] = v_init

a = np.zeros(n_trials)
r = np.zeros(n_trials)

for i in range(0, n_trials - 1):

    # # action selection - greedy epsilon
    # if np.random.uniform() < epsilon:
    #     a[i] = np.round(np.random.uniform())
    # else:
    #     a[i] = np.argmax(v[:, i])

    # action selection - softmax
    sm = np.exp(v[:, i]) / np.sum(np.exp(v[:, i]))
    if np.random.uniform() < sm[0]:
        a[i] = 0

        # reward
        r[i] = np.random.normal(p_reward_1, 2)

        # reward prediction error
        delta = r[i] - v[0, i]

        # value update
        v[0, i + 1] = v[0, i] + alpha * delta
        v[1, i + 1] = v[1, i]

    else:
        a[i] = 1

        # reward
        r[i] = np.random.normal(p_reward_2, 2)

        # reward prediction error
        delta = r[i] - v[1, i]

        # value update
        v[1, i + 1] = v[1, i] + alpha * delta
        v[0, i + 1] = v[0, i]

fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0, 0].plot(v[0, :], label='value 1')
ax[0, 0].plot(v[1, :], label='value 2')
plt.legend()
plt.show()
