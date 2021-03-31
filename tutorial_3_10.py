import numpy as np
import matplotlib.pyplot as plt

n_episodes = 100
n_steps = 10

n_states = 6
n_actions = 3

alpha = 0.1

# initialise q(s,a)
q = np.ones((n_states, n_actions)) * 0.5

# states
S = np.arange(0, 6, 1)

# Actions
A = np.array([0, 1, 2])

# state transition probabilities
T = np.zeros((n_states, n_actions, n_states))

T[0, 0, 1] = 1 # press lever transition to state 1
T[0, 1, 2] = 1 # pull chain transition to state 2
T[0, 2, 3] = 1 # enter magazine terminal no reward

T[1, 0, 3] = 1 # press lever terminal no reward
T[1, 1, 3] = 1 # pull chain terminal no reward
T[1, 2, 4] = 1 # enter magazine terminal reward

T[2, 0, 3] = 1 # press lever terminal no reward
T[2, 1, 3] = 1 # pull chain terminal no reward
T[2, 2, 5] = 1 # enter magazine terminal reward

# state rewards
R = np.zeros(n_states)
R[4] = 1
R[5] = 1

# iterate over episodes
for e in range(n_episodes):

    # initialise s
    s = 0

    # iterate over steps per episodes
    for t in range(n_steps):

        # choose a from s using policy derived from q
        # here, we use softmax
        sm = np.exp(q[s, :]) / np.sum(np.exp(q[s, :]))
        a = np.random.choice(A, size=1, p=np.squeeze(sm))

        # take action a, observe r, s'
        sprime = np.random.choice(S, size=1, p=np.squeeze(T[s, a, :]))
        r = R[sprime]

        # update q-function
        q[s, a] += alpha * (r + np.max(q[sprime, :]) - q[s, a])

        # reset state
        s = sprime

        # stop if s is terminal
        if s == 3 or s == 4 or s == 5:
            break


fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0,0].imshow(q)
ax[0, 0].set_xlabel('action')
ax[0, 0].set_ylabel('state')
plt.show()
