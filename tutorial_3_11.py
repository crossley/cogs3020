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

T[0, 0, 1] = 1  # press lever transition to state 1
T[0, 1, 2] = 1  # pull chain transition to state 2
T[0, 2, 3] = 1  # enter magazine terminal no reward

T[1, 0, 3] = 1  # press lever terminal no reward
T[1, 1, 3] = 1  # pull chain terminal no reward
T[1, 2, 4] = 1  # enter magazine terminal reward

T[2, 0, 3] = 1  # press lever terminal no reward
T[2, 1, 3] = 1  # pull chain terminal no reward
T[2, 2, 5] = 1  # enter magazine terminal reward

# state rewards
R = np.zeros(n_states)
R[4] = 1
R[5] = 1

# model of the environment
n = 10
T_hat = np.zeros((n_states, n_actions, n_states))
R_hat = np.zeros(n_states)
S_past = np.array([])
A_past = np.ones((n_states, n_actions)) * -1

# iterate over episodes
for e in range(n_episodes):

    # initialise s
    s = 0

    # iterate over steps per episodes
    for t in range(n_steps):

        # choose a from s using policy derived from q
        # here, we use softmax
        sm = np.exp(q[s, :]) / np.sum(np.exp(q[s, :]))
        a = np.random.choice(A, size=1, p=np.squeeze(sm))[0]

        # take action a, observe r, s'
        sprime = np.random.choice(S, size=1, p=np.squeeze(T[s, a, :]))[0]
        r = R[sprime]

        # update q-function
        q[s, a] += alpha * (r + np.max(q[sprime, :]) - q[s, a])

        # update models of the environment (tabular Dyna-Q p. 164)
        # assuming deterministic environment
        T_hat[s, a, sprime] = 1
        R_hat[sprime] = r

        # keep track of experienced states and actions
        S_past = np.append(S_past, [s])
        S_past = np.unique(S_past)
        A_past[s, a] = 1

        # Simulate experience
        for i in range(n):
            # pick a previously experienced state
            s = np.random.choice(S_past, size=1)[0].astype(int)

            # select an action previously taken from state s
            eligible_actions = A_past[s, :] == 1
            a = np.random.choice(np.where(eligible_actions)[0], size=1)[0]

            # simulate the outcome
            sprime_sim = np.random.choice(S,
                                          size=1,
                                          p=np.squeeze(T_hat[s, a, :]))[0]
            r = R[sprime_sim]

            # update the real Q function on the basis of the simulated outcome
            q[s, a] += alpha * (r + np.max(q[sprime_sim, :]) - q[s, a])

        # reset state
        s = sprime

        # stop if s is terminal
        if s == 3 or s == 4 or s == 5:
            break

fig, ax = plt.subplots(1, 1, squeeze=False)
ax[0, 0].imshow(q)
ax[0, 0].set_xlabel('action')
ax[0, 0].set_ylabel('state')
plt.show()
