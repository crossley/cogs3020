import numpy as np
import matplotlib.pyplot as plt

n_episodes = 100
n_steps = 10

n_states = 6
n_actions = 3

alpha = 0.1

# initialise q(s,a)
q = np.ones((n_states, n_actions)) * 0.5

# iterate over episodes
for e in range(n_episodes):

    # initialise s
    s = 0

    # iterate over steps per episodes
    for t in range(n_steps):

        # choose a from s using policy derived from q
        # here, we use softmax
        sm = np.exp(q[s, :]) / np.sum(np.exp(q[s, :]))
        a = np.random.choice([0, 1, 2], size=1, p=np.squeeze(sm))

        # take action a, observe r, s'

        if s==0:
            # press lever
            if a == 0:
                r = 0
                sprime = 1

            # pull chain
            elif a == 1:
                r = 0
                sprime = 2

            # enter magazine
            elif a == 2:
                r = 0
                sprime = 3

        elif s==1:
            # press lever
            if a == 0:
                r = 0
                sprime = 3

            # pull chain
            elif a == 1:
                r = 0
                sprime = 3

            # enter magazine
            elif a == 2:
                r = 1
                sprime = 4

        elif s==2:
            # press lever
            if a == 0:
                r = 0
                sprime = 3

            # pull chain
            elif a == 1:
                r = 0
                sprime = 3

            # enter magazine
            elif a == 2:
                r = 1
                sprime = 5

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
