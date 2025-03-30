"""
passive_k_dynamics.py

Simulates passive membrane dynamics driven by potassium (K⁺) ions.
Uses Euler's method to solve a simple first-order ODE:
    dV/dt = (E_K - V) / tau + I_ext(t)

Author: Matthew J. Crossley
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# === Constants ===
K_in = 140  # mM
K_out = 5   # mM
E_K = 61 * np.log10(K_out / K_in)  # Nernst potential for K+
tau = 10  # ms

# === Simulation parameters ===
T = 300  # ms
dt = 0.1
t = np.arange(0, T, dt)
N = len(t)

# === Pulse parameters ===
pulse_amps = [0.5, 1.0, 2.0, 3.0]
colors = cm.Reds(np.linspace(0.4, 0.9, len(pulse_amps)))
pulse_width = N // 6
start = N // 3
end = start + pulse_width

# === Storage ===
V_traces = []
I_K_traces = []
I_ext_traces = []

# === Run simulation for each pulse amplitude ===
for amp in pulse_amps:
    I_ext = np.zeros(N)
    I_ext[start:end] = amp

    V = np.zeros(N)
    V[0] = E_K  # Start at equilibrium potential
    I_K = np.zeros(N)
    I_K[0] = (E_K - V[0]) / tau

    for i in range(1, N):
        I_K[i] = (E_K - V[i-1]) / tau
        dVdt = I_K[i] + I_ext[i-1]
        V[i] = V[i-1] + dVdt * dt

    V_traces.append(V)
    I_K_traces.append(I_K)
    I_ext_traces.append(I_ext)

# === Plotting ===
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# External inputs
for i, I_ext in enumerate(I_ext_traces):
    ax[0].plot(t, I_ext, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[0].set_ylabel('External Input (mV/ms)')
ax[0].set_title('Input Currents')

# Membrane potential
for i, V in enumerate(V_traces):
    ax[1].plot(t, V, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[1].set_ylabel('Membrane Potential (mV)')
ax[1].set_title('Passive K⁺ Response to Varying Inputs')

# Potassium current
for i, I_K in enumerate(I_K_traces):
    ax[2].plot(t, I_K, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[2].set_ylabel('K⁺ Current (mV/ms)\n(+ outward, - inward)')
ax[2].set_xlabel('Time (ms)')

ax[0].legend(loc='upper right')
plt.tight_layout()
plt.show()
