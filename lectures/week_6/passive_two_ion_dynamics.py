"""
passive_two_ion_dynamics.py

Simulates passive membrane dynamics with both potassium (K⁺) and sodium (Na⁺).
Each ion contributes to the total membrane current based on its reversal potential.

Equation:
    dV/dt = (1/τ) * (g_K * (E_K - V) + g_Na * (E_Na - V)) + I_ext(t)

Author: Matthew J. Crossley
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# === Constants ===
K_in = 140  # mM
K_out = 5   # mM
Na_in = 15  # mM
Na_out = 145  # mM
E_K = 61 * np.log10(K_out / K_in)
E_Na = 61 * np.log10(Na_out / Na_in)
g_K = 0.3   # mS/cm^2
g_Na = 0.1  # mS/cm^2
tau = 10    # ms (assumed uniform for simplicity)

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
I_Na_traces = []
I_ext_traces = []

# === Run simulation for each pulse amplitude ===
for amp in pulse_amps:
    I_ext = np.zeros(N)
    I_ext[start:end] = amp

    V = np.zeros(N)
    V[0] = (g_K * E_K + g_Na * E_Na) / (g_K + g_Na)  # Start at weighted resting potential
    I_K = np.zeros(N)
    I_Na = np.zeros(N)

    I_K[0] = g_K * (E_K - V[0])
    I_Na[0] = g_Na * (E_Na - V[0])

    for i in range(1, N):
        I_K[i] = g_K * (E_K - V[i-1])
        I_Na[i] = g_Na * (E_Na - V[i-1])
        dVdt = (I_K[i] + I_Na[i] + I_ext[i-1])  # Total current
        V[i] = V[i-1] + dVdt * dt

    V_traces.append(V)
    I_K_traces.append(I_K)
    I_Na_traces.append(I_Na)
    I_ext_traces.append(I_ext)

# === Plotting ===
fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

# External input
for i, I_ext in enumerate(I_ext_traces):
    ax[0].plot(t, I_ext, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[0].set_ylabel('External Input (mV/ms)')
ax[0].set_title('Input Currents')

# Membrane potential
for i, V in enumerate(V_traces):
    ax[1].plot(t, V, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[1].set_ylabel('Membrane Potential (mV)')
ax[1].set_title('Passive Na⁺ + K⁺ Response to Varying Inputs')

# K+ current
for i, I_K in enumerate(I_K_traces):
    ax[2].plot(t, I_K, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[2].set_ylabel('K⁺ Current (mV/ms)\n(+ outward, - inward)')

# Na+ current
for i, I_Na in enumerate(I_Na_traces):
    ax[3].plot(t, I_Na, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[3].set_ylabel('Na⁺ Current (mV/ms)\n(+ outward, - inward)')
ax[3].set_xlabel('Time (ms)')

ax[0].legend(loc='upper right')
plt.tight_layout()
plt.show()
