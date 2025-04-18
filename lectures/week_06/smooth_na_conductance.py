"""
smooth_na_conductance.py

Simulates membrane voltage and Na⁺ conductance using a smooth,
voltage-gated activation function. Na⁺ conductance increases
when membrane potential exceeds threshold and decays over time.

Author: Matthew J. Crossley
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# === Constants ===
K_in = 140
K_out = 5
Na_in = 15
Na_out = 145
E_K = 61 * np.log10(K_out / K_in)
E_Na = 61 * np.log10(Na_out / Na_in)
g_K = 0.3
g_Na_max = 1.2  # Max Na conductance
tau = 10

# === Sigmoid activation function for Na conductance ===
def sigmoid(V, V_half=-40, slope=6):
    return 1 / (1 + np.exp(-(V - V_half) / slope))

# === Simulation parameters ===
T = 300
dt = 0.1
t = np.arange(0, T, dt)
N = len(t)

# === Pulse parameters ===
pulse_amps = [0.5, 1.0, 2.0, 4.0, 4.5, 4.865, 5]
colors = cm.Reds(np.linspace(0.4, 0.9, len(pulse_amps)))
pulse_width = N // 6
start = N // 3
end = start + pulse_width

# === Storage ===
V_traces = []
I_K_traces = []
I_Na_traces = []
g_Na_traces = []
I_ext_traces = []

# === Run simulation for each pulse amplitude ===
for amp in pulse_amps:
    I_ext = np.zeros(N)
    I_ext[start:end] = amp

    V = np.zeros(N)
    V[0] = (g_K * E_K + g_Na_max * sigmoid(-65) * E_Na) / (g_K + g_Na_max * sigmoid(-65))
    I_K = np.zeros(N)
    I_Na = np.zeros(N)
    g_Na = np.zeros(N)

    g_Na[0] = g_Na_max * sigmoid(V[0])
    I_K[0] = g_K * (E_K - V[0])
    I_Na[0] = g_Na_max * sigmoid(V[0]) * (E_Na - V[0])

    for i in range(1, N):
        g_Na[i] = g_Na_max * sigmoid(V[i-1])
        I_K[i] = g_K * (E_K - V[i-1])
        I_Na[i] = g_Na[i] * (E_Na - V[i-1])
        dVdt = I_K[i] + I_Na[i] + I_ext[i-1]
        V[i] = V[i-1] + dVdt * dt

    V_traces.append(V)
    I_K_traces.append(I_K)
    I_Na_traces.append(I_Na)
    g_Na_traces.append(g_Na)
    I_ext_traces.append(I_ext)

# === Plotting ===
fig, ax = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

# External input
for i, I_ext in enumerate(I_ext_traces):
    ax[0].plot(t, I_ext, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[0].set_ylabel('External Input (mV/ms)')
ax[0].set_title('Input Currents')

# Membrane potential
for i, V in enumerate(V_traces):
    ax[1].plot(t, V, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[1].set_ylabel('Membrane Potential (mV)')
ax[1].set_title('Na⁺ Activation with Smooth Conductance')

# K⁺ current
for i, I_K in enumerate(I_K_traces):
    ax[2].plot(t, I_K, color=colors[i])
ax[2].set_ylabel('K⁺ Current (mV/ms)\n(+ outward, - inward)')

# Na⁺ current
for i, I_Na in enumerate(I_Na_traces):
    ax[3].plot(t, I_Na, color=colors[i])
ax[3].set_ylabel('Na⁺ Current (mV/ms)\n(+ outward, - inward)')

# Na⁺ conductance
for i, g_Na in enumerate(g_Na_traces):
    ax[4].plot(t, g_Na, color=colors[i])
ax[4].set_ylabel('g_Na (mS/cm²)')
ax[4].set_xlabel('Time (ms)')

ax[0].legend(loc='upper right')
plt.tight_layout()
plt.show()

