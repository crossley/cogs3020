"""
active_na_conductance.py

Simulates membrane dynamics with voltage-dependent Na⁺ channels using
both activation (m) and inactivation (h) gating variables.

Demonstrates how increasing depolarizing input leads to larger Na⁺ currents
that rise and then fall due to inactivation, even during a sustained input.

Also includes outward K⁺ current and shows membrane potential and Na⁺ channel
dynamics over time for a range of input amplitudes.

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
g_Na_max = 1.2
tau = 10

# === Activation (m) and Inactivation (h) Gate Dynamics ===
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10 + 1e-6))
def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)
def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))

# === Simulation parameters ===
T = 300
dt = 0.1
t = np.arange(0, T, dt)
N = len(t)

# === Pulse parameters ===
pulse_amps = [0.5, 5.0, 20.0, 50.0]
colors = cm.Reds(np.linspace(0.4, 0.9, len(pulse_amps)))
pulse_width = N // 6
start = N // 3
end = start + pulse_width

# === Storage ===
V_traces, I_K_traces, I_Na_traces = [], [], []
m_traces, h_traces, g_Na_traces, I_ext_traces = [], [], [], []

# === Run simulation for each input ===
for amp in pulse_amps:
    I_ext = np.zeros(N)
    I_ext[start:end] = amp

    V = np.zeros(N)
    V[0] = (g_K * E_K + g_Na_max * 0.05 * E_Na) / (g_K + g_Na_max * 0.05)
    m = np.zeros(N)
    h = np.ones(N)
    g_Na = np.zeros(N)
    I_K = np.zeros(N)
    I_Na = np.zeros(N)

    m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
    h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
    g_Na[0] = g_Na_max * m[0]**3 * h[0]
    I_K[0] = g_K * (E_K - V[0])
    I_Na[0] = g_Na[0] * (E_Na - V[0])

    for i in range(1, N):
        a_m, b_m = alpha_m(V[i-1]), beta_m(V[i-1])
        a_h, b_h = alpha_h(V[i-1]), beta_h(V[i-1])
        tau_m = 1 / (a_m + b_m)
        tau_h = 1 / (a_h + b_h)
        m_inf = a_m * tau_m
        h_inf = a_h * tau_h

        dm_dt = (m_inf - m[i-1]) / tau_m
        dh_dt = (h_inf - h[i-1]) / tau_h
        m[i] = m[i-1] + dm_dt * dt
        h[i] = h[i-1] + dh_dt * dt

        g_Na[i] = g_Na_max * m[i]**3 * h[i]
        I_K[i] = g_K * (E_K - V[i-1])
        I_Na[i] = g_Na[i] * (E_Na - V[i-1])
        dVdt = I_K[i] + I_Na[i] + I_ext[i-1]
        V[i] = V[i-1] + dVdt * dt

    V_traces.append(V)
    I_K_traces.append(I_K)
    I_Na_traces.append(I_Na)
    m_traces.append(m)
    h_traces.append(h)
    g_Na_traces.append(g_Na)
    I_ext_traces.append(I_ext)

# === Plotting ===
fig, ax = plt.subplots(7, 1, figsize=(10, 11), sharex=True)

# External input
for i, I_ext in enumerate(I_ext_traces):
    ax[0].plot(t, I_ext, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[0].set_ylabel('External Input')
ax[0].set_title('Input Currents')

# Membrane potential
for i, V in enumerate(V_traces):
    ax[1].plot(t, V, color=colors[i])
ax[1].set_ylabel('Membrane Potential (mV)')
ax[1].set_title('Na+ Activation + Inactivation')

# K⁺ current
for i, I_K in enumerate(I_K_traces):
    ax[2].plot(t, I_K, color=colors[i])
ax[2].set_ylabel('K+ Current \n + outward \n - inward)')

# Na⁺ current
for i, I_Na in enumerate(I_Na_traces):
    ax[3].plot(t, I_Na, color=colors[i])
ax[3].set_ylabel('Na+ Current \n + outward \n - inward')

# Na⁺ conductance
for i, g in enumerate(g_Na_traces):
    ax[4].plot(t, g, color=colors[i])
ax[4].set_ylabel('g_Na')

# Na⁺ activation gate
for i, m in enumerate(m_traces):
    ax[5].plot(t, m, color=colors[i])
ax[5].set_ylabel('Na+ Activation m(t)')
ax[5].set_xlabel('Time')

# Na⁺ inactivation gate
for i, h in enumerate(h_traces):
    ax[6].plot(t, h, color=colors[i])
ax[6].set_ylabel('Na+ Inactivation h(t)')
ax[6].set_xlabel('Time')

ax[0].legend(loc='upper right')
plt.tight_layout()
plt.show()

