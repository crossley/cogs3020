import numpy as np
import matplotlib.pyplot as plt

# TODO: Add the indirect pathway

tau = 0.1
T = 300
t = np.arange(0, T, tau)
n = t.shape[0]

# response of each spike on post synaptic membrane v
psp_amp = 1e5
psp_decay = 10

# allocate memory for network elements
g_in = np.zeros(n)
spike_in = np.zeros(n)
spike_in[1000:1500:20] = 1

w_in_spn = 0.5
w_spn_gpi = 2.0
w_gpi_thal = 2.0
w_thal_motor = 0.5

v_spn = np.zeros(n)
u_spn = np.zeros(n)
g_spn = np.zeros(n)
spike_spn = np.zeros(n)
v_spn[0] = -80
E_spn = 0

v_gpi = np.zeros(n)
u_gpi = np.zeros(n)
g_gpi = np.zeros(n)
spike_gpi = np.zeros(n)
v_gpi[0] = -60
E_gpi = 2000

v_thal = np.zeros(n)
u_thal = np.zeros(n)
g_thal = np.zeros(n)
spike_thal = np.zeros(n)
v_thal[0] = -60
E_thal = 2000

v_motor = np.zeros(n)
u_motor = np.zeros(n)
g_motor = np.zeros(n)
spike_motor = np.zeros(n)
v_motor[0] = -60
E_motor = 10

for i in range(1, n):

    dt = t[i] - t[i - 1]

    # external input
    dgdt_in = (-g_in[i - 1] + psp_amp * spike_in[i - 1]) / psp_decay
    g_in[i] = g_in[i - 1] + dgdt_in * dt


    # spn
    # striatal project neuron
    C = 50; vr = -80; vt = -25; vpeak = 40;
    k = 1; a = 0.01; b = -20; c = -55; d = 150;
    dvdt_spn = (k * (v_spn[i - 1] - vr) * (v_spn[i - 1] - vt) - u_spn[i - 1] +
                w_in_spn * g_in[i - 1] + E_spn) / C
    dudt_spn = a * (b * (v_spn[i - 1] - vr) - u_spn[i - 1])
    dgdt_spn = (-g_spn[i - 1] + psp_amp * spike_spn[i - 1]) / psp_decay
    v_spn[i] = v_spn[i - 1] + dvdt_spn * dt
    u_spn[i] = u_spn[i - 1] + dudt_spn * dt
    g_spn[i] = g_spn[i - 1] + dgdt_spn * dt
    if v_spn[i] >= vpeak:
        v_spn[i - 1] = vpeak
        v_spn[i] = c
        u_spn[i] = u_spn[i] + d
        spike_spn[i] = 1

    # gpi
    # regular spiking (rs)
    C = 100; vr = -60; vt = -40; vpeak = 35;
    a = 0.03; b = -2; c = -50; d = 100; k = 0.7;
    dvdt_gpi = (k * (v_gpi[i - 1] - vr) * (v_gpi[i - 1] - vt) - u_gpi[i - 1] -
                w_spn_gpi * g_spn[i - 1] + E_gpi) / C
    dudt_gpi = a * (b * (v_gpi[i - 1] - vr) - u_gpi[i - 1])
    dgdt_gpi = (-g_gpi[i - 1] + psp_amp * spike_gpi[i - 1]) / psp_decay
    v_gpi[i] = v_gpi[i - 1] + dvdt_gpi * dt
    u_gpi[i] = u_gpi[i - 1] + dudt_gpi * dt
    g_gpi[i] = g_gpi[i - 1] + dgdt_gpi * dt
    if v_gpi[i] >= vpeak:
        v_gpi[i - 1] = vpeak
        v_gpi[i] = c
        u_gpi[i] = u_gpi[i] + d
        spike_gpi[i] = 1

    # thal
    dvdt_thal = (k * (v_thal[i - 1] - vr) * (v_thal[i - 1] - vt) - u_thal[i - 1] -
                 w_gpi_thal * g_gpi[i - 1] + E_thal) / C
    dudt_thal = a * (b * (v_thal[i - 1] - vr) - u_thal[i - 1])
    dgdt_thal = (-g_thal[i - 1] + psp_amp * spike_thal[i - 1]) / psp_decay
    v_thal[i] = v_thal[i - 1] + dvdt_thal * dt
    u_thal[i] = u_thal[i - 1] + dudt_thal * dt
    g_thal[i] = g_thal[i - 1] + dgdt_thal * dt
    if v_thal[i] >= vpeak:
        v_thal[i - 1] = vpeak
        v_thal[i] = c
        u_thal[i] = u_thal[i] + d
        spike_thal[i] = 1

    # motor
    dvdt_motor = (k * (v_motor[i - 1] - vr) * (v_motor[i - 1] - vt) - u_motor[i - 1] +
                  w_thal_motor * g_thal[i - 1] + np.random.normal(E_motor, 10)) / C
    dudt_motor = a * (b * (v_motor[i - 1] - vr) - u_motor[i - 1])
    dgdt_motor = (-g_motor[i - 1] + psp_amp * spike_motor[i - 1]) / psp_decay
    v_motor[i] = v_motor[i - 1] + dvdt_motor * dt
    u_motor[i] = u_motor[i - 1] + dudt_motor * dt
    g_motor[i] = g_motor[i - 1] + dgdt_motor * dt
    if v_motor[i] >= vpeak:
        v_motor[i - 1] = vpeak
        v_motor[i] = c
        u_motor[i] = u_motor[i] + d
        spike_motor[i] = 1

fig, ax = plt.subplots(5, 1, squeeze=False)
ax[0, 0].plot(t, g_in)
ax[1, 0].plot(t, v_spn)
ax[2, 0].plot(t, v_gpi)
ax[3, 0].plot(t, v_thal)
ax[4, 0].plot(t, v_motor)
plt.tight_layout()
plt.show()
