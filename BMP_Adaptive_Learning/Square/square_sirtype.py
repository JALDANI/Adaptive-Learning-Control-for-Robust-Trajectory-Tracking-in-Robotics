import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, pinv
import math

# =============================================================================
# Plot/Export Settings (similar to MATLAB "s" structure)
# =============================================================================
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# =============================================================================
# Initialization
# =============================================================================
num_trials = 100
ts = 0.01                      # Sampling time interval
paradigm_trials = 7 * num_trials  # Total number of trials
time_total = 1                 # Total simulation time (s)
duration = time_total
export_plots_to_pdf = True

# =============================================================================
# System Definition (2R Robot)
# =============================================================================
m1 = 0.2       # mass of link 1
m2 = 0.2       # mass of link 2
lc1 = 0.1      # midlength of link 1
lc2 = 0.1      # midlength of link 2
I1 = (1/3)*m1*(2*lc1)   # Moment of inertia of link 1
I2 = (1/3)*m2*(2*lc2)   # Moment of inertia of link 2
Q1 = math.pi/4          # Equilibrium point of q1
Q2 = -math.pi/2         # Equilibrium point of q2
d1 = 0.3                # Damping 1
d2 = 0.3                # Damping 2

denom = (I1*I2 + 4*lc1*lc2**2*m2**2 + 4*I2*lc1*m2 + I2*lc1**2*m1 +
         I1*lc2**2*m2 + lc1**2*lc2**2*m1*m2 - 
         4*lc1**2*lc2**2*m2**2*(math.cos(Q2)**2))

Dinv = np.array([
    [(m2*lc2**2 + I2) / denom,
     -(m2*lc2**2 + 2*lc1*m2*math.cos(Q2)*lc2) / denom],
    [-(m2*lc2**2 + 2*lc1*m2*math.cos(Q2)*lc2) / denom,
     (m1*lc1**2 + 4*m2*math.cos(Q2)*lc1*lc2 + 4*m2*lc1 + m2*lc2**2 + I1 + I2) / denom]
])

a11 = np.zeros((2,2))
a12 = np.eye(2)
a21 = np.zeros((2,2))
a22 = -Dinv @ np.diag([d1, d2])
b11 = np.zeros((2,2))
b12 = Dinv @ np.eye(2)

Ac = np.block([[a11, a12],
               [a21, a22]])
Bc = np.vstack([b11, b12])
C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

n = Ac.shape[0]  # State dimension = 4
A = 0.5 * (np.eye(n) + Ac * ts)
B = Bc

# =============================================================================
# Controller Equations Parameterization
# =============================================================================
lambda_ILC = 1       # Determination factor
lambda_pred = 0.7
lambda_FF = 0.7      # Confidence on the internal learnt model
gamma_ILC = 0.1      # Learning rate for ILC
gamma_pred = 0.2     # Learning rate for prediction
gamma_FF = 0.1       # Learning rate for feedforward
nx = 0.01            # Noise on state update
ny = 0.01            # Noise on measurement

# =============================================================================
# Perturbation Definition
# =============================================================================
beta = -0.1       # Feedforward perturbation
alpha = 15        # Velocity-dependent perturbation
th = 0.5236       # Rotation perturbation angle (approx 30°)
# Enable flags
enableVMR = 0
enableFFPT = 0
enableVLPT = 0

# =============================================================================
# Trajectory Generation (Square Trajectory)
# =============================================================================
xi, yi = 0, 0  
xf, yf = 1, 1  
total_steps = len(np.arange(0, time_total, ts))
yd1 = np.zeros((2, total_steps))
ydx = np.zeros((2, total_steps))
yd = np.zeros((2, total_steps))

duration_seg = time_total / 4  # Each segment duration

t_idx = 0
for loop_time in np.arange(0, time_total, ts):
    t_idx += 1
    segment = int(np.floor(loop_time / duration_seg)) + 1
    if segment > 4:
        segment = 4
    segment_time = loop_time % duration_seg
    s = 10*(segment_time/duration_seg)**3 - 15*(segment_time/duration_seg)**4 + 6*(segment_time/duration_seg)**5
    if segment == 1:
        yd1[:, t_idx-1] = [xi + (xf - xi)*s, yi]
        ydx[:, t_idx-1] = [xi + (xf - xi)*s, yi]
    elif segment == 2:
        yd1[:, t_idx-1] = [xf, yi + (yf - yi)*s]
        ydx[:, t_idx-1] = [xf, yi + (yf - yi)*s]
    elif segment == 3:
        yd1[:, t_idx-1] = [xf - (xf - xi)*s, yf]
        ydx[:, t_idx-1] = [xf - (xf - xi)*s, yf]
    elif segment == 4:
        yd1[:, t_idx-1] = [xi, yf - (yf - yi)*s]
        ydx[:, t_idx-1] = [xi, yf - (yf - yi)*s]

# =============================================================================
# Initialize Variables for ILC Simulation
# =============================================================================
E_ = np.zeros(paradigm_trials)
normTE = np.zeros(paradigm_trials)
normSPE = np.zeros(paradigm_trials)
normMOE = np.zeros(paradigm_trials)
normu = np.zeros(paradigm_trials)
normuFF = np.zeros(paradigm_trials)
normuILC = np.zeros(paradigm_trials)

y1_conv = y2_conv = y3_conv = y4_conv = None

# Preallocate previous variables (with extra indices)
y_prev = np.zeros((2, total_steps+2))
uILC_prev = np.zeros((2, total_steps))
ypred_prev = np.zeros((2, total_steps+2))
uFF_prev = np.zeros((2, total_steps))
u_prev = np.zeros((2, total_steps))

# =============================================================================
# Across-the-Trials Simulation Loop
# =============================================================================
fig1 = plt.figure(1)

for k in range(paradigm_trials):
    x = np.zeros((n, total_steps))
    x_ILC = np.zeros((n, total_steps))
    x_FF = np.zeros((n, total_steps))
    y = C @ x  # y: shape (2, total_steps)
    
    TE = np.zeros((2, total_steps+2))
    SPE = np.zeros((2, total_steps+2))
    MOE = np.zeros((2, total_steps+2))
    ypred = np.zeros((2, total_steps+2))
    
    uILC_trial = np.zeros((2, total_steps))
    uFF_trial = np.zeros((2, total_steps))
    u_trial = np.zeros((2, total_steps))
    
    # Phase selection based on trial number
    if k < (4 * num_trials):
        yd_trial = yd1.copy()
        RR = np.eye(2)
        FFPT = np.zeros(2)
        VLPT = 0
    elif (k >= (4 * num_trials)) and (k < (5 * num_trials)):
        # Perturbation phase PT1: for example, scale x by -0.75
        yd_trial = np.array([[-0.75, 0], [0, 1]]) @ ydx
        RR = np.eye(2)
        FFPT = np.zeros(2)
        VLPT = 0
    elif (k >= (5 * num_trials)) and (k < (6 * num_trials)):
        yd_trial = yd1.copy()
        RR = np.eye(2)
        FFPT = np.zeros(2)
        VLPT = 0
    elif (k >= (6 * num_trials)) and (k < (7 * num_trials)):
        yd_trial = ydx.copy()
        RR = np.eye(2)
        FFPT = np.zeros(2)
        VLPT = 0
    
    for t in range(total_steps - 2):
        if t == 0:
            vdx = np.zeros(2)
        else:
            vdx = y_prev[:, t] - y_prev[:, t-1]
        
        TE[:, t+2] = yd_trial[:, t+2] - (RR @ y_prev[:, t+2])
        SPE[:, t+2] = (C @ A @ B @ u_prev[:, t]).flatten() - ypred[:, t+2]
        MOE[:, t+2] = (RR @ y_prev[:, t+2]) - ypred[:, t+2]
        
        if np.log(norm(TE[:, t+2]) + 1e-12) >= -5:
            uILC_trial[:, t] = lambda_ILC * uILC_prev[:, t] + gamma_ILC * np.tanh(TE[:, t+2])
            ypred[:, t+2] = lambda_pred * ypred_prev[:, t+2] + gamma_pred * np.tanh(MOE[:, t+2])
            uFF_trial[:, t] = lambda_FF * uFF_prev[:, t] + gamma_FF * (pinv(C @ A @ B) @ np.tanh(SPE[:, t+2])).flatten()
            u_trial[:, t] = uILC_trial[:, t] + uFF_trial[:, t]
        else:
            u_trial[:, t] = u_prev[:, t]
        
        x[:, t+1] = A @ x[:, t] + nx * np.random.randn(n) + B @ (u_trial[:, t] + pinv(C @ A @ B) @ FFPT + VLPT * vdx)
        y[:, t] = C @ x[:, t] + ny * np.random.randn(C.shape[0])
    
    u_prev = u_trial.copy()
    y_prev = y.copy()
    ypred_prev = ypred.copy()
    uILC_prev = uILC_trial.copy()
    uFF_prev = uFF_trial.copy()
    
    normTE[k] = norm(TE[:, :total_steps])
    normSPE[k] = norm(SPE[:, :total_steps])
    normMOE[k] = norm(MOE[:, :total_steps])
    normuFF[k] = norm(uFF_trial[:, :total_steps-2])
    normu[k] = norm(u_trial[:, :total_steps-2])
    normuILC[k] = norm(uILC_trial[:, :total_steps-2])
    
    E_trial = norm(yd_trial[:, :total_steps] - y[:, :total_steps])
    E_[k] = E_trial
    
    if k == (4 * num_trials - 1):
        y1_conv = y.copy()
        x1_conv = x.copy()
    if k == (5 * num_trials - 1):
        y2_conv = y.copy()
        x2_conv = x.copy()
    if k == (6 * num_trials - 1):
        y3_conv = y.copy()
        x3_conv = x.copy()
    if k == (7 * num_trials - 1):
        y4_conv = y.copy()
        x4_conv = x.copy()
    
    # Optional intermediate plotting
    if k < (2 * num_trials):
        if (k % 20 == 0) or (k == (2 * num_trials - 1)):
            plt.subplot(1, 4, 1)
            plt.plot(y[0, :-2], y[1, :-2], 'b-', linewidth=0.1)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Baseline')
            plt.ylim([0, 1.1])
    elif (k >= (4 * num_trials)) and (k < (5 * num_trials)):
        if (k % 20 == 0) or (k == (4 * num_trials)):
            plt.subplot(1, 4, 2)
            plt.plot(y[0, :-2], y[1, :-2], 'b-', linewidth=0.1)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('PT1')
            plt.ylim([0, 1.1])
    elif (k >= (5 * num_trials)) and (k < (6 * num_trials)):
        if (k % 20 == 0) or (k == (5 * num_trials)):
            plt.subplot(1, 4, 3)
            plt.plot(y[0, :-2], y[1, :-2], 'b-', linewidth=0.1)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Washout')
            plt.ylim([0, 1.1])
    elif (k >= (6 * num_trials)) and (k < (7 * num_trials)):
        if (k % 20 == 0) or (k == (6 * num_trials)):
            plt.subplot(1, 4, 4)
            plt.plot(y[0, :-2], y[1, :-2], 'b-', linewidth=0.1)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('PT2')
            plt.ylim([0, 1.1])

# =============================================================================
# Plotting Final Results
# =============================================================================
plt.figure(1)
plt.subplot(1,4,1)
plt.plot(yd1[0, :-2], yd1[1, :-2], 'r--', linewidth=2)
plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.ylim([0, 0.6])
plt.title('Baseline')
plt.subplot(1,4,2)
plt.plot(-0.5 * ydx[0, :-2], ydx[1, :-2], 'r--', linewidth=2)
plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.ylim([0, 0.6])
plt.title('PT1')
plt.subplot(1,4,3)
plt.plot(yd1[0, :-2], yd1[1, :-2], 'r--', linewidth=2)
plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.ylim([0, 0.6])
plt.title('Washout')
plt.subplot(1,4,4)
plt.plot(ydx[0, :-2], ydx[1, :-2], 'r--', linewidth=2)
plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.ylim([0, 0.6])
plt.title('PT2')

if y1_conv is not None:
    plt.subplot(1,4,1)
    plt.plot(y1_conv[0, :-2], y1_conv[1, :-2], 'g-', linewidth=1)
    plt.ylim([0, 0.6])
if y2_conv is not None:
    plt.subplot(1,4,2)
    plt.plot(y2_conv[0, :-2], y2_conv[1, :-2], 'g-', linewidth=1)
    plt.ylim([0, 0.6])
if y3_conv is not None:
    plt.subplot(1,4,3)
    plt.plot(y3_conv[0, :-2], y3_conv[1, :-2], 'g-', linewidth=1)
    plt.ylim([0, 0.6])
if y4_conv is not None:
    plt.subplot(1,4,4)
    plt.plot(y4_conv[0, :-2], y4_conv[1, :-2], 'g-', linewidth=1)
    plt.ylim([0, 0.6])

if export_plots_to_pdf:
    plt.savefig('ConvergedXY.pdf', format='pdf')

plt.figure(2)
plt.plot(normTE, linewidth=1, label='TE')
plt.plot(normSPE, linewidth=1, label='SPE')
plt.plot(normMOE, linewidth=1, label='MOE')
plt.grid(True)
plt.xlabel('Iterations (k)')
plt.ylabel('Norm of individual components')
plt.legend(loc='upper left')
plt.title(f'γ_ILC={gamma_ILC}, γ_FF={gamma_FF}, γ_pred={gamma_pred}, λ_ILC={lambda_ILC}, λ_FF={lambda_FF}, λ_pred={lambda_pred}')
if export_plots_to_pdf:
    plt.savefig('ErrorProportions.pdf', format='pdf')

plt.figure(3)
plt.subplot(2,1,1)
plt.stem(E_, linefmt='-', markerfmt=' ', basefmt=' ')
plt.plot(E_, linewidth=0.5)
plt.grid(True)
plt.xlabel('Iterations (k)')
plt.ylabel('||E||2=||y_ref-y||2')
plt.title(f'γ_ILC={gamma_ILC}, γ_FF={gamma_FF}, γ_pred={gamma_pred}, λ_ILC={lambda_ILC}, λ_FF={lambda_FF}, λ_pred={lambda_pred}')
plt.subplot(2,1,2)
plt.stem(np.log(E_ + 1e-12), linefmt='-', markerfmt=' ', basefmt=' ')
plt.plot(np.log(E_ + 1e-12), linewidth=0.5)
plt.grid(True)
plt.xlabel('Iterations (k)')
plt.ylabel('log(||E||2)')
if export_plots_to_pdf:
    plt.savefig('Errors.pdf', format='pdf')

plt.figure(9)
idx = np.arange(1, paradigm_trials - 3*num_trials + 1)
plt.subplot(2,1,1)
plt.stem(idx, E_[:len(idx)], linefmt='-', markerfmt=' ', basefmt=' ')
plt.plot(idx, E_[:len(idx)], linewidth=0.5)
plt.xlim([0*num_trials+1, 2*num_trials])
plt.grid(True)
plt.xlabel('Iterations (k)')
plt.ylabel('||E||2')
plt.title(f'γ_ILC={gamma_ILC}, γ_FF={gamma_FF}, γ_pred={gamma_pred}, λ_ILC={lambda_ILC}, λ_FF={lambda_FF}, λ_pred={lambda_pred}')
plt.subplot(2,1,2)
plt.stem(idx, np.log(E_[:len(idx)] + 1e-12), linefmt='-', markerfmt=' ', basefmt=' ')
plt.plot(idx, np.log(E_[:len(idx)] + 1e-12), linewidth=0.5)
plt.xlim([0*num_trials+1, 2*num_trials])
plt.grid(True)
plt.xlabel('Iterations (k)')
plt.ylabel('log(||E||2)')
if export_plots_to_pdf:
    plt.savefig('BaselineError.pdf', format='pdf')

plt.show()
