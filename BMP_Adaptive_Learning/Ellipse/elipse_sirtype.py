import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, pinv
import math

# -------------------------------
# Export/Plot Settings (similar to MATLAB s structure)
# -------------------------------
fig_width = 8   # inches
fig_height = 5  # inches
plt.rcParams['figure.figsize'] = (fig_width, fig_height)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# -------------------------------
# Initialization
# -------------------------------
num_trials = 100
ts = 0.01                      # Sampling time interval [s]
paradigm_trials = 7 * num_trials  # Total number of trials
time_total = 1                 # Total simulation time [s]
duration = time_total
export_plots_to_pdf = True

# -------------------------------
# System Definition (2R Robot)
# -------------------------------
m1 = 0.2       # mass of link 1
m2 = 0.2       # mass of link 2
lc1 = 0.1      # midlength of link 1
lc2 = 0.1      # midlength of link 2
I1 = (1/3)*m1*(2*lc1)  # Moment of inertia of link 1
I2 = (1/3)*m2*(2*lc2)  # Moment of inertia of link 2
Q1 = math.pi/4         # Equilibrium point of q1
Q2 = -math.pi/2        # Equilibrium point of q2
d1 = 0.3       # Damping 1
d2 = 0.3       # Damping 2

den = (I1*I2 + 4*lc1*lc2**2*m2**2 + 4*I2*lc1*m2 +
       I2*lc1**2*m1 + I1*lc2**2*m2 + lc1**2*lc2**2*m1*m2 -
       4*lc1**2*lc2**2*m2**2*(math.cos(Q2)**2))

Dinv = np.array([
    [(m2*lc2**2 + I2) / den,
     -(m2*lc2**2 + 2*lc1*m2*math.cos(Q2)*lc2) / den],
    [-(m2*lc2**2 + 2*lc1*m2*math.cos(Q2)*lc2) / den,
     (m1*lc1**2 + 4*m2*math.cos(Q2)*lc1*lc2 + 4*m2*lc1 + m2*lc2**2 + I1 + I2) / den]
])

a11 = np.zeros((2, 2))
a12 = np.eye(2)
a21 = np.zeros((2, 2))
a22 = -Dinv @ np.diag([d1, d2])
b11 = np.zeros((2, 2))
b12 = Dinv @ np.eye(2)

Ac = np.block([[a11, a12],
               [a21, a22]])
Bc = np.vstack([b11, b12])
C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

n = Ac.shape[0]  # State dimension = 4
A = 0.5 * (np.eye(n) + Ac * ts)
B = Bc

# -------------------------------
# Controller Equations Parameterization
# -------------------------------
lambda_ILC = 1       
lambda_pred = 0.7
lambda_FF = 0.7      
gamma_ILC = 0.1      
gamma_pred = 0.2     
gamma_FF = 0.1       
nx = 0.0             
ny = 0.0             

# -------------------------------
# Perturbation Definition
# -------------------------------
beta = -0.1       
alpha = 15        
th1 = math.pi/4   
th2 = -math.pi/4  

enableVMR = 0
enableFFPT = 0
enableVLPT = 0

# -------------------------------
# Trajectory Generation (Elliptical Path)
# -------------------------------
a_ellipse = 0.5   
b_ellipse = 0.3   
xc = 0.5          
yc = 0.5          
duration_traj = time_total / 4

total_steps = int(np.round(time_total / ts))
yd1 = np.zeros((2, total_steps))
yd = np.zeros((2, total_steps))
ydx = np.zeros((2, total_steps))  

t_idx = 0
for loop_time in np.arange(0, time_total, ts):
    t_idx += 1
    segment_time = loop_time % duration_traj
    s = 10*(segment_time/duration_traj)**3 - 15*(segment_time/duration_traj)**4 + 6*(segment_time/duration_traj)**5
    theta = s * 2 * math.pi
    x_val = xc + a_ellipse * math.cos(theta)
    y_val = yc + b_ellipse * math.sin(theta)
    yd1[:, t_idx-1] = [x_val, y_val]

# -------------------------------
# Preallocate Learning and Storage Variables
# -------------------------------
E_ = np.zeros(paradigm_trials)
normTE = np.zeros(paradigm_trials)
normSPE = np.zeros(paradigm_trials)
normMOE = np.zeros(paradigm_trials)
normu = np.zeros(paradigm_trials)
normuFF = np.zeros(paradigm_trials)
normuILC = np.zeros(paradigm_trials)

y1 = y2 = y3 = y4 = None

# Preallocate "previous" variables as 2D arrays with proper shape
y_prev = np.zeros((2, total_steps+2))
uILC_prev = np.zeros((2, total_steps))
ypred_prev = np.zeros((2, total_steps+2))
uFF_prev = np.zeros((2, total_steps))
u_prev = np.zeros((2, total_steps))

# -------------------------------
# Across-the-Trials Simulation Loop
# -------------------------------
fig1 = plt.figure(1)

for k in range(paradigm_trials):
    # Initialize state and control arrays for this trial
    x = np.zeros((n, total_steps))
    x_ILC = np.zeros((n, total_steps))
    x_FF = np.zeros((n, total_steps))
    y = C @ x  # y has shape (2, total_steps)
    
    # Preallocate time-indexed variables (with two extra time steps)
    TE = np.zeros((2, total_steps+2))
    SPE = np.zeros((2, total_steps+2))
    MOE = np.zeros((2, total_steps+2))
    ypred = np.zeros((2, total_steps+2))
    
    # Preallocate control arrays for this trial
    uILC_trial = np.zeros((2, total_steps))
    uFF_trial = np.zeros((2, total_steps))
    u_trial = np.zeros((2, total_steps))
    
    # Select desired trajectory and rotation matrix based on trial phase
    if k < (4 * num_trials):
        yd_trial = yd1.copy()
        RR = np.eye(2)
        FFPT = np.zeros(2)
        VLPT = 0
    elif k < (5 * num_trials):
        rad = np.deg2rad(-45)
        RR = np.array([[np.cos(rad), -np.sin(rad)],
                       [np.sin(rad),  np.cos(rad)]])
        yd_trial = RR @ yd1
        FFPT = np.zeros(2)
        VLPT = 0
    elif k < (6 * num_trials):
        yd_trial = yd1.copy()
        RR = np.eye(2)
        FFPT = np.zeros(2)
        VLPT = 0
    else:
        rad = np.deg2rad(45)
        RR = np.array([[np.cos(rad), -np.sin(rad)],
                       [np.sin(rad),  np.cos(rad)]])
        yd_trial = RR @ yd1
        FFPT = np.zeros(2)
        VLPT = 0

    # Time-loop: iterate for each time step (from 0 to total_steps-2)
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
    
    # Update "previous" variables for next trial
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

    # Save converged trajectories from selected trials
    if k == (4 * num_trials - 1):
        y1 = y.copy()
        x1 = x.copy()
    if k == (5 * num_trials - 1):
        y2 = y.copy()
        x2 = x.copy()
    if k == (6 * num_trials - 1):
        y3 = y.copy()
        x3 = x.copy()
    if k == (7 * num_trials - 1):
        y4 = y.copy()
        x4 = x.copy()
    
    # Optionally, plot trajectory every 20 iterations for different phases
    if k < (2 * num_trials):
        if (k % 20 == 0) or (k == (2 * num_trials - 1)):
            plt.subplot(1, 4, 1)
            plt.plot(y[0, :-2], y[1, :-2], 'b-', linewidth=0.1)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Baseline')
            plt.ylim([0, 1.1])
    elif k < (5 * num_trials) and k >= (4 * num_trials):
        if (k % 20 == 0) or (k == (4 * num_trials)):
            plt.subplot(1, 4, 2)
            plt.plot(y[0, :-2], y[1, :-2], 'b-', linewidth=0.1)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('PT1')
            plt.ylim([0, 1.1])
    elif k < (6 * num_trials) and k >= (5 * num_trials):
        if (k % 20 == 0) or (k == (5 * num_trials)):
            plt.subplot(1, 4, 3)
            plt.plot(y[0, :-2], y[1, :-2], 'b-', linewidth=0.1)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('Washout')
            plt.ylim([0, 1.1])
    elif k < (7 * num_trials) and k >= (6 * num_trials):
        if (k % 20 == 0) or (k == (6 * num_trials)):
            plt.subplot(1, 4, 4)
            plt.plot(y[0, :-2], y[1, :-2], 'b-', linewidth=0.1)
            plt.xlabel('X (m)')
            plt.ylabel('Y (m)')
            plt.title('PT2')
            plt.ylim([0, 1.1])

# End of trial loop

# -------------------------------
# Plotting Results
# -------------------------------
plt.figure(1)
plt.subplot(1, 4, 1)
plt.plot(yd1[0, :-2], yd1[1, :-2], 'r--', linewidth=2)
plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.ylim([0, 0.6])
plt.title('Baseline')
plt.subplot(1, 4, 2)
plt.plot(-0.5 * ydx[0, :-2], ydx[1, :-2], 'r--', linewidth=2)
plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.ylim([0, 0.6])
plt.title('PT1')
plt.subplot(1, 4, 3)
plt.plot(yd1[0, :-2], yd1[1, :-2], 'r--', linewidth=2)
plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.ylim([0, 0.6])
plt.title('Washout')
plt.subplot(1, 4, 4)
plt.plot(ydx[0, :-2], ydx[1, :-2], 'r--', linewidth=2)
plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.ylim([0, 0.6])
plt.title('PT2')

if y1 is not None:
    plt.subplot(1, 4, 1)
    plt.plot(y1[0, :-2], y1[1, :-2], 'g-', linewidth=1)
    plt.ylim([0, 0.6])
if y2 is not None:
    plt.subplot(1, 4, 2)
    plt.plot(y2[0, :-2], y2[1, :-2], 'g-', linewidth=1)
    plt.ylim([0, 0.6])
if y3 is not None:
    plt.subplot(1, 4, 3)
    plt.plot(y3[0, :-2], y3[1, :-2], 'g-', linewidth=1)
    plt.ylim([0, 0.6])
if y4 is not None:
    plt.subplot(1, 4, 4)
    plt.plot(y4[0, :-2], y4[1, :-2], 'g-', linewidth=1)
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
plt.title(r'$\gamma_{ILC}=$' + f'{gamma_ILC}, ' +
          r'$\gamma_{FF}=$' + f'{gamma_FF}, ' +
          r'$\gamma_{pred}=$' + f'{gamma_pred}, ' +
          r'$\lambda_{ILC}=$' + f'{lambda_ILC}, ' +
          r'$\lambda_{FF}=$' + f'{lambda_FF}, ' +
          r'$\lambda_{pred}=$' + f'{lambda_pred}')
if export_plots_to_pdf:
    plt.savefig('ErrorProportions.pdf', format='pdf')

plt.figure(3)
plt.subplot(2,1,1)
plt.stem(E_)
plt.plot(E_, linewidth=0.5)
plt.grid(True)
plt.xlabel('Iterations (k)')
plt.ylabel(r'$||E||_2=||y^{ref}_k-y_k||_2$')
plt.title(r'$\gamma_{ILC}=$' + f'{gamma_ILC}, ' +
          r'$\gamma_{FF}=$' + f'{gamma_FF}, ' +
          r'$\gamma_{pred}=$' + f'{gamma_pred}, ' +
          r'$\lambda_{ILC}=$' + f'{lambda_ILC}, ' +
          r'$\lambda_{FF}=$' + f'{lambda_FF}, ' +
          r'$\lambda_{pred}=$' + f'{lambda_pred}')
plt.subplot(2,1,2)
plt.stem(np.log(E_ + 1e-12))
plt.plot(np.log(E_ + 1e-12), linewidth=0.5)
plt.grid(True)
plt.xlabel('Iterations (k)')
plt.ylabel(r'$\log(||E||_2)$')
if export_plots_to_pdf:
    plt.savefig('Errors.pdf', format='pdf')

plt.show()
