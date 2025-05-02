import os
import mujoco as mj
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set font to serif (Times New Roman) and enable LaTeX for labels if desired
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
# mpl.rcParams['text.usetex'] = True  # Uncomment if LaTeX is installed

# -------------------------------
# Simulation & Controller Parameters
# -------------------------------
simend = 1.0           # Simulation time per trial [s]
baseline_trials = 100  # Number of trials for each phase base
num_trials = baseline_trials * 7  # Total trials (7 phases)
ts = 0.01              # Sampling time [s]
num_timesteps = int(simend / ts)  # Timesteps per trial

# ILC Controller Gains
lambda_ILC = 1.0     # Retention factor for ILC update
gamma_ILC  = 0.1     # ILC learning gain
lambda_FF  = 0.7     # Retention factor for feedforward update
gamma_FF   = 0.1     # Feedforward learning gain
lambda_op  = 0.7     # Model output retention (for prediction)
gamma_op   = 0.2     # Model output learning gain

# -------------------------------
# Model & XML Path Setup
# -------------------------------
xml_path = '2R.xml'  # Path to your MuJoCo XML model file
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname, xml_path)
xml_path = abspath

model = mj.MjModel.from_xml_path(xml_path)  # Load MuJoCo model
data = mj.MjData(model)                     # Create data container

# -------------------------------
# 2R Robot Kinematics Parameters
# -------------------------------
l1 = 2 * 0.1  # Link 1 length (from MATLAB: 2*lc1 with lc1=0.1)
l2 = 2 * 0.1  # Link 2 length (from MATLAB: 2*lc2 with lc2=0.1)

# Inverse Kinematics (for a 2-link planar arm)
def inverse_kinematics(x, y, l1=l1, l2=l2):
    # Compute q1 using cosine law (clip for numerical safety)
    cos_q1 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_q1 = np.clip(cos_q1, -1.0, 1.0)
    q1 = np.arccos(cos_q1)
    # Compute q0 from geometry
    q0 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q1), l1 + l2 * np.cos(q1))
    return q0, q1

# Forward Kinematics (for plotting end-effector trajectory)
def forward_kinematics(q0, q1, l1=l1, l2=l2):
    x = l1 * np.cos(q0) + l2 * np.cos(q0 + q1)
    y = l1 * np.sin(q0) + l2 * np.sin(q0 + q1)
    return x, y

# -------------------------------
# Hexagon Trajectory Generation (Cartesian, Minimum-Jerk Interpolation)
# -------------------------------
def hexagon_trajectory(total_time, ts, center, side_length):
    """
    Generates a hexagon trajectory using 6 segments (minimum-jerk on each segment).
    center: (xc, yc)
    side_length: length of each side.
    Returns a 2 x num_timesteps array of (x, y) coordinates.
    """
    num_steps = int(total_time / ts)
    traj = np.zeros((2, num_steps))
    
    # Define hexagon vertices (counterclockwise)
    xc, yc = center
    theta_hex = np.linspace(0, 2*np.pi, 7)  # 7 points, first equals last
    x_hex = xc + side_length * np.cos(theta_hex)
    y_hex = yc + side_length * np.sin(theta_hex)
    
    seg_duration = total_time / 6  # each segment duration
    for i in range(num_steps):
        t = i * ts
        # Determine current segment (1 to 6)
        seg = int(np.floor(t / seg_duration))
        if seg >= 6:
            seg = 5
        # Normalized time within the segment
        t_seg = t - seg * seg_duration
        tau = t_seg / seg_duration
        # Minimum-jerk scaling function
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        # Linear interpolation between vertex seg and seg+1
        x_start, y_start = x_hex[seg], y_hex[seg]
        x_end,   y_end   = x_hex[seg+1], y_hex[seg+1]
        traj[0, i] = x_start + s * (x_end - x_start)
        traj[1, i] = y_start + s * (y_end - y_start)
    return traj

# Generate baseline (hexagon) trajectory
center = (0.5, 0.5)
side_length = 0.4
yd_base = hexagon_trajectory(simend, ts, center, side_length)

# Rotation utility to create perturbed trajectories
def rotate_trajectory(traj, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    return R @ traj

# Desired trajectories for different phases
# Phase definitions (in trials):
#   Baseline: trial <= 4*num_trials_baseline
#   PT1:      4*num_trials_baseline < trial <= 5*num_trials_baseline  (rotate -45°)
#   Washout:  5*num_trials_baseline < trial <= 6*num_trials_baseline  (baseline)
#   PT2:      6*num_trials_baseline < trial <= 7*num_trials_baseline  (rotate 45°)
yd_PT1 = rotate_trajectory(yd_base, -45)
yd_PT2 = rotate_trajectory(yd_base, 45)

# -------------------------------
# Convert Cartesian Reference Trajectory to Joint Angles
# -------------------------------
reference_q0 = np.zeros(num_timesteps)
reference_q1 = np.zeros(num_timesteps)
for t in range(num_timesteps):
    # For baseline phase, use yd_base; other phases will be applied later in the loop
    reference_q0[t], reference_q1[t] = inverse_kinematics(yd_base[0, t], yd_base[1, t])

# -------------------------------
# Storage for Trial Data
# -------------------------------
u_ILC = np.zeros((num_timesteps, 2))   # ILC control input (joint torques)
u_FF  = np.zeros((num_timesteps, 2))   # Feedforward control input
y_history = np.zeros((num_trials, num_timesteps, 2))  # Joint angle outputs per trial
u_history = np.zeros((num_trials, num_timesteps, 2))  # Control history
error_norms = np.zeros(num_trials)     # Error norm per trial

# -------------------------------
# MuJoCo Controller Function
# -------------------------------
def controller(model, data, u, disturbance=0):
    # Apply control (u for both joints)
    data.ctrl[0] = u[0]
    data.ctrl[1] = u[1]
    # Optionally apply disturbance to a channel (if needed)
    data.qfrc_applied[1] = disturbance
    mj.mj_step(model, data)  # Step the simulation

# -------------------------------
# Plotting Function for a Trial
# -------------------------------
def plot_after_trial(trial, time_data, ref_q0, ref_q1, y_trial, u_ILC, err):
    plt.figure(1)
    plt.plot(range(1, num_trials+1), error_norms, 'bo-', label='Error Norm (MSE)')
    plt.xlabel('Trial Number')
    plt.ylabel('Error Norm (MSE)')
    plt.title('Error Norm vs Trial Number')
    plt.legend()
    plt.grid(True)
    
    plt.figure(2, figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time_data, ref_q0, 'r--', label='Reference q0')
    plt.plot(time_data, y_trial[:, 0], 'b-', label=f'Output q0 (Trial {trial+1})')
    plt.plot(time_data, ref_q1, 'g--', label='Reference q1')
    plt.plot(time_data, y_trial[:, 1], 'm-', label=f'Output q1 (Trial {trial+1})')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint Angles [rad]')
    plt.title(f'Trial {trial+1}: Joint Angles vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time_data, u_ILC[:, 0], label=f'ILC q0 (Trial {trial+1})')
    plt.plot(time_data, u_ILC[:, 1], label=f'ILC q1 (Trial {trial+1})')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Input (Torque)')
    plt.title(f'Trial {trial+1}: Control Input vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time_data, err[:, 0], label=f'Error q0 (Trial {trial+1})')
    plt.plot(time_data, err[:, 1], label=f'Error q1 (Trial {trial+1})')
    plt.xlabel('Time [s]')
    plt.ylabel('Error')
    plt.title(f'Trial {trial+1}: Tracking Error vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()

# -------------------------------
# Main ILC Loop Over Trials
# -------------------------------
time_data = np.linspace(0, simend, num_timesteps)
# Preallocate arrays for the current trial simulation
y_trial = np.zeros((num_timesteps, 2))
errors_trial = np.zeros((num_timesteps, 2))
model_op = np.zeros((num_timesteps, 2))  # For a simple internal model output

for trial in range(num_trials):
    print(f"Trial {trial+1} / {num_trials}")
    data = mj.MjData(model)  # Reset simulation data for each trial
    
    # Set initial joint positions to the first reference point (for baseline)
    init_q0, init_q1 = inverse_kinematics(yd_base[0, 0], yd_base[1, 0])
    data.qpos[0] = init_q0
    data.qpos[1] = init_q1
    data.qvel[:] = 0
    
    # Select desired Cartesian trajectory based on trial phase:
    if trial < 4 * baseline_trials:
        yd_des = yd_base.copy()
    elif trial < 5 * baseline_trials:
        yd_des = yd_PT1.copy()
    elif trial < 6 * baseline_trials:
        yd_des = yd_base.copy()
    else:
        yd_des = yd_PT2.copy()
    
    # Convert current desired Cartesian trajectory to joint angle references
    ref_q0 = np.zeros(num_timesteps)
    ref_q1 = np.zeros(num_timesteps)
    for t in range(num_timesteps):
        ref_q0[t], ref_q1[t] = inverse_kinematics(yd_des[0, t], yd_des[1, t])
    
    # Run simulation for this trial over all timesteps
    for t in range(num_timesteps):
        # Read current joint positions
        y_trial[t, 0] = data.qpos[0]
        y_trial[t, 1] = data.qpos[1]
        # Compute tracking error
        errors_trial[t, 0] = ref_q0[t] - y_trial[t, 0]
        errors_trial[t, 1] = ref_q1[t] - y_trial[t, 1]
        
        # (Optional) Update a simple internal model output
        model_op[t, 0] = lambda_op * model_op[t, 0] + gamma_op * np.tanh(errors_trial[t, 0])
        model_op[t, 1] = lambda_op * model_op[t, 1] + gamma_op * np.tanh(errors_trial[t, 1])
        
        # ILC control update if error is above a threshold (here always update)
        u_ILC[t, 0] = lambda_ILC * u_ILC[t, 0] + gamma_ILC * np.tanh(errors_trial[t, 0])
        u_ILC[t, 1] = lambda_ILC * u_ILC[t, 1] + gamma_ILC * np.tanh(errors_trial[t, 1])
        
        # Feedforward control update (using a pseudo-inverse gain; here a simple proportional term)
        u_FF[t, :] = lambda_FF * u_FF[t, :] + gamma_FF * (0.5*(u_ILC[t, :] + u_FF[t, :]) - model_op[t, :])
        
        # Total control input is the sum of ILC and feedforward terms
        u = u_ILC[t, :] + u_FF[t, :]
        
        # Apply control input to MuJoCo model
        controller(model, data, u)
    
    # Record trial data
    y_history[trial, :, :] = y_trial
    u_history[trial, :, :] = u_ILC + u_FF
    error_norms[trial] = np.linalg.norm(errors_trial) / num_timesteps
    
    # (Optional) Plot trial data after each trial
    # plot_after_trial(trial, time_data, ref_q0, ref_q1, y_trial, u_ILC, errors_trial)

# -------------------------------
# Plotting Summary Results
# -------------------------------
# Plot error norm (MSE) vs trial
plt.figure(1)
plt.plot(range(1, num_trials+1), error_norms, 'bo-')
plt.xlabel('Trial Number')
plt.ylabel('Error Norm (MSE)')
plt.title('Tracking Error Norm vs Trial Number')
plt.grid(True)

# Plot the last trial joint trajectories and control
plt.figure(2, figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time_data, ref_q0, 'r--', label='Reference q0')
plt.plot(time_data, y_trial[:, 0], 'b-', label='Output q0')
plt.plot(time_data, ref_q1, 'g--', label='Reference q1')
plt.plot(time_data, y_trial[:, 1], 'm-', label='Output q1')
plt.xlabel('Time [s]')
plt.ylabel('Joint Angles [rad]')
plt.title('Joint Angles vs Time (Last Trial)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_data, u_ILC[:, 0] + u_FF[:, 0], label='Control Input q0')
plt.plot(time_data, u_ILC[:, 1] + u_FF[:, 1], label='Control Input q1')
plt.xlabel('Time [s]')
plt.ylabel('Torque')
plt.title('Control Inputs vs Time (Last Trial)')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time_data, errors_trial[:, 0], label='Error q0')
plt.plot(time_data, errors_trial[:, 1], label='Error q1')
plt.xlabel('Time [s]')
plt.ylabel('Error')
plt.title('Tracking Errors vs Time (Last Trial)')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Plot End-Effector Trajectory in Cartesian Space
actual_X = np.zeros(num_timesteps)
actual_Y = np.zeros(num_timesteps)
for t in range(num_timesteps):
    actual_X[t], actual_Y[t] = forward_kinematics(y_trial[t, 0], y_trial[t, 1], l1, l2)

plt.figure(3)
plt.plot(yd_base[0, :], yd_base[1, :], 'r--', label='Baseline Reference')
plt.plot(yd_PT1[0, :], yd_PT1[1, :], 'm--', label='PT1 Reference')
plt.plot(yd_PT2[0, :], yd_PT2[1, :], 'c--', label='PT2 Reference')
plt.plot(actual_X, actual_Y, 'b-', label='Actual Trajectory')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('End-Effector Trajectory')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
