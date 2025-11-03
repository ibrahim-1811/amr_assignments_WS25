#!/usr/bin/env python
# coding: utf-8

# |        |        |        |
# |--------|--------|--------|
# ![H-BRS](logos/h-brs.png) | ![A2S](logos/a2s.png) | ![b-it](logos/b-it.png) |
# 
# # Autonomous Mobile Robots
# 
# # AMR Assignment 3

# ### General information
# 
# * Please do not add or delete any cells. Answers belong into the already provided cells (below the question).
# * If a function is given (either as a signature or a full function), you should not change the name, arguments, or return value of the function.
# * If you encounter empty cells underneath the answer that can not be edited, please ignore them; they are for testing purposes.
# * Please note that variables declared in the notebook cells have global scope. To make sure your assignment works correctly as a whole, please restart the kernel and run all cells before submitting (e.g. via *Kernel -> Restart & Run All*).
# * Code cells where you are supposed to give your answer often include the line  ```raise NotImplementedError```. This makes it easier to automatically grade answers. Once you fill out a function, please delete this line.
# 
# ### Submission
# 
# Please make sure to write all your team members 2s IDs in the cell below before submission. Please submit your notebook via the JupyterHub web interface (in the main view -> Assignments -> Submit). If it is a group assignment, please make only one submission per group (for easier bookkeeping, it is best if this is always the same team member).
# 
# ### Questions about the assignment
# 
# If you have questions about the assignment, you are encouraged to post them in the LEA forum. Proactive discussions lead to better understanding. Let's keep the forum active.

# ## Team members (2s IDs):
# 
# YOUR ANSWER HERE
# 
# * Mmemon2s
# * Akanch2s
# * ekidan2s

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt


# # A. Differential Drive Kinematics [50 points]
# 
# In this exercise, you will work with the kinematics model of a simple differential drive robot, which, as you know from the lecture, is driven by two active, steerable standard wheels (often supported by a passive caster or spherical wheel). With this setup, only spinning of the wheels is possible, and there is no lateral motion (motion perpendicular to the plane of the wheel). We additionally assume that the system and the environment are ideal, i.e. there is no slippage or loss of contact at the wheels, and the surface is flat.
# 
# We will consider the schematic of the robot from the lecture slides on kinematics.
# 
# ![kinematics_diagram](img/DifferentialRobot.png)
# 
# We will follow the notation below throughout this assignment:
# * $R$ and $I$ are the local and global Cartesian coordinate reference frames, respectively
# * The position of point $P$ at time $t$ is $(^ix_t, ^iy_t)$, which can also be represented as $({x_t}, {y_t})$ for simplicity. Similarly, its velocity at time $t$ can be represented as $({v_x}_t, {v_y}_t)$
# * $l$ is the distance of point $P$ from any of the two wheels. Thus, the distance between the two wheels is $2l$
# * $\theta_t$ is the angle measured from $X_I$ to $X_R$ at time $t$
# * $\omega_t$ is the angular velocity of the platform at time $t$, i.e. $\omega_t = \frac{d\theta_t}{dt}$
# * $\dot{\phi}_l$ and $\dot{\phi}_r$ are the angular velocities of the left and the right wheel, respectively
# * ${v_l}^t$ and ${v_r}^t$ are the linear velocities of the left and the right wheel, respectively, at time $t$
# * $r_l$ and $r_r$ are the radii of the left and the right wheel, respectively

# ## Kinematic equations [30 Points]
# 
# First, answer the following questions, which will assist you in building your kinematic model. Please follow the notation described above, and (if needed) define any new symbols you introduce.

# 1. Describe the linear velocity of both wheels in terms of their radius and angular velocity. **[5 Point]**

# Linear Velocity of the wheel is the product of its radius and its angular velocity
# 
# v_{l,t}= (r_l*phi_l + r_r *phi_r) / 2 ;
# 

# 2. Describe the instantaneous change of the robot's orientation ($\omega_t$) in terms of linear velocities of the left and right wheels along with the distance between the two wheels. i.e. find $\frac{d\theta_t}{dt}$. **[5 Point]**

# The angular velocity (\omega_t or \frac{d\theta_t}{dt}) is determined by the difference in the linearvelocities of the two wheels, divided by the wheel separation.
# 
# (d\theta_t)/dt = (v_{r,t} - v_{l,t}) / 2l

# 3. Describe the instantaneous change of the robot's position in terms of linear velocities of the left and right wheels and the robot's orientation. i.e. find $ \frac{dx}{dt} $ and $\frac{dy}{dt}$  with respect to the global frame $I$. **[5 Point]**

# Instantaneous change in Robot Position dx/dt and dy/dt ; The robot's forward linear velocity, v_t is the average of the two wheel velocities.and the velocity is directed along the robot's X_R axis.
# 
# v_t = (v_{r,t} + v_{l,t})/2
# 
# tofind the instantaneous change in the global frame I, we project the robot's forward velocity v_t onto the global X_I and Y_I axes using the robots current orientation theta_t
# 
# dx_t/dt = (v_{r,t} + v_{l,t}/2) \ cos(theta_t)
# 
# dy_t/dt = (v_{r,t} + v_{l,t})/2 \ sin(theta_t)

# 4. Convert the equations in questions 2 and 3 from instantaneous time to discrete time. i.e., discretize the equations with respect to $dt$ and describe the movements $\Delta \theta$, $\Delta x$, and $\Delta y$ in terms of $\Delta t$, $\theta_t$, $l$, ${v_l}_t$, and ${v_r}_t$ in the global frame $I$. **[5 Point]**

# Discrete Time Movement (Delta_theta, Delta_x, Delta_y)
#  here we assume the velocities v{l,t}, v{r,t} are constant over the small interval t, t + Delta_t, when converting the instantaneous equations to a discrete time step  Delta_t.
#  
#  Delta_theta = 
#  \Delta\theta = \frac{d\theta_t}{dt} \Delta t
# \mathbf{\Delta\theta} = \left(\frac{v_{r,t} - v_{l,t}}{2l}\right) \Delta t
# \Delta x = \frac{dx_t}{dt} \Delta t
# \mathbf{\Delta x} = \left(\frac{v_{r,t} + v_{l,t}}{2}\right) \cos(\theta_t) \Delta t
# \Delta y = \frac{dy_t}{dt} \Delta t
# \mathbf{\Delta y} = \left(\frac{v_{r,t} + v_{l,t}}{2}\right) \sin(\theta_t) \Delta t
# 

# 5. Finally, describe the current values, $x_t$, $y_t$, $\theta_t$ in terms of the previous values, the change of state, the angular velocities, and the dimension parameters, $x_{t-1}$, $y_{t-1}$, $\theta_{t-1}$, $\Delta t$, ${v_l}^t$, ${v_r}^t$, and $l$. i.e. create the state update equations. **[5 Point]**

# The current state at time t is calculated by adding the discrete movement (\Delta x, \Delta y, \Delta \theta) to the previous state at time t-1:
# \mathbf{\theta_t = \theta_{t-1} + \left(\frac{v_{r,t} - v_{l,t}}{2l}\right) \Delta t}
# \mathbf{x_t = x_{t-1} + \left(\frac{v_{r,t} + v_{l,t}}{2}\right) \cos(\theta_{t-1}) \Delta t}
# \mathbf{y_t = y_{t-1} + \left(\frac{v_{r,t} + v_{l,t}}{2}\right) \sin(\theta_{t-1}) \Delta t}
# (Note: For the position update, \theta_{t-1} is typically used for the rotation component in the simple Euler integration shown here.)
# 

# 6. For a desired platform velocity vector of a differential drive robot, is it always possible to derive suitable wheel velocities? **[5 Point]**
#     - If yes, derive the angular velocities of the individual wheels with the same radius for a desired platform velocity vector $v = [v_x, v_y, \omega]$, described w.r.t the global frame of reference.
#     - If not, derive the condition for which the desired platform velocity vector $v = (v_x, v_y, \omega)$, (described w.r.t the global frame) is achievable by the robot.

# 
# No, for a desired platform velocity vector \mathbf{v} = [v_x, v_y, \omega] (w.r.t the global frame I), it is not always possible to derive suitable wheel velocities.
# Condition for Achievability
# A differential drive robot is a non-holonomic system, meaning it cannot move instantaneously in any direction. Specifically, it cannot move laterally (along the Y_R axis) because of the non-slipping constraint of the wheels.
# The velocity vector \mathbf{v} = [v_x, v_y, \omega] (in the global frame I) must satisfy the non-holonomic constraint for the motion to be physically possible by the robot.
# The constraint is that the velocity in the robot's lateral direction (Y_R axis) must be zero. This lateral velocity v_{Y_R} is the projection of the global velocity [v_x, v_y] onto the Y_R axis, which is at an angle of \theta (plus 90^\circ relative to X_I).
# The condition for the desired velocity \mathbf{v}=(v_x, v_y, \omega) (w.r.t. global frame I) to be achievable is:
# v_x \sin(\theta) - v_y \cos(\theta) = 0
# If this condition is not met, the robot would need lateral slip, which violates the ideal kinematics assumption.
# Derivation for Achievable Velocities (Given the Condition is Met)
# If the constraint is met, we can transform the desired global velocity (v_x, v_y) into the robot's local forward velocity v_t (along X_R).
#  * Robot Forward Velocity (v_t): This is the projection of the global velocity onto the X_R axis:
#    
#    v_t = v_x \cos(\theta) + v_y \sin(\theta)
#  * Combine Kinematic Equations: We use the two fundamental kinematic equations:
#    
#    \text{I. } v_t = \frac{v_r + v_l}{2}
#    
#    
#    \text{II. } \omega = \frac{v_r - v_l}{2l}
#  * Solve for v_r and v_l:
#    * From (II): v_r - v_l = 2l\omega
#    * From (I): v_r + v_l = 2v_t
#    Adding the two equations:
#    
#    2v_r = 2v_t + 2l\omega \implies v_r = v_t + l\omega
#    Subtracting the two equations:
#    
#    2v_l = 2v_t - 2l\omega \implies v_l = v_t - l\omega
#  * Final Wheel Angular Velocities (\dot{\phi}_r, \dot{\phi}_l) (assuming r_l=r_r=r):
#    Substitute v_t from Step 1 and use \dot{\phi} = v/r:
#    \mathbf{\dot{\phi}_r = \frac{1}{r} \left( v_x \cos(\theta) + v_y \sin(\theta) + l\omega \right)}
#    
#    
#    \mathbf{\dot{\phi}_l = \frac{1}{r} \left( v_x \cos(\theta) + v_y \sin(\theta) - l\omega \right)}

# ## Kinematic implementation [10 points]
# 
# Implement the above equations and model a $3s$ movement starting from position $(0.1, 0.1)$ with a current orientation of $\theta_0 = 0.2 rad$ in the global frame $I$. Use an update time of $0.02s$, i.e. $\Delta t = 0.02s$, and move the robot until $t_{total} = 3s$.
# 
# *Hint*: You may find it easier to create a class with a `state_update` function.
# 
# Print the final position and orientation of the robot with respect to the global frame for following three configurations of the robot:
# 
# 1. Configuration 1: $r_l = 3cm$, $r_r = 3cm$, $l = 10 cm$, $\dot{\phi}_l = 1.0 rad/s$, $\dot{\phi}_r = 2.0 rad/s$

# In[27]:


import math
import numpy as np
import matplotlib.pyplot as plt

class Diff_Drive_Robot:
    def __init__(self,rl,rr,l,phi_l,phi_r):
        """convert everything to meters from cm
        phi_l and phi_r left and right wheel angular velocity
        respectivly"""
        self.rl = rl
        self.rr = rr
        self.l = l
        self.phi_l = phi_l
        self.phi_r = phi_r
        self.state = np.array([0.1,0.1,0.2])
        
    def state_update(self,dt):
        """dt time step"""
        x,y,theta = self.state
        # linear veloity
        vl = self.rl * self.phi_l  # Left velocity
        vr = self.rr * self.phi_r  # right velocity
        
        
        #  linear and angular velocities
        v = (vr + vl) / 2.0  # Robot linear velocity
        omega = (vr - vl) / (self.l)  # Robot angular velocity
        
        # Update kinematic equations
        if abs(omega) < 1e-10:  
            x_new = x + v * math.cos(theta) * dt
            y_new = y + v * math.sin(theta) * dt
            theta_new = theta
        else:
            # Circular motion
            ICC_R = v / omega  # Radius of curvature
            ICC_x = x - ICC_R * math.sin(theta)  # Instantaneous Center of Curvature
            ICC_y = y + ICC_R * math.cos(theta)
            
            # Rotation matrix for the change in orientation
            delta_theta = omega * dt
            cos_delta = math.cos(delta_theta)
            sin_delta = math.sin(delta_theta)
            
            # Update position and orientation
            x_new = cos_delta * (x - ICC_x) - sin_delta * (y - ICC_y) + ICC_x
            y_new = sin_delta * (x - ICC_x) + cos_delta * (y - ICC_y) + ICC_y
            theta_new = theta + delta_theta
        
        self.state = np.array([x_new, y_new, theta_new])
    
    def simulate(self, total_time, dt):
        """
        Simulate robot movement for given total time with time step dt
        
        Parameters:
        total_time: total simulation time 3(s)
        dt: time step (s)
        """
        num_steps = int(total_time / dt)
        
        for step in range(num_steps):
            self.state_update(dt)
    
    def get_state(self):
        """Return current state [x, y, theta]"""
        return self.state
    
    
rl = 0.03  # 3 cm
rr = 0.03  # 3 cm
l = 0.10    # 10 cm
phi_l = 1.0  # rad/s
phi_r = 2.0  # rad/s

# Simulation parameters
dt = 0.02  # s
total_time = 3.0  # s

robot_config1 = Diff_Drive_Robot(rl, rr, l, phi_l, phi_r)
robot_config1.simulate(total_time, dt)

# Get final state
final_x, final_y, final_theta = robot_config1.get_state()

print("Configuration 1 Results:")
print(f"Final Position: ({final_x:.4f}, {final_y:.4f}) m")
print(f"Final Orientation: {final_theta:.4f} rad")
print(f"Final Orientation: {math.degrees(final_theta):.2f}°")


# 2. Configuration 2: $r_l = 3cm$, $r_r = 6cm$, $l = 10 cm$, $\dot{\phi}_l = 2.0 rad/s$, $\dot{\phi}_r = 2.0 rad/s$

# In[22]:


# YOUR CODE HERE
rl = 0.03  # 3 cm
rr = 0.06  # 3 cm
l = 0.10    # 10 cm
phi_l = 2.0  # rad/s
phi_r = 2.0  # rad/s

# Simulation parameters
dt = 0.02  # s
total_time = 3.0  # s

robot_config1 = Diff_Drive_Robot(rl, rr, l, phi_l, phi_r)
robot_config1.simulate(total_time, dt)

# Get final state
final_x, final_y, final_theta = robot_config1.get_state()

print("Configuration 1 Results:")
print(f"Final Position: ({final_x:.4f}, {final_y:.4f}) m")
print(f"Final Orientation: {final_theta:.4f} rad")
print(f"Final Orientation: {math.degrees(final_theta):.2f}°")


# 3. Configuration 3: $r_l = 0.06m$, $r_r = 0.03m$, $l = 10 cm$, $\dot{\phi}_l = 57.3 deg/s$, $\dot{\phi}_r = 114.6 deg/s$

# In[14]:


# YOUR CODE HERE
rl = 0.06  # 3 cm
rr = 0.03  # 3 cm
l = 0.10    # 10 cm
phi_l = 57.3 * (np.pi/180) # rad/s
phi_r = 114.6 * (np.pi/180)  # rad/s

# Simulation parameters
dt = 0.02  # s
total_time = 3.0  # s

robot_config1 = Diff_Drive_Robot(rl, rr, l, phi_l, phi_r)
robot_config1.simulate(total_time, dt)

# Get final state
final_x, final_y, final_theta = robot_config1.get_state()

print("Configuration 1 Results:")
print(f"Final Position: ({final_x:.4f}, {final_y:.4f}) m")
print(f"Final Orientation: {final_theta:.4f} rad")
print(f"Final Orientation: {math.degrees(final_theta):.2f}°")


# ## Motion tracking [10 points]
# 
# Use the code you wrote previously to plot/track the movement of the robot given in the `data/encoder.dat` file. In the `encoder.dat` file, the left column represents encoder measurement of the left wheel, and the right column encoder measurements of the right wheel. The total number of encoder ticks is read at 10Hz and the counter is never reset, i.e. the measurements are cumulative. In this encoder, the number of ticks per rotation of a wheel is 4000. Helper code to get the angular velocity from the encoder measurements is provided below.
# 
# Plot the movement of the robot as a line (not as scatter plot) given the robot has the following different configurations of wheels. Write your observations of the effect of the wheel radii on the motion of the robot.
# 
# Optionally, check the effect of varying the distance between the wheels and comment on its qualitative effect.
# 
# **Configuration 1:** $r_l = 3cm$, $r_r = 3cm$

# In[ ]:


# Configuration {config_num}: r_l={r_l_cm}cm, r_r={r_r_cm}cm, l={l_cm}cm
# This code block fulfills the requirement to provide the helper code for the configuration.
def get_angular_velocity(file_name, frequency, ticks_per_rotation):
    data = np.genfromtxt(file_name)
    ticks_count = np.zeros_like(data)
    ang_vel = np.zeros_like(data)
    freq = 10
    ticks_per_rotation = 4000

    for i in range (1,len(data)):
        ticks_count[i] = data[i]-data[i-1]

    ang_vel = (ticks_count*2*np.pi*freq) /ticks_per_rotation
    
    return ang_vel


# In[ ]:


def get_angular_velocity(file_name, frequency, ticks_per_rotation):
    data = np.genfromtxt(file_name)
    ticks_count = np.zeros_like(data)
    ang_vel = np.zeros_like(data)
    freq = 10
    ticks_per_rotation = 4000

    for i in range (1,len(data)):
        ticks_count[i] = data[i]-data[i-1]

    ang_vel = (ticks_count*2*np.pi*freq) /ticks_per_rotation
    
    return ang_vel


# In[24]:


ticks_per_rotation = 4000
frequency = 10 # Hz
dt = 1.0 / frequency
file_name = 'data/encoder.dat' # modify the path according to relative location
ang_vel = get_angular_velocity(file_name, frequency, ticks_per_rotation)

# YOUR CODE HERE

class Diff_Drive_Robot:
    def __init__(self, rl, rr, l):
        """Robot initialized at origin facing along x-axis."""
        self.rl = rl
        self.rr = rr
        self.l = l
        self.state = np.array([0.0, 0.0, 0.0])  # x, y, theta

    def state_update(self, phi_l, phi_r, dt):
        """Update state given angular velocities."""
        x, y, theta = self.state
        
        # Linear velocities of wheels
        vl = self.rl * phi_l
        vr = self.rr * phi_r

        # Linear and angular velocity of the robot
        v = (vr + vl) / 2.0
        omega = (vr - vl) / (2 * self.l)

        # Update equations
        if abs(omega) < 1e-10:
            x_new = x + v * math.cos(theta) * dt
            y_new = y + v * math.sin(theta) * dt
            theta_new = theta
        else:
            ICC_R = v / omega
            delta_theta = omega * dt
            x_new = x + ICC_R * (math.sin(theta + delta_theta) - math.sin(theta))
            y_new = y - ICC_R * (math.cos(theta + delta_theta) - math.cos(theta))
            theta_new = theta + delta_theta

        self.state = np.array([x_new, y_new, theta_new])
        return self.state
# ---- Helper function ----
def get_angular_velocity(file_name, frequency, ticks_per_rotation):
    data = np.genfromtxt(file_name)
    ticks_count = np.zeros_like(data)
    ang_vel = np.zeros_like(data)
    
    for i in range(1, len(data)):
        ticks_count[i] = data[i] - data[i-1]
    
    ang_vel = (ticks_count * 2 * np.pi * frequency) / ticks_per_rotation
    return ang_vel




# Extract left and right wheel angular velocities
phi_l = ang_vel[:, 0]
phi_r = ang_vel[:, 1]

# ---- Robot configuration ----
rl = 0.03  # 3 cm
rr = 0.03  # 3 cm
l = 0.10   # 10 cm between wheels

robot = Diff_Drive_Robot(rl, rr, l)

# ---- Simulate ----
trajectory = []
for i in range(len(phi_l)):
    state = robot.state_update(phi_l[i], phi_r[i], dt)
    trajectory.append(state)

trajectory = np.array(trajectory)

# ---- Plot ----
plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
plt.title("Robot Trajectory (Configuration 1: rl=3cm, rr=3cm)")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.axis("equal")
plt.grid(True)
plt.show()


# **Configuration 2:** $r_l = 3cm$, $r_r = 6cm$

# In[25]:


# YOUR CODE HERE
rl = 0.03  # 3 cm
rr = 0.06  # 6 cm
l = 0.10   # 10 cm between wheels

robot = Diff_Drive_Robot(rl, rr, l)

# ---- Simulate ----
trajectory = []
for i in range(len(phi_l)):
    state = robot.state_update(phi_l[i], phi_r[i], dt)
    trajectory.append(state)

trajectory = np.array(trajectory)

# ---- Plot ----
plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
plt.title("Robot Trajectory (Configuration 1: rl=3cm, rr=3cm)")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.axis("equal")
plt.grid(True)
plt.show()


# **Configuration 3:** $r_l = 6cm$, $r_r = 3cm$

# In[26]:


# YOUR CODE HERE
rl = 0.06  # 6 cm
rr = 0.03  # 3 cm
l = 0.10   # 10 cm between wheels

robot = Diff_Drive_Robot(rl, rr, l)

# ---- Simulate ----
trajectory = []
for i in range(len(phi_l)):
    state = robot.state_update(phi_l[i], phi_r[i], dt)
    trajectory.append(state)

trajectory = np.array(trajectory)

# ---- Plot ----
plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
plt.title("Robot Trajectory (Configuration 1: rl=3cm, rr=3cm)")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.axis("equal")
plt.grid(True)
plt.show()

### Write your observations below as a comment ###


# # B. Odometry motion model [50 points]
# 
# As you are already familiar with running the Robile in simulation as well as the concepts of different frames associated with it, we will now look into controlling the robot in simulation by sending commands to the `/cmd_vel` topic.
# 
# Consider a scenario where the robot has to reach a particular pose (i.e. a position and an orientation) with respect to the `odom` frame. For example, consider $(x, y, \theta) = (2.0, -3.0, -2.0)$ as a desired pose with respect to the `odom` frame (where $\theta$ is, as usual, in radians), but please feel free to vary the goal pose while testing your implementation.
# 
# In this scenario, we will implement a simple motion model (called an odometry motion model) that decomposes the process of moving towards the goal into three motion components:
# 1. Rotating towards the goal position and stopping the motion when facing the goal.
# 2. Moving straight towards the goal until the robot's position overlaps with the goal.
# 3. Rotating the robot until it reaches the goal orientation.
# 
# For this exercise, you will write your own ROS2 node, which allows you to send a goal request with the target pose with respect to `base_link`.
# 
# *Hints*:
# * If need be, please refer to the [documentation](!https://robile-amr.readthedocs.io/en/latest/source/Tutorial/Demo%20Simulation.html) for the Robile simulation.
# * As a reminder, the direction in front of the robot is the `x-axis` of the `base_link` frame (and base link is a right-handed coordinate frame).
# * The odometry motion model does not do any obstacle avoidance, so please make sure that the goal is within reach of the robot without any obstacles in the way while testing.
# * Due to noise, your robot will never be able to reach the goal exactly - for a robust behaviour, you need to use thresholds for determining when the desired position and orientation have been reached.
# * As this exercise requires a ROS implementation, keep in minde that you need to work on this component and perform all tests on your local machine. Once you are done, please paste the working code of the node in the cell. Please also include screenshots to demonstrate your tests in the cell further below.

# In[ ]:


# YOUR CODE HERE
raise NotImplementedError()


# YOUR ANSWER HERE
