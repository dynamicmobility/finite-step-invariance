# This python script illustrates an example of a step-to-step map that is stable within a small neighborhood of the equlibrium point,
# but unstable everywhere else.
# The objective is to find this set (or some robust approximation of it).

import numpy as np
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt
from ellipsoids import p_ball, project_ellipsoid_to_2d, plot_ellipse, in_ellipsoid

# This is the A-norm with offset v 
def h(x, A, v):
    return np.diag((x-v).T@A@(x-v))

# This is a toy step-to-step map (contracts within the A-norm unit ball but expands otherwise)
def P(x, A, v):
    return (x-v)*np.sqrt(h(x, A, v)) + v

# Generate the dataset
N = 1000
A = np.array([[2, 0], [0, 1]])
v = np.array([[1], [0]])

# Randomly sample points uniformly from some rectangular interval
x = np.random.uniform(-2, 2, size=(2, N))


opt, init_A, init_b = p_ball(x.T, 2)
init_b = init_b.flatten()

# Apply the transformation which contracts for x inside the A-norm ball
Px = P(x, A, v)


inside_inds = np.where(h(x, A, v) < 1) # 1-norm ball
outside_inds = np.where(h(x, A, v) >= 1) # 1-norm ball



# # Plotting the ball associated with 1-level set of h(x, A, v)
# rho = np.ones((100, ))
# phi = np.linspace(0, 2*np.pi, 100)
# pts_ball = np.linalg.inv(sqrtm(A))@np.vstack((rho*np.cos(phi), rho*np.sin(phi))) + v # The procedure should recover A and v (or something like that) given pairs of x, Px

# plt.plot(pts_ball[0, :], pts_ball[1, :], 'k-') 

################################################################################################################################

# Calculate ellipsoid of Px after 1 step
data_step1 = []
step = []
for p in x.T:
    step.append(P([[p[0]], [p[1]]], A, v).T.ravel())
    if in_ellipsoid(init_A, init_b, P([[p[0]], [p[1]]], A, v).T.ravel()):
        data_step1.append(p)

opt, step1_A, step1_b = p_ball(data_step1, 2)
step1_b = step1_b.flatten()
step = np.array(step).T


fig, ax = plt.subplots()

Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(init_A, init_b, 0, 1)
plot_ellipse(Q_proj, c_proj, r_proj, ax, color='blue')

Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(step1_A, step1_b, 0, 1)
plot_ellipse(Q_proj, c_proj, r_proj, ax, color='green')


ax.scatter(x[0, inside_inds], x[1, inside_inds], c='gray', marker='o') # Plot the x inside the A-norm ball
ax.scatter(x[0, outside_inds], x[1, outside_inds], c='black', marker='o') # Plot the x outside the A-norm ball
ax.scatter(step[0, inside_inds], step[1, inside_inds], c='gray', marker='x') # Plot Px inside the A-norm ball (should contract within the ball)
ax.scatter(step[0, outside_inds], step[1, outside_inds], c='black', marker='x') # Plot Px outside the A-norm ball (should get further away)
# changed plotting of Px to match data I check for set inclusion 
ax.set_title('After 1 Step')

plt.show() 

################################################################################################################################

# Calculate ellipsoid of Px after 2 step

data_step2 = []
step2 = []
for p in step.T:
    step2.append(P([[p[0]], [p[1]]], A, v).T.ravel())
    if in_ellipsoid(step1_A, step1_b, P([[p[0]], [p[1]]], A, v).T.ravel()):
        data_step2.append(p)

opt, step2_A, step2_b = p_ball(data_step2, 2)
step2_b = step2_b.flatten()
step2 = np.array(step2).T

fig, ax = plt.subplots()

Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(init_A, init_b, 0, 1)
plot_ellipse(Q_proj, c_proj, r_proj, ax, color='blue')

Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(step1_A, step1_b, 0, 1)
plot_ellipse(Q_proj, c_proj, r_proj, ax, color='green')

Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(step2_A, step2_b, 0, 1)
plot_ellipse(Q_proj, c_proj, r_proj, ax, color='cyan')

ax.scatter(x[0, inside_inds], x[1, inside_inds], c='gray', marker='o') # Plot the x inside the A-norm ball
ax.scatter(x[0, outside_inds], x[1, outside_inds], c='black', marker='o') # Plot the x outside the A-norm ball
ax.scatter(step[0, inside_inds], step[1, inside_inds], c='gray', marker='x') # Plot Px inside the A-norm ball (should contract within the ball)
ax.scatter(step[0, outside_inds], step[1, outside_inds], c='black', marker='x') # Plot Px outside the A-norm ball (should get further away)
# changed plotting of Px to match data I check for set inclusion 
ax.set_title('After 2 Steps')

plt.show() 

################################################################################################################################

# Calculate ellipsoid of Px after 3 step

data_step3 = []
step3 = []
for p in step2.T:
    step3.append(P([[p[0]], [p[1]]], A, v).T.ravel())
    if in_ellipsoid(step2_A, step2_b, P([[p[0]], [p[1]]], A, v).T.ravel()):
        data_step3.append(p)

opt, step3_A, step3_b = p_ball(data_step3, 2)
step3_b = step3_b.flatten()
step3 = np.array(step3).T

fig, ax = plt.subplots()

Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(init_A, init_b, 0, 1)
plot_ellipse(Q_proj, c_proj, r_proj, ax, color='blue')

Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(step1_A, step1_b, 0, 1)
plot_ellipse(Q_proj, c_proj, r_proj, ax, color='green')

Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(step2_A, step2_b, 0, 1)
plot_ellipse(Q_proj, c_proj, r_proj, ax, color='cyan')

Q_proj, c_proj, r_proj = project_ellipsoid_to_2d(step3_A, step3_b, 0, 1)
plot_ellipse(Q_proj, c_proj, r_proj, ax, color='pink')

ax.scatter(x[0, inside_inds], x[1, inside_inds], c='gray', marker='o') # Plot the x inside the A-norm ball
ax.scatter(x[0, outside_inds], x[1, outside_inds], c='black', marker='o') # Plot the x outside the A-norm ball
ax.scatter(step[0, inside_inds], step[1, inside_inds], c='gray', marker='x') # Plot Px inside the A-norm ball (should contract within the ball)
ax.scatter(step[0, outside_inds], step[1, outside_inds], c='black', marker='x') # Plot Px outside the A-norm ball (should get further away)
# changed plotting of Px to match data I check for set inclusion 
ax.set_title('After 3 Steps')

plt.show() 