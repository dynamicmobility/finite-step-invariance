import numpy as np
from scipy.linalg import solve_discrete_lyapunov, sqrtm
from matplotlib import pyplot as plt
from ellipsoids import Ellipsoid, plot_projection, sample_from
import CompassGaitWalker
import ExpanderContractor
import TBone
import jax
from datetime import datetime
import jax.numpy as jnp
from probabilistic_bounds import binomial_tail

N_VERIFY = 2000
N_STEPS = 10
filename = 'Ellipsoid_Invariant_20260324_181527.txt'

def parse_ellipsoid_file(filepath):
    """Parse ellipsoid file and return Ellipsoid object."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract Q matrix
    q_start = content.find('Q =') + 3
    q_end = content.find('c =')
    q_text = content[q_start:q_end].strip()
    # Parse Q matrix by extracting floating point numbers
    q_lines = [line.strip() for line in q_text.split('\n') if line.strip()]
    q_values = []
    for line in q_lines:
        # Remove brackets and split by whitespace
        numbers = line.replace('[', '').replace(']', '').split()
        q_values.extend([float(x) for x in numbers if x])

    N = int(np.sqrt(len(q_values)))
    Q = np.array(q_values).reshape(N, N)
    
    # Extract c vector
    c_start = content.find('c =') + 3
    c_text = content[c_start:].strip()
    c_lines = [line.strip() for line in c_text.split('\n') if line.strip()]
    c_values = []
    for line in c_lines:
        numbers = line.replace('[', '').replace(']', '').split()
        c_values.extend([float(x) for x in numbers if x])
    c = np.array(c_values)[:, np.newaxis]
    assert c.shape == (N, 1)

    e = Ellipsoid(N)
    e.Q = Q
    e.c = c
    
    return e

# Load ellipsoid from file
ellipsoid = parse_ellipsoid_file('/home/varun/Documents/DynamoLab/WalkerScenarioOptimization/Ellipsoid_Invariant_20260324_173847.txt')
samples = sample_from(ellipsoid, N_VERIFY).T
assert np.all(ellipsoid.contains_point(samples.T))


steps = [samples]
dists = np.empty((N_STEPS, N_VERIFY))
sys = CompassGaitWalker

for i in range(N_STEPS):
    # print(steps[-1])
    # print(sys.P_batch(steps[-1]))
    dists[i, :] = ellipsoid.mahalanobis_distance(steps[-1].T)
    steps.append(sys.P_batch(steps[-1]))

# Create 3D plot with color by step index
if samples.shape[1] >= 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.viridis(np.linspace(0, 1, N_STEPS))
    
    for i in range(N_STEPS):
        ax.scatter(steps[i][:, 0], steps[i][:, 1], steps[i][:, 2], 
                   c=[colors[i]], s=10, alpha=0.6, label=f'Step {i}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory colored by step index')
    plt.show()

plt.plot(range(N_STEPS), dists)
plt.show()


