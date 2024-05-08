import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import Bounds, LinearConstraint
import numpy as np
import math
# import robust_pg

#########################################
###############define MDP################
#########################################
# Robust Value Iteration for Average Reward MDP
S_n = 4  # number of state
A_n = 3  # number of action

r = [[1, 2, 3], [1, 2, 3], [4, 2, 1], [3, 3, 1]] # reward

# Define P as transition kernel
# Initialize P with random values
P = np.random.rand(S_n, A_n, S_n)

# Normalize so that each row sums to 1 (to represent probabilities)
for i in range(S_n):
    for j in range(A_n):
        P[i][j, :] /= np.sum(P[i][j, :])
#########################################
###############define MDP################
#########################################
# S_n, A_n = 4, 3
# rewards, transitions = robust_pg.generate_mdp(S_n, A_n)

#########################################
########define uncertainty set###########
#########################################
# Consider KL-Divergence uncertainty set
V_ro = np.zeros(S_n)  # initialize relative value function
s_star = 0           # choose an arbitrary reference state
w = np.zeros(S_n)   # Here w[s] = V_ro[s] - V_ro[s_star] for any s


# Solve the worst-case value function under the KL Divergence uncertainty set using its dual formulation.
def objective_function(x, V, R, distribution):

    return R * x + x * np.log(np.inner(np.exp(-np.array(V) / (x + 0.001)), distribution) + 0.001)


# Define the optimization problem.
# Here 'V' is value function, 'distribution' is the transition kernel, 'R' is the radius of the KL divergence uncertainty set
def sigma(V, distribution, R):
    C1 = Bounds(np.zeros(1), np.inf * (np.ones(1)))
    res = minimize(
        objective_function,
        x0=np.random.random(1),
        args=(V, R, distribution),
        bounds=C1,
    )

    return -1 * res.fun
#########################################
########define uncertainty set###########
#########################################


#########################################
########robust value iteration###########
#########################################
# Robust Value iteration to solve the worst-case value function
# Let pi be the current policy
R = 0.1  # Radius of uncertainty set
pi = [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]
for step in range(100):
    w_next = np.zeros(S_n)
    for state in range(S_n):
            V_ro[state] = sum(pi[state][act]*(r[state][act] + sigma(w, P[state][act], R)) for act in range(A_n))
            w_next[state] = V_ro[state] - V_ro[s_star]
    w = w_next


# Compute Q(s, a) using V(s)
Q_ro = np.zeros((S_n, A_n))

for state in range(S_n):
    for act in range(A_n):
        Q_ro[state][act] = r[state][act] + sum(P[state][act][next_state]*V_ro[next_state] for next_state in range(S_n)) - V_ro[s_star] # Here V_ro[star] can be viewed as average reward.
#########################################
########robust value iteration###########
#########################################

#########################################
########robust policy gradient###########
#########################################
# We can then use Q_ro to do the gradient descent
# pi is the current policy, which is pi_k in the algorithm
pi_next = [[0]*A_n for _ in range(S_n)] # Define the updated policy
print("pi_next = ", pi_next)
# pi_next = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
alpha = 0.1 # step size

# The first step is gradient descent
for state in range(S_n):
    for act in range(A_n):
        pi_next[state][act] = pi[state][act] - alpha*Q_ro[state][act]


# The second step is to project pi_next to the collection of all policies
# We do this for each state separately
def policy_projection(pi_next_state):

    pi = pi_next_state

    # We project the policy by minimizing the Euclidean distance
    def project(x, pi):
        return sum((x[act]-pi[act])**2 for act in range(A_n))

    # Constraint: sum of x equals to 1
    def cons(x):
        return 1 - x[0] - x[1] - x[2]

    args = (pi)
    constraints = []
    constraints.append({'type': 'eq', 'fun': cons})

    # Bounds: each entry of x is in [0, 1]
    bnds = []
    for i in range(A_n):
        bnds.append([0, 1])

    # Initialization
    init = [1 / 3] * A_n

    # Solve the optimization
    res = minimize(project, init, args=args, bounds=bnds, constraints=constraints)
    pi_project = res.x

    return pi_project


# Initialize the projected policy
projected_pi = []

for state in range(S_n):
    projected_pi.append(policy_projection(pi_next[state]))

print("Updated policy pi_{k+1} after projection:", projected_pi)
# Therefore, we performed a one step update to get the policy projected_pi, which is pi_{k+1} in the algorithm
#########################################
########robust policy gradient###########
#########################################
