import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import Bounds, LinearConstraint
from numpy.linalg import norm
def generate_mdp(S_n, A_n, reward_range=(0, 1)):
    """
    Generate a Markov Decision Process (MDP) represented by states, actions, randomly generated rewards, and transition probabilities.

    Args:
    S_n (int): Number of states in the MDP.
    A_n (int): Number of actions in the MDP.
    reward_range (tuple): Tuple representing the minimum and maximum range of rewards.

    Returns:
    tuple: A tuple containing:
        - r (list of lists): Reward matrix where r[i][j] is the reward for taking action j in state i.
        - P (numpy array): Transition probability matrix where P[i][j][k] is the probability of transitioning from state i to state k by taking action j.
    """
    # Initialize rewards randomly within the specified range
    r = [[np.random.randint(reward_range[0], reward_range[1] + 1) for _ in range(A_n)] for _ in range(S_n)]

    # Define P as transition kernel
    # Initialize P with random values
    P = np.random.rand(S_n, A_n, S_n)

    # Normalize so that each row sums to 1 (to represent probabilities)
    for i in range(S_n):
        for j in range(A_n):
            P[i][j, :] /= np.sum(P[i][j, :])

    return r, P


def objective_function(x, V, uncertainty, P, reg_term=0.01):
    return uncertainty * x + x * np.log(max(0.0001, np.inner(np.exp(-np.array(V) / (x + reg_term)), P) + reg_term))

def sigma(V, P, uncertainty):
    C1 = Bounds(np.zeros(1), np.inf * (np.ones(1)))
    res = minimize(
        objective_function,
        x0=np.random.random(1),
        args=(V, uncertainty, P),
        bounds=C1,
    )
    return -1 * res.fun


'''
def non_robust_vi(pi, S_n, A_n, r, P, epsilon=0.000001, max_iterations=1000, check_if=False):
    V = np.zeros(S_n)  # initialize value function
    V_prev = np.zeros(S_n)  # to store the previous value function for convergence check

    for iteration in range(max_iterations):
        V_prev = V.copy()  # store the previous value function for convergence check
        for state in range(S_n):
            V[state] = max([r[state][act] + sum([P[state][act][next_state] * V_prev[next_state] for next_state in range(S_n)]) for act in range(A_n)])
        
        # Check for convergence using spectral norm
        if check_if and norm(V - V_prev, ord=2) < epsilon:
            print(f"Convergence reached in {iteration + 1} iterations.")
            break

    Q = np.zeros((S_n, A_n))
    for state in range(S_n):
        for act in range(A_n):
            Q[state][act] = r[state][act] + sum([P[state][act][next_state] * V[next_state] for next_state in range(S_n)])

    return V, Q
'''

def non_robust_vi(pi, S_n, A_n, r, P, s_star=0, epsilon=0.000001, max_iterations=1000, check_if=False):
    V = np.zeros(S_n)  # initialize value function
    w = V.copy()  # initialize w at t=0
    w -= V[s_star]  # adjust relative to reference state

    for iteration in range(max_iterations):
        w_prev = w.copy()  # store the previous w for convergence check
        for state in range(S_n):
            V[state] = sum(r[state][act] + np.inner(P[state][act], w) for act in range(A_n))
            w[state] = V[state] - V[s_star]

        # Check for convergence using spectral norm
        if check_if and norm(w - w_prev, ord=2) < epsilon:
            print(f"Convergence reached in {iteration + 1} iterations.")
            break

    Q = np.zeros((S_n, A_n))
    for state in range(S_n):
        for act in range(A_n):
            Q[state][act] = r[state][act] + sum([P[state][act][next_state] * V[next_state] for next_state in range(S_n)]) - V[s_star]

    return w, V, Q

def robust_rvi(pi, S_n, A_n, r, P, s_star=0, epsilon=0.000001, max_iterations=1000, uncertainty=0.1, check_if=False):
    """
    Perform Robust Relative Value Iteration (RVI) for a given MDP.

    Args:
        S_n (int): Number of states.
        A_n (int): Number of actions.
        r (np.array): Reward matrix.
        P (np.array): Transition probability matrix.
        s_star (int): Reference state.
        epsilon (float): Convergence threshold.
        max_iterations (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        np.array, np.array: Final relative value function and value function.
    """
    V = np.zeros(S_n)  # initialize value function
    w = V.copy()  # initialize w at t=0
    w -= V[s_star]  # adjust relative to reference state

    for iteration in range(max_iterations):
        w_prev = w.copy()  # store the previous w for convergence check
        for state in range(S_n):
            V[state] = sum(pi[state][act]*(r[state][act] + sigma(w, P[state][act], uncertainty)) for act in range(A_n))
            w[state] = V[state] - V[s_star]
        
        # Check for convergence using spectral norm
        if check_if and norm(w - w_prev, ord=2) < epsilon:
            print(f"Convergence reached in {iteration + 1} iterations.")
            break

    Q = np.zeros((S_n, A_n))
    for state in range(S_n):
        for act in range(A_n):
            Q[state][act] = r[state][act] + sum(P[state][act][next_state]*V[next_state] for next_state in range(S_n)) - V[s_star]

    return w, V, Q

def robust_rvi_baseline(pi, S_n, A_n, r, P, s_star=0, epsilon=0.000001, max_iterations=1000, uncertainty=0.1, check_if=False):
    """
    Perform Robust Relative Value Iteration (RVI) for a given MDP.

    Args:
        S_n (int): Number of states.
        A_n (int): Number of actions.
        r (np.array): Reward matrix.
        P (np.array): Transition probability matrix.
        s_star (int): Reference state.
        epsilon (float): Convergence threshold.
        max_iterations (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        np.array, np.array: Final relative value function and value function.
    """
    V = np.zeros(S_n)  # initialize value function
    w = V.copy()  # initialize w at t=0
    w -= V[s_star]  # adjust relative to reference state

    for iteration in range(max_iterations):
        w_prev = w.copy()  # store the previous w for convergence check
        for state in range(S_n):
            V[state] = max([(1-uncertainty)*r[state][act] + sigma(w, P[state][act], uncertainty) for act in range(A_n)])
            w[state] = V[state] - V[s_star]
        
        # Check for convergence using spectral norm
        if check_if and norm(w - w_prev, ord=2) < epsilon:
            print(f"Convergence reached in {iteration + 1} iterations.")
            break

    Q = np.zeros((S_n, A_n))
    for state in range(S_n):
        for act in range(A_n):
            Q[state][act] = r[state][act] + sum(P[state][act][next_state]*V[next_state] for next_state in range(S_n)) - V[s_star]

    return w, V, Q


def generate_random_policy(S_n, A_n):
    """
    Generate a random policy for an MDP with state number S_n and action number A_n.

    Args:
        S_n (int): The number of states in the MDP.
        A_n (int): The number of actions in the MDP.

    Returns:
        np.array: A stochastic policy represented as a 2D array where each row corresponds to a state
                  and each column corresponds to the probability of taking an action in that state.
    """
    # Initialize a random policy as a 2D array filled with random values
    policy = np.random.rand(S_n, A_n)
    
    # Normalize the policy to ensure that the probabilities sum to 1 for each state
    policy /= policy.sum(axis=1, keepdims=True)
    
    return policy

def non_robust_policy_gradient(pi, Q, alpha=0.1):
    """
    Perform a robust policy gradient update step.

    Args:
        pi (list): Current policy pi_k.
        Q (list): The robust Q-values.
        alpha (float): Step size for the gradient descent.

    Returns:
        list: Updated and projected policy pi_{k+1}.
    """

    S_n = len(pi)       # Number of states.
    A_n = len(pi[0])    # Number of actions.
    pi_next = np.zeros((S_n, A_n))  # Initialize the updated policy.

    # Gradient descent to update the policy.
    for state in range(S_n):
        for act in range(A_n):
            pi_next[state, act] = pi[state][act] + alpha * Q[state][act]
    
    pi_next /= pi_next.sum(axis=1, keepdims=True)
    return pi_next

def robust_policy_gradient(pi, Q, alpha=0.1):
    """
    Perform a robust policy gradient update step.

    Args:
        pi (list): Current policy pi_k.
        Q (list): The robust Q-values.
        alpha (float): Step size for the gradient descent.

    Returns:
        list: Updated and projected policy pi_{k+1}.
    """

    S_n = len(pi)       # Number of states.
    A_n = len(pi[0])    # Number of actions.
    pi_next = np.zeros((S_n, A_n))  # Initialize the updated policy.

    # Gradient descent to update the policy.
    for state in range(S_n):
        for act in range(A_n):
            pi_next[state, act] = pi[state][act] + alpha * Q[state][act]

    # Project the updated policy back to the probability simplex.
    def policy_projection(pi_next_state):
        def project(x):
            return np.sum((x - pi_next_state) ** 2)

        def cons(x):
            return 1 - np.sum(x)

        constraints = [{'type': 'eq', 'fun': cons}]
        bnds = [(0, 1)] * A_n
        init = np.full(A_n, 1 / A_n)

        res = minimize(project, init, bounds=bnds, constraints=constraints)
        return res.x

    # Apply projection to each state.
    projected_pi = [policy_projection(pi_next_state) for pi_next_state in pi_next]

    return projected_pi

# Example of usage
S_n = 5  # number of states
A_n = 2  # number of actions
r, P = generate_mdp(S_n, A_n)

# from robot import *
# from inventory import *
# robot = Robot()
# inventory = Inventory()

# r, P = robot.get_rewards(), robot.get_transition_probabilities()
# r, P = inventory.get_rewards(), inventory.get_transition_probabilities()

# S_n, A_n = 17, 9
pi = generate_random_policy(S_n, A_n)

# pi = [[1/A_n]*A_n for _ in range(S_n)]  # Uniform policy initialization

# pi = [[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]
# r = [[1, 2, 3], [1, 2, 3], [4, 2, 1], [3, 3, 1]]

# pi = [[1, 0, 0], [1, 0, 0]]
# r = [[1, 2, 3], [1, 2, 3]]
pi_n = pi.copy()
pi_b = pi.copy()
print("S_n = ", S_n)
print("A_n = ", A_n)
print("Rewards:", r)
print("Transition Probabilities:", P)
print("Generated stochastic policy (π):")
print(pi)

print("----------------------------------testing algorihtms")
w, V, Q= robust_rvi(pi, S_n, A_n, r, P)
print("w = ", w)
print("V = ", V)
print("Q = ", Q)

projected_pi = robust_policy_gradient(pi, Q, alpha=0.1)
print("projected_pi = ", projected_pi)

print("----------------------------------start to learn")
for step in range(50):
    print("step = ", step)
    _, V, Q= robust_rvi(pi, S_n, A_n, r, P, uncertainty=0.1, check_if=False)
    _, V_b, Q_b = robust_rvi_baseline(pi_b, S_n, A_n, r, P, uncertainty=0.1, check_if=False)
    _, _, Q_n = non_robust_vi(pi_n, S_n, A_n, r, P, check_if=False)
    
    _, V_n, _ = robust_rvi(pi_n, S_n, A_n, r, P, uncertainty=0.1, check_if=False)
    
    print("V_r", V)
    print("V_n", V_n)
    print("V_b", V_b)
    pi = robust_policy_gradient(pi, Q, alpha=0.1)
    pi_n = robust_policy_gradient(pi_n, Q_n, alpha=0.1)
    pi_b = robust_policy_gradient(pi_b, Q_b, alpha=0.1)

print(pi)
