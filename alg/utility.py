import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.optimize import Bounds, LinearConstraint
from numpy.linalg import norm

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