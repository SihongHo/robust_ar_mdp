from utility import *
import os
import pickle

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

def train_robust(training_steps, pi, S_n, A_n, r, P, uncertainty, alpha, save_path):
    pi_history, V_history = [], []
    for step in range(training_steps):
        print("step = ", step)
        _, V, Q= robust_rvi(pi, S_n, A_n, r, P, s_star=0, epsilon=0.000001, max_iterations=1000, uncertainty=0.1, check_if=False)
        print("V_r", V[0])
        pi = robust_policy_gradient(pi, Q, alpha)
        pi_history.append(pi.copy())  # 确保存储策略的副本
        V_history.append(V.copy())  # 确保存储值函数的副本
    
    # 检查路径是否存在，如果不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 存储所有数据
    with open(os.path.join(save_path, 'training_data_robust_our.pkl'), 'wb') as f:
        pickle.dump({'pi_history': pi_history, 'V_history': V_history}, f)
    return pi, pi_history, V_history 


def train_robust_base(training_steps, pi_b, S_n, A_n, r, P, uncertainty, alpha, save_path):
    pi_history, V_history = [], []
    for step in range(training_steps):
        print("step = ", step)
        _, V_b, Q_b = robust_rvi_baseline(pi_b, S_n, A_n, r, P, s_star=0, epsilon=0.000001, max_iterations=1000, uncertainty=0.1, check_if=False)     
        print("V_b", V_b[0])
        pi_b = robust_policy_gradient(pi_b, Q_b, alpha)
        pi_history.append(pi_b.copy())  # 确保存储策略的副本
        V_history.append(V_b.copy())  # 确保存储值函数的副本

    # 检查路径是否存在，如果不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 存储所有数据
    with open(os.path.join(save_path, 'training_data_robust_base.pkl'), 'wb') as f:
        pickle.dump({'pi_history': pi_history, 'V_history': V_history}, f)
    return pi_b, pi_history, V_history 