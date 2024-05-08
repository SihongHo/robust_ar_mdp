from alg.utility import *
from alg.robust_alg import *
import pickle
import os

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

def train_non_robust(training_steps, pi_n, S_n, A_n, r, P, uncertainty, alpha, save_path):
    pi_history, V_history = [], []
    for step in range(training_steps):
        print("step = ", step)
        _, _, Q_n = non_robust_vi(pi_n, S_n, A_n, r, P, check_if=False)
        _, V_n, _ = robust_rvi(pi_n, S_n, A_n, r, P, uncertainty=uncertainty, check_if=False)

        print("V_n", V_n[0])
        pi_n = robust_policy_gradient(pi_n, Q_n, alpha)
        pi_history.append(pi_n.copy())  # 确保存储策略的副本
        V_history.append(V_n.copy())  # 确保存储值函数的副本

    # 检查路径是否存在，如果不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 存储所有数据
    with open(os.path.join(save_path, 'training_data_non_robust.pkl'), 'wb') as f:
        pickle.dump({'pi_history': pi_history, 'V_history': V_history}, f)

    return pi_n, pi_history, V_history  # 确保返回最后更新的策略
