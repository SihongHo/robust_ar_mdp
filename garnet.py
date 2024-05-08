import numpy as np

class Garnet:
    def __init__(self, S_n, A_n):
        """
        Initialize the Markov Decision Process with a given number of states and actions, 
        and a range for the rewards.

        Args:
        S_n (int): Number of states in the MDP.
        A_n (int): Number of actions in the MDP.
        reward_range (tuple): Tuple representing the minimum and maximum range of rewards.
        """
        self.S_n = S_n
        self.A_n = A_n
        self.r, self.P = self.generate_mdp()

    def generate_mdp(self):
        """
        Generate the rewards matrix and transition probability matrix for the MDP.

        Returns:
        tuple: A tuple containing:
            - r (list of lists): Reward matrix where r[i][j] is the reward for taking action j in state i.
            - P (numpy array): Transition probability matrix where P[i][j][k] is the probability of transitioning from state i to state k by taking action j.
        """
        # Initialize rewards randomly within the specified range
        r = [[np.random.rand() for _ in range(self.A_n)] for _ in range(self.S_n)]

        # Define P as transition kernel
        # Initialize P with random values
        P = np.random.rand(self.S_n, self.A_n, self.S_n)

        # Normalize so that each row sums to 1 (to represent probabilities)
        for i in range(self.S_n):
            for j in range(self.A_n):
                P[i][j, :] /= np.sum(P[i][j, :])

        return r, P

    def get_rewards(self):
        """
        Returns the reward matrix of the MDP.
        """
        return self.r

    def get_transition_probabilities(self):
        """
        Returns the transition probability matrix of the MDP.
        """
        return self.P
