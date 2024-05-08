import numpy as np

class Robot:
    def __init__(self):
        # Constants for number of states and actions
        self.S_n = 2  # States: 'high', 'low'
        self.A_n = 3  # Actions: 'search', 'wait', 'recharge'

        # Constants for transition probabilities
        self.alpha = 0.1  # Pr(stay at high charge if searching | now have high charge)
        self.beta = 0.1   # Pr(stay at low charge if searching | now have low charge)

        # Constants for rewards
        self.r_search = 2   # reward for searching
        self.r_wait = 1     # reward for waiting
        self.r_rescue = -3  # reward (actually penalty) for running out of charge

        # Transition probabilities matrix P
        self.P = np.zeros((self.S_n, self.A_n, self.S_n))

        # Rewards matrix r
        self.r = np.zeros((self.S_n, self.A_n))

        # Initialize the matrices
        self.init_matrices()

    def init_matrices(self):
        # Action 'search' (action index 0)
        self.P[0, 0, 0] = self.alpha     # High to High
        self.P[0, 0, 1] = 1 - self.alpha # High to Low
        self.P[1, 0, 0] = 1 - self.beta  # Low to High (rescue)
        self.P[1, 0, 1] = self.beta      # Low to Low

        # Action 'wait' (action index 1)
        self.P[0, 1, 0] = 1  # High stays High
        self.P[1, 1, 1] = 1  # Low stays Low

        # Action 'recharge' (action index 2)
        self.P[1, 2, 0] = 1  # Low goes to High

        # Rewards for actions
        self.r[0, 0] = self.r_search  # High state, search
        self.r[1, 0] = self.r_rescue  # Low state, search
        self.r[0, 1] = self.r_wait    # High state, wait
        self.r[1, 1] = self.r_wait    # Low state, wait
        self.r[1, 2] = 0              # Low state, recharge

    def get_transition_probabilities(self):
        return self.P

    def get_rewards(self):
        return self.r

# Example usage
# robot = Robot()
# print("Transition Probability Matrix:")
# print(robot.get_transition_probabilities())

# print("Rewards Matrix:")
# print(robot.get_rewards())
