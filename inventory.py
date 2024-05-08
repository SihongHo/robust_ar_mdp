import numpy as np

class Inventory:
    def __init__(self):
        self.states = np.arange(0, 17)  # Inventory levels from 0 to 16
        self.actions = np.arange(0, 9)  # Possible orders from 0 to 8 units
        
        # Assuming max demand we are considering is 29 (must cover all realistic demands)
        self.max_demand = 29
        self.S_n = len(self.states)
        self.A_n = len(self.actions)
        
        # Initialize transition probabilities matrix
        self.transition_probs = np.zeros((len(self.states), len(self.actions), len(self.states)))
        
        # Demand distribution (assuming some arbitrary distribution for demonstration)
        self.demand_prob = np.random.rand(self.max_demand + 1)  # Demand probabilities for values from 0 to max_demand
        self.demand_prob /= self.demand_prob.sum()  # Normalize to make it a valid probability distribution

        # Initialize rewards matrix
        self.rewards = np.zeros((len(self.states), len(self.actions)))

        # Costs and rewards
        self.order_cost = 3
        self.holding_cost = 3
        self.sale_price = 5
        self.penalty_cost = -15

        self.compute_transition_probabilities()
        self.compute_rewards()

    def compute_transition_probabilities(self):
        # Calculate transition probabilities
        for s in range(len(self.states)):
            for a in range(len(self.actions)):
                for d in range(self.max_demand + 1):
                    next_state = max(0, s + a - d)  # Calculate the next state after demand is met
                    if next_state < len(self.states):
                        self.transition_probs[s, a, next_state] += self.demand_prob[d]
    
    def get_transition_probabilities(self):
        return self.transition_probs

    def get_rewards(self):
        return self.rewards

    def compute_rewards(self):
        # Compute rewards considering each state and action
        for s in range(len(self.states)):
            for a in range(len(self.actions)):
                expected_reward = 0
                for d in range(self.max_demand + 1):
                    if d <= s + a:
                        reward = (self.sale_price * d) - (self.holding_cost * (s + a))
                    else:
                        reward = self.penalty_cost
                    expected_reward += reward * self.demand_prob[d]
                expected_reward -= self.order_cost * a
                self.rewards[s, a] = expected_reward
        return self.rewards

# Example usage
# inventory = Inventory()

# prob_matrix = inventory.get_transition_probabilities()

# print("Transition Probability Matrix:")
# print(prob_matrix)  

# print("Transition Probability Matrix for a specific state and action:")
# print(inventory.transition_probs[0, 1])  # Example: transition probabilities from state 0 with action 1

# rewards_matrix = inventory.get_rewards()  # Compute rewards matrix

# print("Rewards Matrix:")
# print(rewards_matrix)
