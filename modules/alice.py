import numpy as np
class Alice:
    def __init__(self, n_arms, config):
        self.n_arms = n_arms
        self.learning_rate = config.get("learning_rate", 0.1)
        # Epsilon-greedy exploration strategy with decay
        self.epsilon = config.get("epsilon", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.99)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        #Sets initial values for each arm
        self.values=np.zeros(n_arms)

    def choose_action(self):
        #Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        #Exploitation
        else:
            return np.argmax(self.values)
    
    #Update the arm's value based on received reward
    def update(self, action, reward):
        error=reward - self.values[action]
        self.values[action] += self.learning_rate * error
        #Epsilon decay
        if (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay
    
    def get_internal_state(self):
        return {
            "values": self.values.copy(),
            "epsilon": self.epsilon
        }
    