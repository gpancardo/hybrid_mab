import random as rd

class SimEnv:
    def __init__(self, reward_probs, noise=0.0, epistemic_info=False):
        """
        Creates an environment for a multi-armed bandit problem with 3 arms.

        Args:
            reward_probs (list[float]): Probability of reward per arm (0 to 1).
            noise (float): Aditional uncertainty to the reward. Default 0.0.
            epistemic_info (bool): Signals if the environment has to include epistemic clues for the agents that need them.
        """
        if ((len(reward_probs) == 3) and  (all(0 <= p <= 1 for p in reward_probs))):

            self.n_arms = 3
            self.reward_probs = reward_probs
            self.noise = noise
            self.epistemic_info = epistemic_info
            
            # Internal variables to save action histoty if needed.
            self.round = 0
            self.history = [] 

            #Finds optimal arm.
            self.optimal_arm = self.reward_probs.index(max(self.reward_probs))
        else:
            print("Error: The environment must have exactly 3 arms with probabilities between 0 and 1.")

    def reset(self):
        #Resets the environment to its initial state.
        self.round = 0
        self.history = []
        return self.get_observation()
    
    #Get the true robabilities
    def get_true_probs(self):
        return list(self.reward_probs)
    
    #Run round
    def step(self, action):
        #Probability
        if (action < 0 or action >= self.n_arms):
            self.round+=1
            true_prob = self.reward_probs[action]
            effective_prob=min(max(true_prob + rd.uniform(-self.noise, self.noise), 0), 1)
            #Calculating reward
            if (rd.random() < effective_prob):
                reward = 1
            else:
                reward = 0
            self.history.append(
                {
                    "round": self.round,
                    "action": action,
                    "reward": reward
                }
            )
            if self.epistemic_info:
                info={"optimal_arm":self.optimal_arm}
                info["epistemic_clue"]=self.generate_epistemic_clue(action)
            return reward, info

        else:
            print("Error: Action must be between 0 and 2.")