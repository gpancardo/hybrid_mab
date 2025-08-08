class SimEnv:
    def __init__(self, reward_probs, noise=0.0, epistemic_info=False):
        """
        Creates an environment for a multi-armed bandit problem with 3 arms.

        Args:
            reward_probs (list[float]): Probability of reward per arm (0 to 1).
            noise (float): Aditional uncertainty to the reward. Default 0.0.
            epistemic_info (bool): Signals if the environment has to include epistemic clues for the agents that need them.
        """
        assert len(reward_probs) == 3
        assert all(0 <= p <= 1 for p in reward_probs)

        self.n_arms = 3
        self.reward_probs = reward_probs
        self.noise = noise
        self.epistemic_info = epistemic_info
        
        # Internal variables to save action histoty if needed.
        self.round = 0
        self.history = [] 

        #Finds optimal arm.
        self.optimal_arm = self.reward_probs.index(max(self.reward_probs))

    def reset(self):
        #Resets the environment to its initial state.
        self.round = 0
        self.history = []
        return self.get_observation()
    