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

    #Builds clues for the epistemic agents based on previous results to update beliefs.
    def generate_epistemic_clues(self, action, reward):
        import numpy as np
        clues = {"discard_arms": []}

        # The confidence threshold and noise level can be adjusted based on the agent's epistemic model.
        #The confidence threshold is the minimum certainty required to discard an arm, while the noise level simulates uncertainty based on the point of view of the agent.
        confidence_threshold = 0.7
        noise_level = 0.2

        #Evaluates if any arm can be ouright discarted
        for arm in range(self.n_arms):
            if arm == action:
                continue

            # Subjective stimation of optimality
            prob_optimal = self.reward_probs[arm]

            #If there was a reward, the confidence that this arm is not optimal decreases.
            if reward == 1:
                confidence = max(0, 1 - prob_optimal - noise_level)
            else:
                # If there was no reward, the confidence that this arm is not optimal increases.
                confidence = min(1, prob_optimal + noise_level)

            # If the confidence exceeds the threshold, discard arm.
            if confidence > confidence_threshold:
                clues["discard_arms"].append(arm)

        return clues

    def edit_epistemic_parameters(self):
        menu=input("Do you want to edit the epistemic parameters? (Y/N): ")
        if menu.upper() == "Y" or menu.upper() == "YES":
            import json
            import os
            file_path = os.path.join(os.path.dirname(__file__), "epistemic_parameters.json")
            new_confidence_threshold = float(input("Enter new confidence threshold (0 to 1): "))
            new_noise = float(input("Enter new noise level (0 to 1): "))
            new_parameters = [
                {
                    "confidence_threshold": new_confidence_threshold,
                    "noise": new_noise
                }
            ]
            with open(file_path, 'w') as file:
                json.dump(new_parameters, file, indent=4)
            print("Epistemic parameters updated successfully.")
        else:
            print("No changes made to epistemic parameters.")
        return

