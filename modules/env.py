import random as rd
import numpy as np
import os
import json

class SimEnv:
    def __init__(self, reward_probs, noise=0.0, epistemic_info=False):
        #This code focuses on three arm scenarios, but it can be modified if needed.
        if len(reward_probs) == 3 and all(0 <= p <= 1 for p in reward_probs):
            self.n_arms = 3
            self.reward_probs = reward_probs
            self.noise = noise
            self.epistemic_info = epistemic_info
            self.round = 0
            self.history = []
            self.optimal_arm = self.reward_probs.index(max(self.reward_probs))
        else:
            raise ValueError("The environment must have exactly 3 arms with probabilities between 0 and 1.")

    #Resets values for new instance
    def reset(self):
        self.round = 0
        self.history = []
        return self.get_true_probs()

    #Gets true probability per arm
    def get_true_probs(self):
        return list(self.reward_probs)

    #Rewards arms and updates history
    def step(self, action):
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Action must be between 0 and {self.n_arms - 1}.")
        
        self.round += 1
        true_prob = self.reward_probs[action]
        effective_prob = min(max(true_prob + rd.uniform(-self.noise, self.noise), 0), 1)
        reward = 1 if rd.random() < effective_prob else 0

        self.history.append({
            "round": self.round,
            "action": action,
            "reward": reward
        })

        info = {}
        if self.epistemic_info:
            info["optimal_arm"] = self.optimal_arm
            info["epistemic_clue"] = self.generate_epistemic_clues(action, reward)

        return reward, info

    #Generates epistemic clues to update agent's beliefs
    def generate_epistemic_clues(self, action, reward):
        clues = {"discard_arms": []}
        confidence_threshold = 0.7
        noise_level = 0.2

        for arm in range(self.n_arms):
            if arm == action:
                continue

            prob_optimal = self.reward_probs[arm]

            if reward == 1:
                confidence = max(0, 1 - prob_optimal - noise_level)
            else:
                confidence = min(1, prob_optimal + noise_level)

            if confidence > confidence_threshold:
                clues["discard_arms"].append(arm)

        return clues

    def edit_epistemic_parameters(self):
        menu = input("Do you want to edit the epistemic signal generation parameters? (Y/N): ")
        if menu.strip().upper() in {"Y", "YES", "YE"}:
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
