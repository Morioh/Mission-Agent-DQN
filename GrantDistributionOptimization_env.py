import gym
from gym import spaces
import numpy as np


class GrantAllocationEnv(gym.Env):
    def __init__(self):
        super(GrantAllocationEnv, self).__init__()

        # Define the state space
        # Each student profile is represented as [academic_score, financial_need, household_circumstances, grant_type_needed]
        self.num_students = 10
        # Initialize student profiles
        self.state = np.zeros((self.num_students, 4))
        self.current_student = 0

        # Action space: Allocate funding or defer decision
        # Actions = [0 (No allocation), 1 (Allocate $500), 2 (Allocate $700), 3 (Allocate $900)]
        self.action_space = spaces.Discrete(4)

        # Observation space: Each student has 4 attributes
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_students, 4), dtype=np.float32
        )

        # Initialize available budget
        self.total_budget = 10000
        self.remaining_budget = self.total_budget

        # Reward system
        self.total_reward = 0
        self.done = False

    def reset(self):
        # Initialize student profiles
        self.state = np.random.rand(self.num_students, 4)
        self.current_student = 0
        self.remaining_budget = self.total_budget
        self.total_reward = 0
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            raise RuntimeError(
                "Environment has already finished. Call reset to restart.")

        # Get the current student's profile
        student_profile = self.state[self.current_student]
        academic_score, financial_need, household_circumstances, grant_type_needed = student_profile

        # Rewards initialization
        reward = 0

        # Define funding amounts for actions
        funding_amounts = [0, 500, 1000, 2000]

        # Apply action
        allocated_amount = funding_amounts[action]

        # Check if budget is exceeded
        if self.remaining_budget < allocated_amount:
            reward -= 5  # Penalize for exceeding budget
        else:
            self.remaining_budget -= allocated_amount

            # Calculate rewards
            if allocated_amount > 0:
                # Positive reward for helping a student
                reward += 10 * academic_score + 10 * financial_need

                # Additional reward if funding matches needs
                if grant_type_needed >= 0.5 and allocated_amount >= 1000:
                    reward += 5
            else:
                # Penalize for ignoring a high-need student
                if financial_need > 0.7:
                    reward -= 5

        # Move to next student
        self.current_student += 1

        # Check if the episode is done
        if self.current_student >= self.num_students or self.remaining_budget <= 0:
            self.done = True

        # Update total reward
        self.total_reward += reward

        return self.state, reward, self.done, {}

    def render(self, mode="human"):
        print(
            f"Student {self.current_student}, Remaining Budget: {self.remaining_budget}, Total Reward: {self.total_reward}")

    def close(self):
        pass
