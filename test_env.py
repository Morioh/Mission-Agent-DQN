import numpy as np
# Assuming your custom environment is saved in custom_env.py
from custom_env import GrantAllocationEnv

# Test script for the Grant Allocation Environment


def test_environment():
    # Initialize the environment
    env = GrantAllocationEnv()

    # Reset the environment
    state = env.reset()
    print("Initial State:")
    print(state)

    # Variables to track progress
    total_reward = 0
    steps = 0
    done = False

    # Simulate random actions in the environment
    while not done:
        # Random action from the action space
        action = env.action_space.sample()

        # Take the action
        next_state, reward, done, info = env.step(action)

        # Update metrics
        total_reward += reward
        steps += 1

        # Render the environment's current state
        print(f"\nStep {steps}:")
        print(f"Action Taken: {action}")
        print(f"Reward Received: {reward}")
        # Displaying the last student's state
        print(f"Next State: {next_state[env.current_student - 1]}")
        print(f"Remaining Budget: {env.remaining_budget}")
        print(f"Done: {done}")

    # Final results
    print("\nSimulation Complete")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {total_reward}")
    print(f"Remaining Budget: {env.remaining_budget}")


if __name__ == "__main__":
    test_environment()
