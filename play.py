import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from custom_env import GrantAllocationEnv

# Create a neural network to represent the trained policy


def create_policy_network(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        # Softmax for probabilities over actions
        Dense(output_dim, activation="softmax")
    ])
    return model

# Load trained PPO model


def load_trained_policy():
    print("Loading the trained policy...")
    model = PPO.load("grant_allocation_ppo")
    print("Model loaded successfully!")
    return model

# Simulate and visualize the policy in action


def simulate_policy():
    # Create the environment
    env = GrantAllocationEnv()
    state = env.reset()

    # Load the trained policy model
    model = load_trained_policy()

    # Create a figure for the simulation
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ion()  # Interactive mode for dynamic updates
    ax.set_title("Grant Allocation Simulation")
    ax.set_xlabel("Students")
    ax.set_ylabel("Grant Allocated ($)")
    students = list(range(env.num_students))
    allocations = np.zeros(env.num_students)

    # Simulate the policy
    done = False
    step_count = 0
    while not done:
        # Predict action using the trained policy
        action, _ = model.predict(state)

        # Apply the action in the environment
        state, reward, done, _ = env.step(action)

        # Record allocation for visualization
        allocations[env.current_student - 1] = action * \
            500  # Convert action to $ allocation

        # Update the plot
        ax.clear()
        ax.bar(students, allocations, color="blue", alpha=0.7)
        ax.set_title(f"Grant Allocation Simulation - Step {step_count + 1}")
        ax.set_xlabel("Students")
        ax.set_ylabel("Grant Allocated ($)")
        ax.set_xticks(students)
        plt.pause(0.5)

        step_count += 1

    # Finalize simulation
    plt.ioff()
    plt.show()
    print("\nSimulation complete!")
    print(f"Remaining Budget: {env.remaining_budget}")
    print(f"Total Reward: {env.total_reward}")


if __name__ == "__main__":
    simulate_policy()
