import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from custom_env import GrantAllocationEnv  # Import your custom environment

# Build the neural network model


def build_model(input_shape, action_space):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # Linear activation for Q-values
    model.add(Dense(action_space, activation='linear'))
    return model

# Build the DQN agent


def build_agent(model, action_space):
    policy = EpsGreedyQPolicy(eps=0.1)  # Epsilon-greedy policy
    # Memory for experience replay
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        nb_actions=action_space,
        memory=memory,
        nb_steps_warmup=10,
        target_model_update=1e-2,
        policy=policy,
    )
    dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
    return dqn


def train_agent(env, dqn_agent):
    # Train the agent
    dqn_agent.fit(env, nb_steps=5000, visualize=False, verbose=2)
    # Save the trained weights
    dqn_agent.save_weights("dqn_weights.h5f", overwrite=True)
    print("Training complete. Weights saved as 'dqn_weights.h5f'.")


def evaluate_agent(env, dqn_agent):
    # Load the trained weights
    dqn_agent.load_weights("dqn_weights.h5f")
    # Evaluate the agent
    dqn_agent.test(env, nb_episodes=10, visualize=False)


if __name__ == "__main__":
    # Initialize the environment
    env = GrantAllocationEnv()
    states = env.observation_space.shape
    actions = env.action_space.n

    # Build model and agent
    model = build_model(input_shape=(1,) + states, action_space=actions)
    dqn_agent = build_agent(model, actions)

    # Train the agent
    train_agent(env, dqn_agent)

    # Evaluate the agent
    evaluate_agent(env, dqn_agent)
