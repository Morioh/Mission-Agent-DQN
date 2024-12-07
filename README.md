# Mission-Agent-DQN

Mission-Agent-DQN is a reinforcement learning project that optimizes grant distribution in educational settings using a Deep Q-Network (DQN) agent. The custom environment simulates scenarios where an agent allocates limited funds to maximize educational outcomes while balancing fairness, financial constraints, and long-term impacts.
## Environment Details

### 1. Environment Grid
- **Size**: A dynamic “grid” where each row represents a student profile, and columns represent key factors such as academic performance, financial need, household circumstances, and career aspirations.  
- **Example**: A grid with 100 rows (students) and 4 columns (factors).

### 2. Agent (Allocator)
- **Role**: The agent represents a decision-making entity tasked with allocating grants.  
- **Initial State**: Starts with a total budget (e.g., $10,000) and no grants distributed.  
- **Actions**: Distribute grants to students or allocate nothing.

### 3. Goals (Student Outcomes)
- **Key Objectives**: Graduation rates, reduced dropout rates, increased career success, and higher student satisfaction.  
- **Metrics**: Represented as metrics updated after each action. The agent’s goal is to maximize these cumulative metrics.

### 4. Obstacles
- **Constraints that limit the agent’s decisions**:
  - **Budget Limit**: Ensures the agent does not allocate more than the available budget.
  - **Fairness Criteria**: Promotes equitable distribution among students.
  - **Application Timelines**: Forces the agent to consider students within specific windows for funding.

### 5. State Representation
- **A vector for each student containing**:
  - Academic performance (normalized score).
  - Financial needs (scaled amount).
  - Household circumstances (numerical features).
  - Career aspirations (one-hot encoded or numerical representation).

### 6. Reward System
- **Positive Rewards**: Improving key metrics (e.g., +10 for reduced dropouts).  
- **Negative Rewards**: Inefficient allocations or unmet constraints (e.g., -5 for exceeding budget).

### 7. Termination Criteria
- The episode ends when:
  - The budget is exhausted.
  - All students have been evaluated.
  - A predefined number of steps is reached i.e 100

### Example Grid (Environment State)

The table below represents the state of the environment, where each row corresponds to a student profile, and columns capture various factors and grant allocations.

| **Student ID** | **Academic Performance** | **Financial Need** | **Household Dependants** | **Career Aspiration Score** | **Grant Received** |
|-----------------|--------------------------|---------------------|--------------------------|-----------------------------|--------------------|
| 1               | 0.85                    | 500                 | 3                        | 0.9                         | 500                |
| 2               | 0.70                    | 700                 | 2                        | 0.8                         | 700                |
| 3               | 0.60                    | 1200                | 4                        | 0.7                         | 0                  |
| 4               | 0.95                    | 300                 | 1                        | 0.95                        | 300                |
| 5               | 0.40                    | 1500                | 5                        | 0.6                         | 0                  |

## Actions

### Agent Actions in Grant Allocation

The agent’s actions represent these decisions on how to allocate funds 

#### 1. Allocate Funds
- **Choose the grant amount** to allocate to a particular applicant (e.g., $500, $700, $900, or “no allocation”).
- **Select a grant type** (e.g., emergency relief, tuition support, or project-based funding).

#### 2. Prioritize Applicants
- **Rank applicants** based on a chosen metric (e.g., financial need, academic performance, or household circumstances).

#### 3. Defer Allocation
- **Delay funding decisions** for certain applicants due to insufficient information.

#### 4. Terminate
- **End the funding cycle** when resources are exhausted or the application period ends.

### Rewards System

Rewards measure the success of the agent’s decisions. The reward system is designed to reflect the objective of maximizing educational outcomes:

### State-Based Rewards
Rewards based on the outcomes or changes in state after actions are performed:

### Positive Rewards
- **+10** for each applicant who receives a grant and achieves a measurable benefit (e.g., remains enrolled in school or improves academic performance).
- **+20** for every applicant who graduates due to grant support.
- **+5** for allocating funds to applicants in high-need categories (e.g., very low household income or high dependency ratio).

### Negative Rewards
- **-10** for funding allocations to students who drop out or fail.
- **-5** for exceeding the budget or violating fairness constraints (e.g., disproportionate allocations to one group).

### Action-Based Rewards
Rewards for taking specific actions:

### Positive Rewards
- **+2** for allocating funds efficiently (e.g., allocating exactly what is needed, not more or less).
- **+3** for choosing the grant type that matches the student’s stated needs (e.g., emergency relief for urgent cases).

### Negative Rewards
- **-2** for deferring too many applicants, causing a backlog.
- **-3** for allocating funds inefficiently (e.g., overfunding one applicant while ignoring others).

## Long-Term Rewards
Incorporate delayed rewards for outcomes that manifest later:
- **+50** for increasing overall graduation rates.
- **+30** for significant reductions in dropout rates.
- **+20** for achieving a balanced allocation across diverse student profiles (fairness metric).

---

## Features
- Custom OpenAI Gym environment for grant distribution (`GrantAllocationEnv`).
- Deep Q-Network (DQN) agent implemented with Keras-RL.
- Neural network for decision-making.
- Epsilon-Greedy Policy for exploration and exploitation.
- Saves trained weights in an `.h5` file for reuse.

---

## Requirements
Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.x
- Keras-RL2
- OpenAI Gym

Install dependencies with:
```bash
pip install tensorflow keras-rl2 gym numpy
```
## Project Structure

```bash
Mission-Agent-DQN/
├── custom_env.py          # Custom OpenAI Gym environment
├── train_agent.py         # Script to train and evaluate the DQN agent
├── dqn_weights.h5        # Saved weights (generated after training)
├── README.md              # Project documentation
```

## Setup and Run Instructions

1. Clone the Repository
```bash
https://github.com/Morioh/Mission-Agent-DQN.git
```
2. Run the Environment Test
Ensure the custom environment works as expected:
```bash
python GrantDistributionOptimization_env.py
```
3. Train the DNQ Agent
Train the agent to optimize grant distribution:
```bash
python train.py
```
4. Evaluate the Agent
```bash
python train.py --evaluate
```
---

## [Video Presentation](https://youtu.be/f7_HZEDreXM)

---
## Customization

- Modify GrantDistributionOptimization_env.py to adjust environment parameters like student profiles or budget limits.
- Update train.py to experiment with different hyperparameters (e.g., learning rate, memory size, or policy).
