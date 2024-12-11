
# Hybrid TD3-World Model for Robotic Learning in the Robosuite "Lift" Environment

This project implements a hybrid reinforcement learning framework that integrates a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent with a world model. The system is designed to enhance sample efficiency and enable action planning for robotic tasks, specifically in the Robosuite "Lift" environment.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [TD3 Model Training](#td3-model-training)
  - [Data Collection](#data-collection)
  - [World Model Training](#world-model-training)
  - [Hybrid Agent Training](#hybrid-agent-training)
  - [Testing and Visualization](#testing-and-visualization)
- [Results](#results)
- [Future Work](#future-work)

## Overview

This project combines model-free (TD3) and model-based (world model) reinforcement learning techniques:
- **TD3**: Trains a policy to complete tasks using real environment interactions.
- **World Model**: Simulates environment dynamics to generate synthetic data and aid in planning.
- **Hybrid Approach**: Uses real and imagined experiences for efficient learning and integrates planning for action selection.

The framework is tested in the "Lift" task, where a robotic arm manipulates an object to a target position.

## Project Structure

- **`td3_torch.py`**: Implementation of the TD3 agent.
- **`world_model.py`**: Neural network architecture for the world model.
- **`data_collection.py`**: Script to collect state-action-reward data and synchronized video frames.
- **`world_model_training.py`**: Script to train the world model.
- **`hybrid_td3.py`**: Implementation of the hybrid agent.
- **`trajectory_generator.py`**: Generates trajectories using the world model.
- **`model_based_planning.py`**: Implements action planning with the world model.
- **`test_world_model.py`**: Tests and visualizes predictions from the world model.

## Prerequisites

- Python 3.7+
- PyTorch
- Robosuite
- NumPy
- Matplotlib
- torchvision
- imageio

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download Robosuite and verify its installation:
   ```bash
   pip install robosuite
   ```

## Usage

### TD3 Model Training
1. Edit `main.py` to configure training parameters.
2. Run the script:
   ```bash
   python main.py
   ```

### Data Collection
1. Use the trained TD3 model to collect data:
   ```bash
   python data_collection.py
   ```

2. Data will be saved in the `states` and `deletevid` directories.

### World Model Training
1. Train the world model using the collected data:
   ```bash
   python world_model_training.py
   ```

2. Checkpoints will be saved in the current directory.

### Hybrid Agent Training
1. Train the hybrid TD3-world model agent:
   ```bash
   python hybrid_td3.py
   ```

2. Models and reward history will be saved in the `tmp/hybrid_td3` directory.

### Testing and Visualization
1. Test the world model's predictions:
   ```bash
   python test_world_model.py
   ```

2. Visualize planned and real trajectories using the planner:
   ```bash
   python model_based_planning.py
   ```

## Results

The hybrid system demonstrated improved sample efficiency and planning capabilities, with the world model accurately predicting future states and rewards. Refer to the visualizations in the results directory for qualitative comparisons.

## Future Work

- Scaling to complex tasks and multi-robot scenarios.
- Integrating alternative imagination strategies.
- Exploring additional planning techniques for hybrid systems.

## License

This project is released under the MIT License.
