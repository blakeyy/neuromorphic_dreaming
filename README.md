# Neuromorphic dreaming: A pathway to efficient learning in artifical agents

This repository contains a Python implementation of a model-based reinforcement learning (MBRL) approach that uses the DYNAP-SE neuromorphic chip. The agent and model are implemented with spiking neural networks (SNN) to learn to play the Atari game Pong.

## Files

- `agent.py`: Contains the main implementation of the `PongAgent` class, which represents the SNN agent. It includes methods for initializing the policy and model readouts, creating the SNN architecture, processing input states, learning from rewards, and interacting with the DYNAP-SE chip.

- `optimizer.py`: Implements the Adam optimizer used for updating the weights of the agent's readout layers.

- `params.py`: Defines parameter settings for the agent and model networks.

- `functions.py`: Provides utility functions for plotting rewards, policies, spikes, and other relevant data during training.

- `train.ipynb`: Jupyter Notebook that demonstrates how to train the SNN agent to play Pong. It includes steps for connecting to the DYNAP-SE hardware, configuring the agent, and running the training loop.

## Dependencies

Make sure you have the necessary dependencies installed, including `samna`, `numpy`, `matplotlib`, `gymnasium`, and `tqdm`.

## Run the code
You can run the cells in the `train.ipynb` notebook one-by-one.

1. Import the packages.

2. Connect to the DYNAP-SE hardware.

3. Create an instance of the `PongAgent` class and specify whether to use dreaming.

4. Run the training loop to train the agent to play Pong. The notebook includes code for the awake phase (where the agent interacts with the environment) and the dreaming phase (where the agent learns from its own predictions).

5. During training, the agent's performance and other relevant data will be saved in a new training directory. You can visualize the agent's learning progress using the plotting functions provided in `functions.py`.

## Configuration

The `config.ini` file contains various configuration settings for the agent, such as the number of actions, hidden neurons, spike generators per input value, and learning rates. You can modify these settings to experiment with different configurations. However, changing some parameters might require more sophisticated code changes due to hardware constraints.

## Producing Plots from Example Results

The repository includes a `compare_results.py` script that allows you to produce plots from the example results provided in the `results` folder. Here's how you can use it:

1. Make sure you have the necessary dependencies installed, including `numpy`, `matplotlib`, and `scipy`.

2. Place the example results in the `results` folder. The results should be in the form of `.npy` files with names following the pattern `rewards_{repetition}if_dream_{if_dream}.npy`.

3. Run the `compare_results.py` script. The script will load the results files, compute the mean and standard deviation of the rewards, and plot the results.

4. The script will generate a plot named `comparison_10.pdf` saved in the `results` folder showing the average return for each experiment. It will include the mean reward (dashed line), the 80th percentile (solid line), and the standard deviation (shaded area) for each experiment.

Note: Make sure the `results` folder contains the necessary `.npy` files with the correct naming format before running the script.

## Notes

- The code assumes the availability of a DYNAP-SE chip.

- The code includes functionality for dreaming, where the agent learns from predictions of a world model network. You can enable or disable dreaming by setting the `if_dream` variable accordingly.

## License

This code is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. This means that you are free to share and adapt the code for non-commercial purposes, as long as you give appropriate credit to the original authors and distribute your contributions under the same license. If you want to use the code for commercial applications, you need to obtain permission from the authors.

## Acknowledgements

This work reuses parts from [this code](https://github.com/cristianocapone/biodreaming) of the paper "Towards biologically plausible Dreaming and Planning in recurrent spiking networks" by Capone et al. The implementation leverages the DYNAP-SE neuromorphic chip and the [`samna`](https://synsense-sys-int.gitlab.io/samna/) library for interfacing with the hardware. In addition we use the [`gymnasium`](https://gymnasium.farama.org/index.html) library to run the reinforcement learning task (Atari Pong).
