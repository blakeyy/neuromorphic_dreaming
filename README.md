# Neuromorphic Dreaming as a Pathway to Efficient Learning in Artificial Agents

This repository contains the Python code for the paper "Neuromorphic Dreaming as a Pathway to Efficient Learning in Artificial Agents". It implements a model-based reinforcement learning (MBRL) approach using spiking neural networks (SNNs) running directly on the DYNAP-SE mixed-signal neuromorphic processor to learn the Atari Pong game.

## Overview

The project demonstrates how biologically inspired "dreaming" (offline learning using a world model) can improve sample efficiency in reinforcement learning when implemented on energy-efficient neuromorphic hardware. The core idea involves alternating between:

1.  **Awake Phase:** The SNN agent interacts with the real Pong environment (via OpenAI Gym), learning from actual rewards using a policy gradient rule. The world model SNN learns to predict environment transitions.
2.  **Dreaming Phase:** The agent interacts with the learned world model SNN, generating simulated experiences and further refining its policy based on predicted rewards.

Both the agent and world model SNNs leverage the analog neuron and synapse dynamics of the DYNAP-SE chip for computation, with only the final readout layers trained on a host computer.

## Hardware Requirement

Direct execution of the training code requires access to **DYNAP-SE neuromorphic hardware** and the associated `samna` software library. The core SNN dynamics are simulated on the chip in real-time.

## Dependencies

*   Python (tested with 3.x)
*   `samna`: Library for interfacing with DYNAP-SE. Follow SynSense [installation instructions](https://synsense-sys-int.gitlab.io/samna/0.45.3/index.html#).
*   `numpy`: For numerical operations.
*   `matplotlib`: For plotting results.
*   `gymnasium`: For the Atari Pong environment. (Requires Atari ROMs installation: `pip install gymnasium[atari] gymnasium[accept-rom-license]`)
*   `tqdm`: For progress bars.

It is recommended to use a virtual environment. You can typically install Python dependencies via pip:
`pip install numpy matplotlib gymnasium[atari] tqdm`
*(Ensure `samna` is installed separately according to hardware provider instructions)*

## Files

*   `agent.py`: Main `PongAgent` class implementing the SNN agent and world model logic, learning rules, and DYNAP-SE interaction.
*   `optimizer.py`: Adam optimizer implementation.
*   `params.py`: Parameter settings for networks (editable via `config.ini`).
*   `functions.py`: Utility functions for plotting results (rewards, policies, spikes).
*   `train.ipynb`: Jupyter Notebook providing a step-by-step guide to connect to the hardware, configure the agent, and run the training loop (both awake and dreaming phases).
*   `config.ini`: Configuration file for network parameters, learning rates, etc.
*   `results/`: Contains example reward data (`.npy` files) from training runs.
*   `compare_results.py`: Script to generate comparison plots (like Fig 1a in the main paper) from data in the `results` folder.
*   `README.md`: This file.

## Running the Training

1.  Ensure DYNAP-SE hardware is connected and `samna` library is functional.
2.  Install other Python dependencies.
3.  Open and run the cells in `train.ipynb`. This will guide you through:
    *   Connecting to the DYNAP-SE board.
    *   Creating and configuring the `PongAgent` (specify `if_dream=True` or `False`).
    *   Executing the training loop.
4.  Training progress (rewards, weights) will be saved to a new directory.
5.  Use `functions.py` or `compare_results.py` to visualize results.

## Configuration

Adjust parameters like neuron counts, learning rates, and training phases in `config.ini`. Note that major changes to network size may require more complex code modifications due to DYNAP-SE hardware constraints (e.g., fan-in limits, core sizes).

## Generating Comparison Plots

The `compare_results.py` script plots average returns from `.npy` files stored in the `results` folder.
1.  Ensure `.npy` files (e.g., `rewards_{repetition}if_dream_{True/False}.npy`) are in the `results` folder.
2.  Run `python compare_results.py`.
3.  A plot (`comparison_10.pdf`) will be saved in the `results` folder.

## License

This code is released under the MIT license. See the [LICENSE](LICENSE) file for details. Commercial use requires permission from the authors.

## Citation

If you use this code or find our work relevant, please cite the associated paper:
*[[arxiv](https://arxiv.org/abs/2405.15616)]*

## Acknowledgements

This work utilizes the DYNAP-SE neuromorphic chip and the `samna` library from SynSense. The Pong environment is provided by OpenAI Gymnasium. Parts of the code structure reuse elements from Capone et al.'s `biodreaming` repository ([https://github.com/cristianocapone/biodreaming](https://github.com/cristianocapone/biodreaming)). We acknowledge financial support detailed in the main manuscript.
