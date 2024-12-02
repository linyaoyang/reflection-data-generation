# Reflection Data Generation

A tool to leverage Large Language Models (LLMs) for generating step-by-step reasoning data. Incorrect steps in the reasoning process are identified and corrected with a "Wait, alternatively" marker to enhance clarity and accuracy.

---

## Overview

This project is a **Supervised Fine-Tuning (SFT) data generation tool**, built on the [openr](https://github.com/openreasoner/openr) framework. Its goal is to produce high-quality SFT datasets designed to train O1-like models to improve their reflection and reasoning abilities. 

### Key Features:
- **MCTS-based Rollout Generation**: Uses the Monte Carlo Tree Search (MCTS) algorithm to generate multiple reasoning rollouts.
- **Step Evaluation and Correction**: Labels each step based on its correctness and identifies errors. Incorrect steps are corrected with a "Wait, alternatively" mark to reflect logical revisions.

---

## Installation

Ensure you have Python installed, then run the following command to install dependencies:

```bash
pip install torch==2.4.0 transformers>=4.44.2

---

## Usage

Run the main script to start the reasoning data generation process:
"""
python run.py
"""


Customization:
