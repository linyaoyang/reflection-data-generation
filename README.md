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
```
---

## Usage

Run the main script to start the reasoning data generation process:

```
python run.py
```

---

## Customization:

- Modify the input question file or switch to a different LLM for reasoning.
- Configure hyperparameters such as:
  - max_new_tokens
  - top_k
  - top_p

These can be adjusted directly in the run.py file to tailor the tool to your requirements.

---

## Example Output

```
- Question: Melinda will roll two standard six-sided dice and make a two-digit number with the two numbers she rolls. For example, if she rolls a 6 and a 3, she can either form 36 or 63. What is the probability that she will be able to make an integer between 10 and 20, inclusive? Express your answer as a common fraction.

- Answer:

The total number of outcomes when rolling two dice is $6 \times 6 = 36$.

The outcomes that result in a two-digit number between 10 and 20 inclusive are:

10 (1,0), 11 (1,1), 12 (1,2), 13 (1,3), 14 (1,4), 15 (1,5), 16 (1,6), 20 (2,0),

which are 8 outcomes.

Wait, Alternatively

The possible two-digit numbers between 10 and 20 are: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, and 20.

We now count the number of outcomes that will give these numbers.

- For 10, the outcomes are (1, 0), but since we can't roll a 0, there are no such outcomes.
- For 11, the outcomes are (1, 1).
- For 12, the outcomes are (1, 2) and (2, 1).
- For 13, the outcomes are (1, 3), (3, 1).
- For 14, the outcomes are (1, 4), (4, 1).
- For 15, the outcomes are (1, 5), (5, 1).
- For 16, the outcomes are (1, 6), (6, 1).
- For 17, there are no such outcomes, as we can't roll 7 with two dice.
- For 18, there are no such outcomes, as we can't roll 8 with two dice.
- For 19, there are no such outcomes, as we can't roll 9 with two dice.
- For 20, there are no such outcomes, as we can't roll 20 with two dice.

So the total number of favorable outcomes is 1 + 2 + 2 + 2 + 2 + 2 = 11.

The probability that Melinda will be able to make an integer between 10 and 20 is $\frac{11}{36}$.

The answer is: $\frac{11}{36}$.
```
