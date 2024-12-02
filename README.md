# reflection-data-generation
Leverage an LLM to generate step-by-step reasoning data, in which incorrect steps are identified and corrected with "Wait, Alternatively" seperated. 


This project is a SFT data generation tool developed based on the openr (https://github.com/openreasoner/openr) project, aiming at generate high-quality SFT data for training o1-like models's reflection ability. This tool utilizes MCTS algorithm to generate multiple rollouts, and labels each step based on the correctness of its rollout. Then, it identifies those incorrect steps and correct them with a "Wait, Althernatively" mark.

# Installation
pip install torch==2.4.0 transformers>=4.44.2

# Usage
python run.py --question_file ./extracted_problems_and_answers.json --output_file output_results.json --model_name /data/model/Qwen2.5-Math-7B-Instruct
