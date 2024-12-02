import logging
import os
import json
import argparse
from omegaprm import OmegaPRM
from llm_utils import LanguageModel
from typing import Dict, List
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


# Set up logging based on provided log file prefix
def setup_logging(log_file_prefix: str):
    log_filename = f"{log_file_prefix}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# Load questions from JSON
def load_questions(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)


# Filter a single question based on 32 rollouts, only those questions that have both correct and incorrect answers are processed
def should_process_question(question: Dict[str, str], llm: LanguageModel) -> bool:
    prompt = question["problem"]
    correct_answer = question["final_answer"]
    has_correct = False
    has_incorrect = False
    initial_batch_answers = llm.generate_rollout(prompt, 32)

    for answer in initial_batch_answers:
        if llm.evaluate_correctness(answer, correct_answer):
            has_correct = True
        else:
            has_incorrect = True
        if has_correct and has_incorrect:
            logger.info(f"Question passed filter: {question['problem']}")
            return True

    return False


# Run OmegaPRM on a question if it passes the filter
def process_question(omega_prm: OmegaPRM, question: Dict[str, str]):
    logger.info(f"Processing question with OmegaPRM: {question['problem']}")
    # q = "Melinda will roll two standard six-sided dice and make a two-digit number with the two numbers she rolls. For example, if she rolls a 6 and a 3, she can either form 36 or 63. What is the probability that she will be able to make an integer between 10 and 20, inclusive? Express your answer as a common fraction."
    # a = "\\frac{11}{36}"
    # reasoning_steps = omega_prm.run(q, a)
    reasoning_steps = omega_prm.run(question["problem"], question["final_answer"])
    collected_data = []
    if len(reasoning_steps) > 0:
        for rollout in reasoning_steps:
            rollout_steps = rollout.replace(question["problem"], "").strip()
            formatted_data = {"question": question["problem"], "final_answer": question["final_answer"], "reasoning_step": rollout_steps}
            collected_data.append(formatted_data)

    return collected_data


# Save collected data for each question
def save_question_data(collected_data: Dict, index: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    filename = os.path.join(output_dir, f"question_{index}.json")
    with open(filename, 'w') as f:
        json.dump(collected_data, f, indent=4)
    logger.info(f"Saved processed data to {filename}")

# Save all collected data into a unified file
def save_all_data_to_file(collected_data: List[Dict], output_file: str):
    # Check if the file exists and read its content
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    logger.warning(f"File {output_file} contains invalid data, overwriting.")
                    existing_data = []
            except json.JSONDecodeError:
                logger.warning(f"File {output_file} contains invalid JSON, overwriting.")
                existing_data = []
    else:
        existing_data = []

    # Append new data to existing data
    existing_data.extend(collected_data)

    # Write the updated data back to the file
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=4)
    logger.info(f"Appended {len(collected_data)} entries to {output_file}")

def main(args):
    global logger
    logger = setup_logging(args.log_file_prefix)

    logger.info("Starting OmegaPRM processing")
    logger.info(f"Using model: {args.model_name} on device: {args.device}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Question file: {args.question_file}")

    questions = load_questions(args.question_file)

    llm = LanguageModel(
        model_name=args.model_name,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    omega_prm = OmegaPRM(
        LM=llm,
        c_puct=args.c_puct,
        alpha=args.alpha,
        beta=args.beta,
        L=args.length_scale,
        k=args.num_rollouts,
        N=args.max_search_count,
        rollout_budget=args.rollout_budget
    )

    processed_count = 0  # Counter for processed questions

    for idx, question in enumerate(questions):
        if should_process_question(question, llm):
            collected_data = process_question(omega_prm, question)
            # save_question_data(collected_data, idx, args.output_dir)
            save_all_data_to_file(collected_data, args.output_file)
            processed_count += 1
        else:
            logger.info(f"Skipping question: {question['problem']}")

    # Log summary
    logger.info(f"Total questions processed by OmegaPRM: {processed_count}/{len(questions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OmegaPRM on filtered questions")

    parser.add_argument("--question_file", type=str, default="./extracted_problems_and_answers.json",
                        help="Path to the questions JSON file")
    parser.add_argument("--output_dir", type=str, default="output_results", help="Directory to save output JSON files")
    parser.add_argument("--output_file", type=str, default="output_results.json", help="Output JSON file name")
    parser.add_argument("--log_file_prefix", type=str, default="log/omega_prm_single_gpu",
                        help="Prefix for the log files")
    parser.add_argument("--model_name", type=str, default="/data/model/Qwen2.5-Math-7B-Instruct",
                        help="Model name or path for the language model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda', 'cpu')")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max tokens for LLM generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for LLM generation")
    parser.add_argument("--top_k", type=int, default=30, help="Top-K sampling for LLM generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P sampling for LLM generation")

    # OmegaPRM parameters with provided defaults
    parser.add_argument("--c_puct", type=float, default=0.125, help="Exploration constant for OmegaPRM")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for MC(s) in OmegaPRM")
    parser.add_argument("--beta", type=float, default=0.9, help="Length penalty for OmegaPRM")
    parser.add_argument("--length_scale", type=int, default=500, help="length scale in OmegaPRM")
    parser.add_argument("--num_rollouts", type=int, default=6,
                        help="Number of rollouts for Monte Carlo estimation in OmegaPRM")
    parser.add_argument("--max_search_count", type=int, default=10, help="Max search count in OmegaPRM")
    parser.add_argument("--rollout_budget", type=int, default=100, help="Rollout budget for OmegaPRM")

    args = parser.parse_args()
    main(args)
