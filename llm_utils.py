import os
import threading
from transformers import pipeline
from typing import List
from utils import check_correctness


class LLMService:
    """
    A class to generate an LLM service based on a local/HuggingFace LLM model.
    """

    def __init__(self, model_name: str = "/data/model/Qwen2.5-Math-1.5B-Instruct", device: str = "cuda",
                 max_new_tokens: int = 2048, temperature: float = 0.7, top_k: int = 30, top_p: float = 0.9):
        """
        Initialize the LLM service with specified parameters.
        Parameters:
            model_name (str): The path to the local LLM model or the model name on HuggingFace.
            device (str): The device to run the model on, e.g. "cuda" or "cpu".
            max_new_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature parameter for sampling.
            top_k (int): The top-k parameter for sampling.
            top_p (float): The top-p parameter for sampling.
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.pipeline = None
        self.load_lock = threading.Lock()

    def start_service(self):
        """
        Start the LLM service by loading the model into the pipeline if it's not already loaded.
        Ensures thread-safe loading using a lock.
        """
        with self.load_lock:
            if self.pipeline is None:
                print(f"Loading model '{self.model_name}' on device '{self.device}'...")
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                print("Model loaded successfully.")

    def generate_response(self, prompt: str, num_copies: int = 2) -> List[str]:
        """
        Generate responses from the model based on the provided prompt, duplicated to form a batch.

        Parameters:
        - prompt (str): The input prompt to generate responses for.
        - num_copies (int): The number of copies of the prompt to create for batch processing (default is 16).

        Returns:
        - List[str]: A list of generated responses, each corresponding to a duplicate of the input prompt.
        """
        if self.pipeline is None:
            raise ValueError("LLM service not started. Please call start_service() first.")

        # Create a batch of the same prompt
        prompts = [prompt] * num_copies
        responses = self.pipeline(
            prompts,
            max_new_tokens=self.max_new_tokens,
            batch_size=num_copies,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            return_full_text=False
        )
        response_message_batch = [result[0]["generated_text"] for result in responses]
        return response_message_batch

    def generate_single_response(self, prompt: str) -> str:
        if self.pipeline is None:
            raise ValueError("LLM service not started. Please call start_service() first.")

        # Generate responses from the model
        response = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            return_full_text=False
        )
        response_message = response[0]["generated_text"]
        return response_message


class LanguageModel:
    def __init__(self, model_name="/data/model/Qwen2.5-Math-7B-Instruct",
                 device="cuda:3", max_new_tokens=512, temperature=0.7, top_k=30, top_p=0.9):
        """
        Initialize the LanguageModel with parameters for the LLM service.

        Parameters:
        - model_name (str): Path or model name for the LLM.
        - device (str): Device for computation (e.g., 'cuda', 'cpu').
        - max_new_tokens (int): Max tokens for response generation.
        - temperature (float): Sampling temperature for diversity.
        - top_k (int): Top-K sampling for diversity.
        - top_p (float): Top-P sampling for response diversity.
        """
        self.llm_service = LLMService(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        # self.default_prompt = (
        #     "Please complete the answer for the question based on the given steps without generating existing steps again, "
        #     "and separate your following steps using \\n\\n.\n\n"
        # )
        self.default_prompt = (
            """Please complete the answer for the question based on the given steps without generating existing steps again. Ensure that each logical step is written as a single paragraph, without splitting it across multiple lines. Separate your steps using \\n\\n. Avoid breaking a single step into multiple sections, and ensure each step is clear and concise in its own paragraph.\n\n""")
        self.llm_service.start_service()

    def generate_rollout(self, state_prefix: str, num_copies) -> List[str]:
        """
        Combine the default prompt with the state prefix and generate a response.

        Parameters:
        - state_prefix (str): The current solution prefix.

        Returns:
        - str: Generated response from LLM.
        """
        prompt = self.default_prompt + state_prefix
        batch_response = self.llm_service.generate_response(prompt, num_copies)
        return batch_response

    def regenerate_rollout(self, state_prefix: str, expected_answer: str):
        """combine the state prefix with the expected answer to generate a correct response"""
        is_correct = False
        count = 0
        response = None
        while not is_correct:
            prompt = self.default_prompt + state_prefix + "\nNote that the correct answer for this question is " + expected_answer
            response = self.llm_service.generate_single_response(prompt)
            is_correct = self.evaluate_correctness(response, expected_answer)
            count += 1
            if count >= 5:
                return None
        return response

    def update_prompt(self, new_prompt: str):
        """
        Update the default prompt if necessary.

        Parameters:
        - new_prompt (str): The new prompt template.
        """
        self.default_prompt = new_prompt

    def evaluate_correctness(self, response: str, expected_answer: str) -> bool:
        """
        Check if the generated solution matches the expected answer.

        Parameters:
        - solution (str): The complete generated response.
        - expected_answer (str): The expected answer to compare with.

        Returns:
        - bool: True if the expected answer is in the final part of the solution.
        """
        return check_correctness(response, expected_answer)


if __name__ == "__main__":
    llm_service = LLMService()
    llm_service.start_service()

    prompt = "What is game theory?"
    responses = llm_service.generate_single_response(prompt)

    print(responses)
