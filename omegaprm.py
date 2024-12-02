import math
import random
from typing import List, Tuple, Optional
from state import SearchTree, CandidatePool, State
from utils import separate_steps
from llm_utils import LanguageModel


class OmegaPRM:
    def __init__(self, LM: LanguageModel, c_puct: float, alpha: float, beta: float, L: int, k: int, N: int,
                 rollout_budget: int):
        """
        Initialize the OmegaPRM algorithm.

        Parameters:
        - LM (LanguageModel): The language model instance.
        - expected_answer (str): The expected answer for correctness checking.
        - c_puct (float): Exploration constant.
        - alpha (float): Weight for MC(s).
        - beta (float): Length penalty.
        - L (int): Maximum solution length.
        - k (int): Number of rollouts for Monte Carlo estimation.
        - N (int): Maximum search count.
        """
        self.LM = LM  # Language Model
        self.expected_answer = None
        self.c_puct = c_puct
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.k = k
        self.N = N
        self.rollout_budget = rollout_budget

        self.T = SearchTree()
        self.C = CandidatePool()

        self.n = 0
        self.total_rollouts = 0

    def reset(self):
        """Reset internal state variables to prepare for a fresh run."""
        self.expected_answer = None
        self.T = SearchTree()  # Reset search tree
        self.C = CandidatePool()  # Reset candidate pool
        self.n = 0
        self.total_rollouts = 0
        self.collected_data = []  # Clear collected data

    def run(self, question: str, answer: str) -> List:
        """
        Execute the OmegaPRM algorithm.

        Parameters:
        - question (str): The question to generate solutions for.

        Returns:
        - Collected data: List of dictionaries.
        """
        self.reset()

        print(f"Running OmegaPRM for question: '{question}'\n")
        # Initialization
        initial_state = State(solution_prefix=question, step_text="", parent=None)
        self.expected_answer = answer
        self.T.root = initial_state
        self.T.add_state(initial_state)
        self.n = 0

        # Monte Carlo Estimation for initial_state
        self.monte_carlo_estimation(initial_state)
        incorrect_states = []
        reflected_responses = []

        # Main loop
        while self.n < self.N and self.total_rollouts < self.rollout_budget and not self.C.is_empty():
            # Selection Phase
            selected_state, selected_rollout = self.selection_phase()  # 整个树中优先级最高的节点和其rollout
            if selected_state is None or selected_rollout is None:
                print("No more candidates to explore. Terminating search.\n")
                break

            incorrect_states.extend(self.expansion_phase_binary_search(selected_state, selected_rollout))

            # Maintenance Phase
            self.maintenance_phase(selected_state)

            # Increment search count
            self.n += 1

        for incorrect_state in incorrect_states:
            if incorrect_state.parent is not None:
                parent_state = incorrect_state.parent
                if len(parent_state.correct_rollout_records) > 0:
                    solution_prefix = incorrect_state.solution_prefix
                    correct_idx = random.randint(0, len(parent_state.correct_rollout_records) - 1)
                    corrected_response = parent_state.correct_rollout_records[correct_idx].replace(parent_state.step_text, "")
                    reflected_rollout = solution_prefix + '\n\nWait, Alternatively\n\n' + corrected_response
                    print(f"Reflected rollout: {reflected_rollout}")
                    reflected_responses.append(reflected_rollout)
                else:
                    solution_prefix = incorrect_state.solution_prefix.replace(incorrect_state.step_text, "")
                    print('I need to regenerate a correct answer...')
                    corrected_response = self.LM.regenerate_rollout(solution_prefix, answer)
                    if corrected_response is not None:
                        reflected_rollout = solution_prefix + incorrect_state.step_text + '\n\nWait, Alternatively\n\n' + corrected_response
                        print(f"Reflected rollout: {reflected_rollout}")
                        reflected_responses.append(reflected_rollout)


        print(reflected_responses)
        return reflected_responses


    def monte_carlo_estimation(self, state: State):
        """
        Perform Monte Carlo estimation for state by generating k rollouts
        and computing MC(s) = c / k, where c is the number of correct rollouts.
        """
        c = 0  # Correct rollouts count
        incorrect_rollouts = []
        correct_rollouts = []
        # solution_prefix = state.step_text
        # if state.parent:
        #     while state.parent:
        #         solution_prefix = state.parent.step_text + '\n\n' + solution_prefix
        #         state = state.parent
        batch_rollouts = self.LM.generate_rollout(state.solution_prefix, self.k)
        for i, rollout in enumerate(batch_rollouts):
            # 针对当前状态生成一组16个完整回答, 判断正确性
            # Increment number of total rollouts
            self.total_rollouts += 1

            # Generate rollout r_i
            rollout = rollout.strip()
            state.add_rollout(rollout)

            # Evaluate correctness of final answer in rollout
            full_solution = (state.solution_prefix + '\n\n' + rollout).strip() if state.solution_prefix else rollout
            is_correct = self.LM.evaluate_correctness(full_solution, self.expected_answer)

            print(f"Rollout {i + 1} Correctness: {'Correct' if is_correct else 'Incorrect'}\n")

            if is_correct:
                c += 1
                correct_rollouts.append(rollout)
                state.add_correct_rollout(rollout)
            else:
                incorrect_rollouts.append(rollout)
                state.add_incorrect_rollout(rollout)  # Track incorrect rollouts

        # Update total rollouts and correct rollouts
        state.total_rollouts += self.k
        state.correct_rollouts += c
        state.MC = state.correct_rollouts / state.total_rollouts if state.total_rollouts > 0 else 0

        if state.MC == 1.0:
            # Add all correct rollouts to the tree as new states
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
        elif state.MC == 0.0:
            # State is incorrect; 说明该步骤错了, 应该进行反思, 回溯到父节点, 重新生成rollout
            return
        else:
            # 0 < MC(s) < 1.0
            # Add correct rollouts to the tree
            for rollout in correct_rollouts:
                self.add_correct_rollout_to_tree(state, rollout)
            # Add incorrect rollouts to candidate pool with updated priorities
            for rollout in incorrect_rollouts:
                priority = self.compute_selection_score(state, rollout)
                self.C.add_or_update(state, rollout, priority)

    def compute_Q(self, state: State, rollout: str) -> float:
        """
        Compute Q(s, r) = alpha^{1 - MC(s)} * beta^{len(r)/L}, where len(r) is based on word count.
        """
        # Count words in the rollout
        word_count = len(rollout.split())
        length_penalty = word_count / self.L
        Q_value = (self.alpha ** (1 - state.MC)) * (self.beta ** length_penalty)
        return Q_value

    def compute_U(self, state: State) -> float:
        """
        Compute U(s) = c_puct * sqrt(sum_{s'} N(s')) / (1 + N(s))
        """
        N_total = sum(s.N for s in self.T.nodes)
        if N_total == 0:
            N_total = 1  # Prevent division by zero
        U_s = self.c_puct * (math.sqrt(N_total)) / (1 + state.N)
        return U_s

    def compute_selection_score(self, state: State, rollout: str) -> float:
        """
        Compute selection score: Score(s, r) = Q(s, r) + U(s)
        """
        Q_s_r = self.compute_Q(state, rollout)
        U_s = self.compute_U(state)
        score = Q_s_r + U_s
        return score

    def selection_phase(self) -> Tuple[Optional[State], Optional[str]]:
        """
        Select (state, rollout) with the highest score from candidate pool C.
        """
        selected_state, selected_rollout = self.C.pop()
        return selected_state, selected_rollout

    def add_correct_rollout_to_tree(self, parent_state: State, rollout: str):
        """
        Add the correct rollout to the tree as a child of parent_state.
        """
        new_step_text = separate_steps(rollout.strip(), mode='split')[0].strip()
        new_solution_prefix = (
                    parent_state.solution_prefix + '\n\n' + new_step_text).strip() if parent_state.solution_prefix else rollout
        new_state = State(solution_prefix=new_solution_prefix, step_text=new_step_text, parent=parent_state)
        new_state.MC = 1.0  # Since the rollout is correct
        new_state.total_rollouts = 0
        new_state.correct_rollouts = 0
        self.T.add_state(new_state)
        parent_state.children.append(new_state)  # Add to parent's children

    def expansion_phase_binary_search(self, parent_state: State, rollout: str):
        """
        Expansion phase that adds the rollout as a new state and performs Monte Carlo estimation
        using Binary Search to efficiently find the correct rollout.

        Parameters:
        - parent_state (State): The state from which the rollout was selected.
        - rollout (str): The rollout string that was selected and is incorrect.
        """
        # Separate the rollout into individual steps
        steps = separate_steps(rollout.strip(), mode='split')
        steps = [step.strip() for step in steps]
        incorrect_states = []
        self.binary_search_incorrect_step(parent_state, steps, 0, len(steps) - 1, incorrect_states)
        return incorrect_states
        # Perform binary search to find incorrect steps

        # self.binary_search_incorrect_step(parent_state, steps, 0, len(steps) - 1)

    def binary_search_incorrect_step(self, s_ast: State, steps: List[str], left: int, right: int, incorrect_states: List[State]):
        """
        Recursively perform binary search to find all incorrect steps in the rollout.

        Parameters:
        - s_ast (State): The selected parent state.
        - steps (List[str]): The rollout steps as a list.
        - left (int): Left index of the current search interval.
        - right (int): Right index of the current search interval.
        """
        if left > right:
            return

        mid = (left + right) // 2
        new_steps = steps[left:mid + 1]
        if new_steps:
            prefix_solution = s_ast.solution_prefix + '\n\n' + separate_steps(new_steps, mode='join')
            new_step_text = new_steps[-1]
        else:
            prefix_solution = s_ast.solution_prefix
            new_step_text = s_ast.step_text
        # Create new state s_new
        s_new = State(solution_prefix=prefix_solution.strip(), step_text=new_step_text, parent=s_ast)
        self.T.add_state(s_new)
        s_ast.children.append(s_new)

        # Perform Monte Carlo estimation for s_new
        self.monte_carlo_estimation(s_new)

        if s_new.MC == 0:

            # Found incorrect step; continue searching in the left half to find earlier incorrect steps
            incorrect_states.append(s_new)
            self.binary_search_incorrect_step(s_ast, steps, left, mid - 1, incorrect_states)
        else:
            # Steps up to mid are correct; continue searching in the right half

            self.binary_search_incorrect_step(s_new, steps, mid + 1, right, incorrect_states)

    def maintenance_phase(self, state: State):
        """
        Update statistics and candidate pool for all incorrect rollouts associated with the state.

        Parameters:
        - state (State): The state whose incorrect rollouts need to be updated.
        """

        # Iterate through all incorrect rollouts of the state
        for rollout in state.incorrect_rollouts:
            # Since we've already determined these rollouts are incorrect, no need to re-evaluate correctness

            priority = self.compute_selection_score(state, rollout)
            # Update the candidate pool with the new priority
            self.C.add_or_update(state, rollout, priority)


if __name__ == "__main__":
    # Initialize the Language Model
    LM = LanguageModel(
        device="cuda",
        max_new_tokens=2048
    )

    # Define the question and expected answer
    question = "Melinda will roll two standard six-sided dice and make a two-digit number with the two numbers she rolls. For example, if she rolls a 6 and a 3, she can either form 36 or 63. What is the probability that she will be able to make an integer between 10 and 20, inclusive? Express your answer as a common fraction."
    expected_answer = "\\frac{11}{36}"

    # Initialize OmegaPRM with parameters
    omega_prm = OmegaPRM(
        LM=LM,
        c_puct=0.125,
        alpha=0.5,
        beta=0.9,
        L=500,
        k=6,
        N=10,
        rollout_budget=100
    )

    # Run the OmegaPRM algorithm
    collected_data = omega_prm.run(question, expected_answer)

