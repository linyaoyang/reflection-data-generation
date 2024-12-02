import re
import json
from typing import Union, List, Any


# Helper function to separate reasoning steps
def separate_steps(steps: Union[List[str], str], mode: str = 'join') -> Any:
    delimiter = "\n\n"
    if mode == 'join':
        if not isinstance(steps, list):
            raise TypeError("For 'join' mode, 'steps' must be a list of strings.")
        return delimiter.join(steps)
    elif mode == 'split':
        if not isinstance(steps, str):
            raise TypeError("For 'split' mode, 'steps' must be a string.")
        return steps.strip().split(delimiter)
    else:
        raise ValueError("Mode should be either 'join' or 'split'.")


# Helper function to check correctness of a generated response
def check_correctness(generated_response: str, expected_answer: str) -> bool:
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', generated_response.strip())
    last_sentence = sentences[-1] if sentences else ''
    return expected_answer.strip() in last_sentence.strip()


if __name__ == '__main__':
    # 读取 JSON 文件
    with open('output_results.json', 'r') as f:
        data = json.load(f)

    # 确保数据是一个字典列表，并统计字典数量
    if isinstance(data, list):
        dict_count = sum(1 for item in data if isinstance(item, dict))
        print(f"字典的数量是: {dict_count}")
        print(len(data))
    else:
        print("数据不是一个字典列表")
