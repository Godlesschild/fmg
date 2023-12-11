from typing import Iterator

from gpt4all import GPT4All

import utils


class PromptGen:
    def __init__(self):
        self.model = None

    def generate(self) -> Iterator[str]:
        if self.model is None:
            self.model = GPT4All("nous-hermes-llama2-13b.Q4_0.gguf")

        for token in self.model.generate(self.get_prompt(), max_tokens=1000, streaming=True):
            yield token

    @staticmethod
    def get_prompt() -> str:
        config = utils.get_config()["prompt_gen_prompt"]

        prompt = (
            "### Instruction:"
            "\n"
            f"{config['instruction'].strip()}"
            "\n"
            "\n"
            "### Input:"
            f"{config['context'].strip()}"
            "\n"
            "\n"
            "### Response:"
            "\n"
        )

        return prompt


PROMPT_GEN = PromptGen()
