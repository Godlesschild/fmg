from typing import Iterable, Iterator
from gpt4all import GPT4All


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
        with open("gen_prompt.txt") as file:
            return "".join(file.readlines())


PROMPT_GEN = PromptGen()
