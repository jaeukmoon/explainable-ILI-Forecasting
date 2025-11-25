import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from transformers.models.llama.modeling_llama import LlamaModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class llama31(nn.Module):
    def __init__(self,
                 ticker: str,
                 summary: str,
                 target: str,
                 predict_llm = OpenAILLM()
                 ) -> None:

        self.ticker = ticker
        self.summary = summary
        self.target = target
        self.prediction = ''

        self.predict_prompt = ILI_PREDICT_INSTRUCTION
        self.predict_examples = PREDICT_EXAMPLES
        self.llm = predict_llm

        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        facts = "Facts:\n" + self.summary + "\n\nFuture Influenza outbreaks: "
        self.scratchpad += facts
        print(facts, end="")

        self.scratchpad += self.prompt_agent()  # llm의 예측 결과를 추가
        response = self.scratchpad.split('Future Influenza outbreaks: ')[-1]
        self.prediction = response.split()[0]
        print(response, end="\n\n\n\n")

        self.finished = True

    def prompt_agent(self) -> str:
        return self.llm(self._build_agent_prompt())

    def _build_agent_prompt(self) -> str: # 프롬프트 포맷에 목표 주식 종목, 예시, 
        return self.predict_prompt.format(
                            ticker = self.ticker,
                            examples = self.predict_examples,
                            summary = self.summary)

    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.target, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''