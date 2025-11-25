from typing import List, Union, Literal
from utils.llm import * 
from utils.prompts import * #REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER
from utils.fewshots import PREDICT_EXAMPLES


class PredictAgent:
    def __init__(self,args,
                 data_X: str,
                 summary: str,
                 data_Y: str,
                 predict_llm = str
                 ) -> None:

        self.data_X = data_X
        self.summary = summary
        self.data_Y = data_Y
        self.prediction = ''
        self.time_series_prompt = TIME_SERIES_INSTRUCTION

        self.predict_prompt = ILI_PREDICT_INSTRUCTION
        self.explanation_prompt = ILI_EXPLANATION_GENERATION
        self.predict_examples = PREDICT_EXAMPLES
        # self.llm = predict_llm
        self.predict_llm = predict_llm
        if predict_llm =="llama":
            self.llm = LlamaLLM()
        else:
            self.llm = OpenAILLM()
        self.args = args

        self.__reset_agent()

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()

        facts = "Given a list of past time-series ILI occurrences of " + str(self.args.seq_len) + " weeks are: " +self.data_X+". \nGive your response in this format:\n(1) Estimated Influenza outbreaks, which should be numerical values.\n(2) Explanation, which should be in a single, short paragraph.\nFacts:\n" + self.summary + "\n\nFuture Influenza outbreaks after "+ str(self.args.pred_len)+" weeks:\n"+self.data_Y
        self.scratchpad += facts

        self.scratchpad += self.data_Y  # 실제 발생량 결과를 추가
        
        # print(facts, end="\n\n\n\n")

        self.finished = True
        return facts
    def _build_prediction_prompt(self) -> str:
        facts = self.predict_prompt.format(
                            seq_len = str(self.args.seq_len),
                            data_X = self.data_X,
                            summary = self.summary,
                            pred_len = str(self.args.pred_len))
        self.scratchpad += facts

        self.scratchpad += self.data_Y 
        
        return self.scratchpad

    # def prompt_agent(self) -> str:
    #     return self.llm(self._build_agent_prompt())

    def _build_explanation(self,) -> str: # 프롬프트 포맷에 목표 주식 종목, 예시, 
        if self.predict_llm == "llama":
            output = self.llm(self.explanation_prompt.format(
                                data_X = self.data_X,
                                data_Y = self.data_Y,
                                summary = self.summary,
                                pred_len = str(self.args.pred_len), seq_len = str(self.args.seq_len)),max_length=2000)
        else:
            output = self.llm(self.explanation_prompt.format(
                            data_X = self.data_X,
                            data_Y = self.data_Y,
                            summary = self.summary,
                            pred_len = str(self.args.pred_len), seq_len = str(self.args.seq_len)))
        
        return output

    def is_finished(self) -> bool:
        return self.finished

    # def is_correct(self) -> bool:  # 이걸 loss로 바꿔야되나..
    #     return EM(self.target, self.prediction)

    def __reset_agent(self) -> None:
        self.finished = False
        self.scratchpad: str = ''


# class PredictReflectAgent(PredictAgent):
#     def __init__(self,
#                  ticker: str,
#                  summary: str,
#                  target: str,
#                  explanation_llm = OpenAILLM(),
#                  reflect_llm = OpenAILLM()
#                  ) -> None:

#         super().__init__(ticker, summary, target, explanation_llm)
#         self.explanation_llm = explanation_llm
#         self.reflect_llm = reflect_llm
#         self.reflect_prompt = REFLECT_INSTRUCTION
#         self.agent_prompt = ILI_PREDICT_INSTRUCTION
#         self.reflections = []
#         self.reflections_str: str = ''

#     def run(self, reset=True) -> None:
#         if self.is_finished() and not self.is_correct():
#             self.reflect()

#         PredictAgent.run(self, reset=reset)

#     def reflect(self) -> None:
#         print('Reflecting...\n')
#         reflection = self.prompt_reflection()
#         self.reflections += [reflection]
#         self.reflections_str = format_reflections(self.reflections)
#         print(self.reflections_str, end="\n\n\n\n")

#     def prompt_reflection(self) -> str:
#         return self.reflect_llm(self._build_reflection_prompt())

#     def _build_reflection_prompt(self) -> str:
#         return self.reflect_prompt.format(
#                             ticker = self.ticker,
#                             scratchpad = self.scratchpad)

#     def _build_agent_prompt(self) -> str:
#         prompt = self.agent_prompt.format(
#                             ticker = self.ticker,
#                             examples = self.predict_examples,
#                             reflections = self.reflections_str,
#                             summary = self.summary)
#         return prompt

#     def run_n_shots(self, model, tokenizer, reward_model, num_shots=4, reset=True) -> None:
#         self.llm = NShotLLM(model, tokenizer, reward_model, num_shots)
#         PredictAgent.run(self, reset=reset)


# def format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
#     if reflections == []:
#         return ''
#     else:
#         return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# def EM(prediction, sentiment) -> bool:
#     return prediction.lower() == sentiment.lower()
