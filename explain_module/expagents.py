from typing import List, Union, Literal
from utils.llm import *
from utils.prompts import * #REFLECT_INSTRUCTION, PREDICT_INSTRUCTION, PREDICT_REFLECT_INSTRUCTION, REFLECTION_HEADER
from utils.fewshots import EXPLANATION_EXAMPLES
from utils.tools import _add_missing_numbers
import os
import re



class PredictAgent:
    def __init__(self,args, exp_llm = str
                 ) -> None:

        self.args = args
        self.explanation_prompt = eval(args.explanation_prompt)
        # self.predict_examples = PREDICT_EXAMPLES
        # self.exp_examples = EXPLANATION_EXAMPLES
        self.exp_llm= exp_llm
        self._add_missing_numbers = _add_missing_numbers
        if exp_llm.split("/")[0] =="meta-llama":
            self.exp_llm_class = "llama"
            self.llm = LlamaLLM(model_name=self.exp_llm)
        else:
            self.exp_llm_class = "OpenAI_GPT4summarize_" + str(args.explanation_prompt).split('_')[-1]
            self.llm = OpenAILLM(model_name=self.exp_llm)

    def explanation_files_generation(self, data):
            exp_list = []
            past_exp_list = []

            for index, row in data.iterrows():
                # 현재 행의 데이터 준비
                if isinstance(row["data_X"], list):
                    data_X_str = "_".join(map(str, row["data_X"]))  # 리스트를 문자열로 변환
                else:
                    data_X_str = str(row["data_X"])  # 문자열로 변환

                exp_filename = f"{data_X_str}_"+self.exp_llm_class+".txt"
                exp_file_path = os.path.join('./explanations/', str(self.args.seq_len) + "/", exp_filename)

                # 현재 행의 파일 처리
                if os.path.exists(exp_file_path):
                    # print(str(i)+"/"+str(len(data)))
                    with open(exp_file_path, 'r+', encoding='utf-8') as file:
                        exp_output = file.read()
                        # \n\n을 \n으로 변환
                        updated_exp_output = exp_output.replace('\n\n', '\n')
                        # 번호 자동 부여 로직
                        updated_exp_output = self._add_missing_numbers(updated_exp_output,is_summary = False)
                        if exp_output != updated_exp_output:
                            file.seek(0)  # 파일 시작점으로 이동
                            file.write(updated_exp_output)
                            file.truncate()  # 파일 길이를 현재 내용으로 줄임
                            print(f"File updated: {exp_file_path}")
                else:
                    print(f"New explanations {index}/{len(data)} are uploaded.")
                    with open(exp_file_path, 'w', encoding='utf-8') as file:
                        exp_output = self._build_explanation(data_X = row["data_X"],data_Y = row["data_Y"],summary= row["news_summary"], start_X_date = row["start_X_date"], end_X_date = row["end_X_date"], Y_date = row["Y_date"], region_number = row["region_number"])
                        updated_exp_output = self._add_missing_numbers(exp_output,is_summary = False)
                        file.write(updated_exp_output)
                if row.past_data_X is not None:
                    # 이전 행의 데이터 준비 (index-1 번째)
                    if isinstance(row["past_data_X"], list):
                        past_data_X_str = "_".join(map(str, row["past_data_X"]))  # 리스트를 문자열로 변환
                    else:
                        past_data_X_str = str(row["past_data_X"])  # 문자열로 변환

                    past_exp_filename = f"{past_data_X_str}_"+self.exp_llm_class+".txt"
                    past_exp_file_path = os.path.join('./explanations/', str(self.args.seq_len) + "/", past_exp_filename)


                    if os.path.exists(past_exp_file_path):
                        with open(past_exp_file_path, 'r', encoding='utf-8') as past_file:
                            past_exp_output = past_file.read()
                            # 첫 번째 행 건너뛰기
                            # past_exp_output = "\n".join(past_exp_output.splitlines()[1:])
                    else:
                        print(f"New explanations for past data {index}/{len(data)} are uploaded.")
                        with open(exp_file_path, 'w', encoding='utf-8') as file:
                            exp_output = self._build_explanation(data_X = row["past_data_X"],data_Y = row["past_data_Y"],summary= row["past_news_summary"], start_X_date = row["start_past_X_date"], end_X_date = row["end_past_X_date"], Y_date = row["Y_date"], region_number = row["region_number"])
                            past_exp_output = self._add_missing_numbers(exp_output,is_summary = False)
                            file.write(past_exp_output)
                else:
                    past_exp_output = "Nothing"

                exp_list.append(updated_exp_output)
                past_exp_list.append(past_exp_output)


            return exp_list, past_exp_list
    def _build_explanation(self,data_X, data_Y, summary, start_X_date, end_X_date, Y_date, region_number) -> str:
        # if self.exp_llm_class == "meta-llama" or "llama":
        #     output = self.llm(self.explanation_prompt.format(
        #                         data_X = data_X,
        #                         data_Y = data_Y,
        #                         summary = summary,
        #                         examples = self.exp_examples,
        #                         pred_len = str(self.args.pred_len), seq_len = str(self.args.seq_len)),max_length=2000)
        # else:

        output = self.llm(self.explanation_prompt.format(
                            data_X = data_X,
                                data_Y = data_Y,
                                summary = summary, start_X_date = start_X_date, end_X_date=end_X_date, Y_date=Y_date, region_number = region_number,
                                pred_len = str(self.args.pred_len), seq_len = str(self.args.seq_len)))

        return output
    # def _add_missing_numbers(self, text):
    #     """번호 없는 문단에 이전 번호를 이어서 붙여주고 번호 뒤 불필요한 인덱스를 제거하는 함수"""
    #     lines = text.split('\n')  # 텍스트를 줄 단위로 분할
    #     last_number = 0
    #     numbered_lines = []

    #     for line in lines:
    #         # 정규 표현식: 번호와 뒤에 오는 (a), a)와 같은 불필요한 추가 인덱스를 제거
    #         match = re.match(r'(\d+)\.\s*((\([a-z]\)|[a-z]\))\s*)?(.*)', line)  # 숫자와 선택적 하위 인덱스 처리
    #         if match:
    #             last_number = int(match.group(1))  # 마지막 번호 업데이트
    #             content = match.group(4).strip()  # 추가 인덱스 제거 후 텍스트만 추출
    #             numbered_lines.append(f"{last_number}. {content}")
    #         elif line.strip():  # 비어있지 않은 줄에 대해 처리 (번호 없는 경우)
    #             last_number += 1
    #             numbered_lines.append(f"{last_number}. {line.strip()}")
    #         else:
    #             numbered_lines.append(line)  # 빈 줄은 그대로 유지

    #     return "\n".join(numbered_lines)
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
