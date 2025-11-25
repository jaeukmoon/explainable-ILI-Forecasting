from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test, test_vis
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from data_load.dataloader import DataLoader_
from explain_module.expagents import *
from predict_module.merge_peft_adapter import *
from predict_module.supervised_finetune import *
from utils.prompts import *

# from models.textllama31 import llama31


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--multimodal', type=int, default=0)
parser.add_argument("--exp_llm", type=str, default="gpt-4o-2024-11-20")
parser.add_argument("--summarize_llm", type=str, default="gpt-4o-2024-11-20")
parser.add_argument("--ex_data", type=str, default="report")


parser.add_argument("--output_path", type=str, default="./saved_models/Llama-3.2-3B-Instruct-CoT-date")
parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct") #  "lmsys/vicuna-7b-v1.5-16k" "meta-llama/Llama-3.2-3B-Instruct"
parser.add_argument("--prompt_template", type=str, default="ILI_PREDICT_INSTRUCTION_PROMPT_INSTRUCT_CoT_date_0109")
parser.add_argument("--resume_from_supervised_checkpoint", type=str, default=False)
parser.add_argument("--eval_steps", type=int, default=150)
parser.add_argument("--save_steps", type=int, default=500)
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--ignore_data_skip", type=str, default="False")

parser.add_argument('--root_path', type=str, default='./dataset/ILI/')
parser.add_argument('--data_path', type=str, default='ILI_region_feature_2020-2024.csv')
parser.add_argument('--news_data_path', type=str, default='./dataset/US_news/influenza_data')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=0)
parser.add_argument('--target', type=str, default='5')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=100)

parser.add_argument('--seq_len', type=int, default=5)
parser.add_argument('--pred_len', type=int, default=5)
parser.add_argument('--label_len', type=int, default=0)

parser.add_argument('--learning_rate', type=float, default=3e-4)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--micro_batch_size', type=int, default=2)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--cutoff_len', type=int, default=256)
parser.add_argument('--lora_target_modules', type=str, default=["gate_proj", "down_proj", "up_proj"] , help='lora target modules') #["gate_proj", "down_proj", "up_proj"]
parser.add_argument('--lora_r', type=int, default=16)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=int, default=0.05)



# parser.add_argument('--lradj', type=str, default='type1')
# parser.add_argument('--patience', type=int, default=3)
# parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--num_workers', type=int, default=10)
# parser.add_argument('--gpt_layers', type=int, default=6)
# parser.add_argument('--backbone', type=str, default='gpt2')
# parser.add_argument('--textmodel', type=str, default='gpt2')
# parser.add_argument('--e_layers', type=int, default=3)
# parser.add_argument('--d_model', type=int, default=768)
# parser.add_argument('--n_heads', type=int, default=4)
# parser.add_argument('--d_ff', type=int, default=768)
# parser.add_argument('--dropout', type=float, default=0.2)
# parser.add_argument('--enc_in', type=int, default=862)
# parser.add_argument('--c_out', type=int, default=862)
# parser.add_argument('--patch_size', type=int, default=24)
# parser.add_argument('--kernel_size', type=int, default=25)

# parser.add_argument('--loss_func', type=str, default='mse')
# parser.add_argument('--pretrain', type=int, default=1)
# parser.add_argument('--freeze', type=int, default=1)
# parser.add_argument('--model', type=str, default='model')
# parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--max_len', type=int, default=-1)
# parser.add_argument('--hid_dim', type=int, default=16)
# parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--cos', type=int, default=0)



args = parser.parse_args()

def train():

    mses = []
    maes = []

    mses_temp = []
    maes_temp = []

    results = []
    min_mse = float('inf')  # 최소 MSE를 저장할 변수, 무한대로 초기화
    best_epoch_result = {'mse': None, 'mae': None, 'epoch': None, 'model_state': None}  # 최적의 결과를 저장할 딕셔너리


    for ii in range(args.itr):

        # setting = '{}_multimodal{}_{}%_{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, args.multimodal, args.percent,args.backbone, args.textmodel,args.seq_len, args.label_len, args.pred_len,
        #                                                                 args.d_model, args.n_heads, args.e_layers, args.gpt_layers,
        #                                                                 args.d_ff, args.embed, ii)
        train_data, train_loader = data_provider(args, 'train') # 106
        vali_data, vali_loader = data_provider(args, 'val') # 27
        test_data, test_loader = data_provider(args, 'test')  # 90
        device = torch.device('cuda:0')

        time_now = time.time()
        train_steps = len(train_loader)

        # DATAFRAME 형태로 X,Y, news 데이터를 쌓음
        dataloader = DataLoader_(args)
        data_news = dataloader.load(args,train_loader,train_data,vali_data,test_data,flag = "train")
        data_news_exp = data_news

        # Explanation 생성
        exp_agent = PredictAgent(args=args, exp_llm=args.exp_llm)
        exp_filename = args.exp_llm
        exp_file_path = os.path.join('./explanations/', exp_filename)
        exp_list, past_exp_list = exp_agent.explanation_files_generation(data=data_news)
        data_news_exp['explanations'] = exp_list
        data_news_exp['past_explanations'] = past_exp_list
        print("Explanation generation finished")
        data_news_exp_pastpop = data_news_exp[data_news_exp['start_past_X_date'].notna()].reset_index(drop = True)

        # Train supervised policy
        output_dir = args.output_path+"-"+str(args.train_epochs)+"epochs"
        model = supervised_finetune(args,data = data_news_exp_pastpop, model_name=output_dir)
        merged_model = merge_peft_adapter(model_name=output_dir)

def test():
    output_name = f"{args.output_path}-adapter-merged"
    if args.model_path == "lmsys/vicuna-7b-v1.5-16k":
        tokenizer = LlamaTokenizer.from_pretrained(
        args.model_path, add_eos_token=True
    )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, add_eos_token=True
        )
    pipeline = transformers.pipeline(
    "text-generation",
    model=output_name,
    device_map="auto",tokenizer=tokenizer)

    # 데이터 준비
    test_data, test_loader = data_provider(args, 'test')  # len(test_loader): 74
    dataloader = DataLoader_(args)
    data_news_test = dataloader.load(args,test_loader,test_data)

    # Hugging Face Dataset으로 변환
    from datasets import Dataset
    dataset = Dataset.from_pandas(data_news_test)

    # Generate response 함수 정의
    def generate_response_example(row):
        prompt_template = args.prompt_template
        prompt = prompt_template.format(
            data_X=row['data_X'],
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            summary=row['news_summary']
        )
        return {"prompt": prompt}

    # Dataset에 prompt 추가
    dataset = dataset.map(generate_response_example)

    # Batch inference를 수행
    results = pipeline(
    list(dataset['prompt']),
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    max_new_tokens=500,
    pad_token_id=tokenizer.eos_token_id)

    def process_result(result, row):
        predicted_Y = result.split('### Output:')[2] if '### Output:' in result else ""
        ILI_occur = predicted_Y.split('Explanation:')[0].split('Future ILI occurrences:')[-1]
        explanation = predicted_Y.split('Explanation:')[0]
        return {
            'predicted_Y': predicted_Y,
            'text': result,
            'ILI_occur': ILI_occur,
            'explanation': explanation
        }

    # Dataset 결과 저장
    output_data = []
    for result, row in zip(results, data_news_test.itertuples()):
        processed = process_result(result[0]['generated_text'], row)
        print(processed)
        output_data.append(processed)

    output_df = pd.DataFrame(output_data)
    data_news_test['predicted_Y'] = output_df['predicted_Y']
    data_news_test['ILI_occur'] = output_df['ILI_occur']
    data_news_test['explanation'] = output_df['explanation']
    data_news_test['text'] = output_df['text']


    # data_news_test[['predicted_Y', 'text']] = data_news_test.apply(generate_response, axis = 1)
    csv_name = 'test_predicted'+args.output_path.split('/')[-1]+str(args.train_epochs)+'.csv'
    data_news_test.to_csv(csv_name)


    mse_temp, mae_temp = test(merged_model, test_data, test_loader, args, device, ii)

def test_live_new():
    # --------------------------
    # 1) 모델/토크나이저 등 불러오기(한 번만)
    # --------------------------
    output_name = f"{args.output_path}" + "-" + str(args.train_epochs) + "epochs-adapter-merged"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, add_eos_token=True
    )
    pipeline = transformers.pipeline(
        "text-generation",
        model=output_name,
        device_map="auto",
        tokenizer=tokenizer
    )

    # --------------------------
    # 2) 테스트 데이터 로드
    # --------------------------
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    dataloader = DataLoader_(args)
    data_news_test = dataloader.load(
        args, test_loader, train_data, vali_data, test_data, flag="test"
    )

    # --------------------------
    # 3) 과거 설명(past_explanations) 불러오기
    # --------------------------
    exp_agent = PredictAgent(args=args, exp_llm=args.exp_llm)
    exp_filename = args.exp_llm
    exp_file_path = os.path.join('./explanations/', exp_filename)
    exp_list, past_exp_list = exp_agent.explanation_files_generation(data=data_news_test)
    data_news_test['past_explanations'] = past_exp_list

    # --------------------------
    # 4) 실제로 ILI 예측을 수행할 데이터만 필터링
    # --------------------------
    data_news_test_pastpop = data_news_test[data_news_test['past_data_X'].notna()].reset_index(drop=True)

    # 중간에 어떤 문제가 생기더라도 우선 raw 데이터를 저장해 둠
    partial_csv_name = "test_data" + output_name.split('/')[-1] + ".csv"
    data_news_test_pastpop.to_csv(partial_csv_name, index=False)

    # --------------------------
    # 5) Batch Inference를 위한 프롬프트 생성
    # --------------------------
    prompts = []
    for i, row in data_news_test_pastpop.iterrows():
        prompt_template = eval(args.prompt_template)
        prompt = prompt_template.format(start_X_date = row["start_X_date"],
                                end_X_date = row["end_X_date"],
                                data_X = row["data_X"],
                                summary = row["news_summary"],
                                Y_date = row["Y_date"],
                                start_past_X_date = row["start_past_X_date"],
                                end_past_X_date = row["end_past_X_date"],
                                past_data_X = row["past_data_X"],
                                past_summary = row["past_news_summary"],
                                past_Y_date = row["past_Y_date"],
                                past_data_Y = row["past_data_Y"],
                                past_explanations = row["past_explanations"],
                                pred_len = str(args.pred_len), seq_len = str(args.seq_len))
        prompts.append(prompt)

    # --------------------------
    # 6) 한 번에 추론 수행
    # --------------------------
    results = pipeline(
        prompts,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        max_new_tokens=500,
        pad_token_id=tokenizer.eos_token_id
    )

    # --------------------------
    # 7) 추론 결과를 파싱(parsing)하기 위한 함수
    # --------------------------
    def parse_prediction(generated_text: str):
        """
        모델의 출력 결과 중 'Future ILI occurrences'와 'Explanation' 부분을 잘라서 추출하는 함수입니다.
        '[[OUTPUT]]' 문자열을 기준으로 분리하도록 되어 있으므로, 프롬프트 설계에 맞추어 적절히 변경할 수 있습니다.
        """
        if '[[OUTPUT]]' not in generated_text:
            return "", "", generated_text

        # [[OUTPUT]] 기준으로 분리
        split_output = generated_text.split('[OUTPUT]')
        if len(split_output) < 3:
            return "", "", generated_text

        prediction_part = split_output[2]

        # Future ILI occurrences 추출
        ili_occurrences = ""
        explanation = ""

        if 'Future ILI occurrences:' in prediction_part:
            ili_occurrences = prediction_part.split('Future ILI occurrences:')[-1].split('Explanation:')[0].strip()

        if 'Explanation:' in prediction_part:
            explanation = prediction_part.split('Explanation:')[-1].strip()

        return ili_occurrences, explanation, prediction_part

    # 최종 결과 CSV 파일 이름
    final_csv_name = 'test_predicted_' + args.output_path.split('/')[-1] + str(args.train_epochs) + 'epochs.csv'

    # --------------------------
    # 8) 결과를 DataFrame에 저장하면서 진행 상황 출력 & 중간 저장
    # --------------------------
    for idx, (result, row) in enumerate(zip(results, data_news_test_pastpop.itertuples()), 1):
        # result가 리스트 형태라면 첫 번째 요소에서 generated_text를 가져옴
        model_output = result[0]['generated_text'] if isinstance(result, list) else result['generated_text']

        ILI_occur, explanation, predicted_Y = parse_prediction(model_output)

        # DataFrame에 추론 결과 갱신
        data_news_test_pastpop.at[row.Index, 'predicted_Y'] = predicted_Y
        data_news_test_pastpop.at[row.Index, 'text'] = model_output
        data_news_test_pastpop.at[row.Index, 'ILI_occur'] = ILI_occur
        data_news_test_pastpop.at[row.Index, 'explanation'] = explanation

        # 현재 몇 번째 row를 처리했는지, 해당 row의 정보 등 로깅
        print(f"Processing row {idx}/{len(data_news_test_pastpop)}")
        print("start_data_X:", row.start_X_date)
        print("Actual_Y:", row.data_Y)
        print("Predicted Output:")
        print(predicted_Y)
        print("-----")

        # N번째마다 중간 저장(예: 10번째마다)
        if idx % 10 == 0:
            data_news_test_pastpop.to_csv(final_csv_name, index=False)
            print(f"Intermediate save at row {idx}")

    # --------------------------
    # 9) 최종 CSV 저장
    # --------------------------
    data_news_test_pastpop.to_csv(final_csv_name, index=False)
    print(f"Final results saved to {final_csv_name}")
def test_live():
    output_name = f"{args.output_path}"+"-"+str(args.train_epochs)+"epochs-adapter-merged"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, add_eos_token=True
    )
    pipeline = transformers.pipeline(
    "text-generation",
    model=output_name,
    device_map="auto",tokenizer=tokenizer)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')  # len(test_loader): 74
    dataloader = DataLoader_(args)
    data_news_test = dataloader.load(args,test_loader,train_data,vali_data, test_data,flag="test")

    exp_agent = PredictAgent(args=args, exp_llm=args.exp_llm)
    exp_filename = args.exp_llm
    exp_file_path = os.path.join('./explanations/', exp_filename)
    exp_list, past_exp_list = exp_agent.explanation_files_generation(data=data_news_test)
    # data_news_exp['explanations'] = exp_list
    data_news_test['past_explanations'] = past_exp_list
    data_news_test_pastpop = data_news_test[data_news_test['past_data_X'].notna()].reset_index(drop = True)
    data_news_test_pastpop.to_csv("test_data"+output_name.split('/')[-1]+"1.csv")

    def generate_response(row):
        prompt_template = eval(args.prompt_template)
        prompt = prompt_template.format(start_X_date = row["start_X_date"],
                                end_X_date = row["end_X_date"],
                                data_X = row["data_X"],
                                summary = row["news_summary"],
                                Y_date = row["Y_date"],
                                start_past_X_date = row["start_past_X_date"],
                                end_past_X_date = row["end_past_X_date"],
                                past_data_X = row["past_data_X"],
                                past_summary = row["past_news_summary"],
                                past_Y_date = row["past_Y_date"],
                                past_data_Y = row["past_data_Y"],
                                past_explanations = row["past_explanations"],
                                pred_len = str(args.pred_len), seq_len = str(args.seq_len))
        result = pipeline(prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        max_new_tokens=500,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )[0]['generated_text']
        predicted_Y = result.split('[OUTPUT]')[2] if '[OUTPUT]' in result else ""
        ILI_occur = predicted_Y.split('Explanation:')[0].split('Future ILI occurrences:')[-1]
        explanation = predicted_Y.split('Explanation:')[-1]
        print('start_data_X:',row['start_X_date'])
        print('Actual_Y:', row['data_Y'])
        print('past_data_Y',row["past_data_Y"])
        print(predicted_Y)
        return pd.Series({'predicted_Y': predicted_Y, 'text': result, 'ILI_occur': ILI_occur, 'explanation': explanation})


    data_news_test_pastpop[['predicted_Y', 'text', 'ILI_occur', 'explanation']] = data_news_test_pastpop.apply(generate_response, axis = 1)
    csv_name = 'test_predicted'+args.output_path.split('/')[-1]+str(args.train_epochs)+'epochs.csv'
    data_news_test_pastpop.to_csv(csv_name)

if __name__=='__main__':
    # train()
    # test()
    test_live()