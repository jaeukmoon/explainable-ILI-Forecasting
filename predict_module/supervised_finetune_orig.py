from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import sys

import torch
import torch.nn as nn
# import bitsandbytes as bnb
import transformers
import argparse
import warnings

from datasets import load_dataset
from predict_module import sft_dataloader
from accelerate import Accelerator
from typing import List, Union

def supervised_finetune(args,data):
    MICRO_BATCH_SIZE = args.micro_batch_size
    BATCH_SIZE = args.batch_size
    MAX_STEPS = None
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    EPOCHS = args.train_epochs
    LEARNING_RATE = args.learning_rate
    CUTOFF_LEN = args.cutoff_len
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    VAL_PCT = 0
    TARGET_MODULES = args.lora_target_modules
    OUTPUT_DIR = args.output_path

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size
    print(args.model_path)
    # quant_config = BitsAndBytesConfig(
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    # model = LlamaForCausalLM.from_pretrained(
    #     args.model_path,quantization_config=quant_config,
    #     device_map=device_map,
    # )
    if args.model_path == "lmsys/vicuna-7b-v1.5-16k":
        tokenizer = LlamaTokenizer.from_pretrained(
        args.model_path, add_eos_token=True
    )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, add_eos_token=True
    )
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,load_in_8bit=True,
        device_map=device_map,
        torch_dtype=torch.float16,
        eos_token_id = tokenizer.pad_token_id
    )
    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    # tokenizer.padding_side = "left"  # Allow batched inference

    val_set_size = VAL_PCT * len(data)



    # train_data = data.shuffle()
    # tokenized_data = train_data.map(generate_and_tokenize_prompt)
    # tokenized_data = tokenized_data.filter(lambda x: len(x['input_ids']) < filter_length)



    now_max_steps = max(
        (len(data)) // BATCH_SIZE * EPOCHS, EPOCHS)
    if args.resume_from_supervised_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            args.resume_from_supervised_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            pytorch_bin_path = checkpoint_name
            checkpoint_name = os.path.join(
                args.resume_from_supervised_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            if os.path.exists(checkpoint_name):
                os.rename(checkpoint_name, pytorch_bin_path)
                warnings.warn(
                    "The file name of the lora checkpoint'adapter_model.bin' is replaced with 'pytorch_model.bin'")
            else:
                args.resume_from_supervised_checkpoint = (
                    None  # So the trainer won't try loading its state
                )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

        train_args_path = os.path.join(
            args.resume_from_supervised_checkpoint, "trainer_state.json")

        if os.path.exists(train_args_path):
            import json
            base_train_args = json.load(open(train_args_path, 'r'))
            base_max_steps = base_train_args["max_steps"]
            resume_scale = base_max_steps / now_max_steps
            if base_max_steps > now_max_steps:
                warnings.warn("epoch {} replace to the base_max_steps {}".format(
                    EPOCHS, base_max_steps))
                EPOCHS = None
                MAX_STEPS = base_max_steps
            else:
                MAX_STEPS = now_max_steps
    else:
        MAX_STEPS = now_max_steps


    model.print_trainable_parameters()


    dataloader = sft_dataloader.SFTDataLoader(args = args,
        data = data, CUTOFF_LEN = CUTOFF_LEN, tokenizer = tokenizer)
    train_data, val_data = dataloader.load_data()

    accelerator = Accelerator()
    from transformers import TrainerCallback

    class MemoryManagementCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.empty_cache()

    trainer = accelerator.prepare(transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            # max_steps=highest_number,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=20,
            # evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            # eval_steps=args.eval_steps if val_set_size > 0 else None,
            save_steps=args.save_steps,
            output_dir=OUTPUT_DIR,
            save_total_limit=30,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="wandb" if args.wandb else [],
            ignore_data_skip=args.ignore_data_skip,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False), callbacks= [MemoryManagementCallback()]
    ))
    model.config.use_cache = False
    model.print_trainable_parameters()

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print("\n If there's a warning about missing keys above, please disregard :)")

    with torch.autocast("cuda"):
        trainer.train(resume_from_checkpoint=args.resume_from_supervised_checkpoint)

    model.save_pretrained(OUTPUT_DIR)
    model_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
    torch.save({}, model_path)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return model
