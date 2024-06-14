## Inspiration from: https://github.com/alexandrainst/d3a-llm-workshop/blob/main/notebooks/finetune.ipynb
#
# %pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.1 triton --index-url https://download.pytorch.org/whl/cu121
# %pip install "unsloth[cu121_ampere_torch211] @ git+https://github.com/unslothai/unsloth.git"

# standard libraries
import logging
import os
import json
import matplotlib.pyplot as plt

# third-party libraries
import torch
from omegaconf import OmegaConf
from omegaconf import DictConfig
import transformers
from unsloth import FastLanguageModel
from trl import DPOTrainer
from transformers import TrainingArguments
import wandb

# local libraries
import src.finetune.finetune_data_utils as ft_du
import src.utils.hydra_helper as hh


logger = logging.getLogger(__name__)

def train_llm(cfg: DictConfig): 
  transformers.utils.logging.set_verbosity_debug()

## Setup wandb
  api_key = os.getenv(cfg.wandb.api_key_env_var)
  wandb.login(key=api_key)
  wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity)

## Get model info
  model_name = cfg.llm[cfg.llm.tag.for_finetuning].name 
  model_dir = cfg.llm[cfg.llm.tag.for_finetuning].dir

  logger.notice(f'LLM to fine-tune <{model_name}>...')
  cfg_random_seed = cfg.finetuning.random_seed
  max_length = cfg.finetuning.model.max_seq_length 
  OmegaConf.update(cfg, "finetuning.model.model_name",              model_dir,        merge=False)
  OmegaConf.update(cfg, "finetuning.peft.random_state",             cfg_random_seed,  merge=False)
  OmegaConf.update(cfg, "finetuning.dpo_training_args.seed",        cfg_random_seed,  merge=False)
  OmegaConf.update(cfg, "finetuning.dpotrainer.max_length",         max_length,       merge=False)
  OmegaConf.update(cfg, "finetuning.dpotrainer.max_prompt_length",  max_length,       merge=False)
 
  if True:
## Load the Model
    logger.notice('NOTICE: Reference model...')
    ref_model, _ = FastLanguageModel.from_pretrained(**cfg.finetuning.model, token=None)
    logger.notice('NOTICE: Training model...')
    model, tokenizer = FastLanguageModel.from_pretrained(**cfg.finetuning.model, token=None)
    logger.notice('NOTICE: FastLanguageModel.get_peft_model()...')
    model = FastLanguageModel.get_peft_model(model, **cfg.finetuning.peft)
    logger.notice('NOTICE: print_trainable_parameters()...')
    model.print_trainable_parameters()

## Load and Prepare Data
    train_dataset = ft_du.load_dpo_dataset(cfg)


## Fine-tune setup
    dpo_trainer = DPOTrainer(
      model=model,
      ref_model=ref_model,
      tokenizer=tokenizer,
      train_dataset=train_dataset,
      eval_dataset=train_dataset, # intentionally use of train_dataset (dummy to avoid automatic split of train_dataset) 
      **cfg.finetuning.dpotrainer,
      args = TrainingArguments(
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        **cfg.finetuning.dpo_training_args
      ),
    )


  if False:
  ## Load the Model
    #logger.notice('NOTICE: Reference model...')
    #ref_model, _ = FastLanguageModel.from_pretrained(**cfg.finetuning.model, token=None)
    
    logger.notice('NOTICE: Training model...')
    model, tokenizer = FastLanguageModel.from_pretrained(**cfg.finetuning.model, token=None)
    logger.notice('NOTICE: FastLanguageModel.get_peft_model()...')
    model = FastLanguageModel.get_peft_model(model, **cfg.finetuning.peft, adapter_name='dpo_train')
    logger.notice('NOTICE: Load adapter again with different name for ref model()...')
    model.load_adapter(model, adapter_name='dpo_ref')

    logger.notice('NOTICE: print_trainable_parameters()...')
    model.print_trainable_parameters()

  ## Load and Prepare Data
    annotation_keys = ['SRC']
    train_dataset = ft_du.load_data_from_keys_in_dpo_format(cfg, keys=annotation_keys, selected_split='train')
    val_dataset = ft_du.load_data_from_keys_in_dpo_format(cfg, keys=annotation_keys, selected_split='val')

  ## Fine-tune setup
    dpo_trainer = DPOTrainer(
      model=model,
      model_adapter_name="dpo_train",
      ref_adapter_name="dpo_ref",
      tokenizer=tokenizer,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      **cfg.finetuning.dpotrainer,
      args = TrainingArguments(
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        **cfg.finetuning.dpo_training_args
      ),
    )

  # Log GPU stats before finetuning
  gpu_stats = torch.cuda.get_device_properties(0)
  start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
  logger.notice(
    f'Using {gpu_stats.name} GPU with {max_memory:.2f}[GB].'
    f'{start_gpu_memory:.2f}[GB] have been already reserved.'
  )

## Fine-tuning
  logger.notice('Training...')
  trainer_stats = dpo_trainer.train()

  # Log post-training GPU statistics
  used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
  used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
  used_percentage = round(used_memory / max_memory * 100, 3)
  lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
  logger.notice(
    f'Ended using {used_memory:.2f}[GB] GPU memory ({used_percentage:.2f}%), '
    f'of which {used_memory_for_lora:.2f}[GB] ({lora_percentage:.2f}%) was used for LoRa.'
  )

## Save data
  data_dir = hh.get_data_dir(cfg)

  # save training log
  try:
    log_filename = "training_log.json"    
    log_fullfilename = os.path.join(data_dir, f"{log_filename}")
    with open(log_fullfilename, 'w') as file:
        json.dump(trainer_stats, file, indent=2)
  except Exception as e:
    logger.error(f'Error - could not save training summary log file: {e}')


  # Model + tokenizer
  model_name = cfg.finetuning.training_settings.save_finetuned_model

  if model_name:
    try:
      model_save_path = '/home/evs/hf_llm/' + model_name
      logger.notice(f'Saving trained model in: {model_save_path}...\n(might take 5 min)')
      model.to('cpu')  # to avoid OOM in GPU
      merged_model = model.merge_and_unload()
      merged_model.save_pretrained(model_save_path)
      tokenizer.save_pretrained(model_save_path)
    except Exception as e:
      logger.error(f'Error - could not save finetuned model: {e}')

  else:
    logger.notice('Model not requested to be saved.')

  logger.notice('Finished llm_trainer.py')
  
## DEBUG
  logger.notice('DEBUG INFO:')
  print(model.config)