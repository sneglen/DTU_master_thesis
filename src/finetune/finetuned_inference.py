# standard libraries
import logging

# third-party libraries
from omegaconf import OmegaConf
from omegaconf import DictConfig
import transformers
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# local libraries


logger = logging.getLogger(__name__)

def main(cfg: DictConfig): 

## Get model info
  model_name = cfg.llm[cfg.llm.tag.for_inference].name 
  model_dir = cfg.llm[cfg.llm.tag.for_inference].dir

  logger.notice(f'LLM for inference: {model_name}')
  cfg_random_seed = cfg.finetuning.random_seed 
  OmegaConf.update(cfg, "finetuning.model.model_name",   model_dir,       merge=False)
  OmegaConf.update(cfg, "finetuning.peft.random_state",  cfg_random_seed, merge=False)
  OmegaConf.update(cfg, "finetuning.sft_training_args.seed", cfg_random_seed, merge=False)

## Load the Model
  logger.notice('Load the model...')
  model, tokenizer = FastLanguageModel.from_pretrained(**cfg.finetuning.model, token=None)

  # Format prompt
  message = [
      {"role": "system", "content": "You are a helpful assistant chatbot."},
      {"role": "user", "content": "What is a Large Language Model?"}
  ]

## EVS PLAY: add_generation_prompt=True, tokenize=False

  prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
  # must be: tokenize=False

# logger.notice(f'DEFAULT TEMPLATE: {tokenizer.default_chat_template}')

## DEFAULT TEMPLATE: 
# {% if messages[0]['role'] == 'system' %}
#     {% set loop_messages = messages[1:] %}
#     {% set system_message = messages[0]['content'] %}
# {% elif false == true and not '<<SYS>>' in messages[0]['content'] %}
#     {% set loop_messages = messages %}
#     {% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.' %}
# {% else %}
#     {% set loop_messages = messages %}
#     {% set system_message = false %}
# {% endif %}
# {% for message in loop_messages %}
#     {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
#         {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
#     {% endif %}
#     {% if loop.index0 == 0 and system_message != false %}
#         {% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}
#     {% else %}
#         {% set content = message['content'] %}
#     {% endif %}
#     {% if message['role'] == 'user' %}
#         {{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
#     {% elif message['role'] == 'system' %}
#         {{ '<<SYS>>\n' + content.strip() + '\n<</SYS>>\n\n' }}
#     {% elif message['role'] == 'assistant' %}
#         {{ ' '  + content.strip() + ' ' + eos_token }}
#     {% endif %}
# {% endfor %}


  # Create pipeline
  pipeline = transformers.pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer
  )

  # Generate text
  sequences = pipeline(
      prompt,
      do_sample=True,
      temperature=0.7,
      top_p=0.9,
      num_return_sequences=1,
      max_length=200,
  )
  print(sequences[0]['generated_text'])

