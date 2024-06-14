#  pytest -s tests/finetune/test_dpo_full_dataset.py


# standard libraries
import os

# third-party libraries

# local libraries
from tests.load_hydra_for_testing import load_hydra_config
from src.finetune import finetune_data_utils as ft_du


def test_dpo_full_dataset(logger):

  print('To be implemented (get inspiration from here and from data_utils.py)')
  return

  cfg = load_hydra_config()
  cfg.llm.tag.for_finetuning = 'munin'
  cfg.finetuning.training_settings.indices.train    = ['0-87', '89-193']
  cfg.finetuning.training_settings.indices.val      = ['0-64']
  cfg.finetuning.training_settings.pct_of_datasize  = 100
  cfg.finetuning.training_settings.shuffle          = False  

  training_keys = ['PER', 'ORG', 'QUOTES', 'SRC', 'GEN', 'FUN', 'ROL', 'EMP']
  selected_split = 'train'
  dataset = ft_du.load_data_from_keys_in_dpo_format(cfg, keys=training_keys, selected_split=selected_split)

  # Ensure directory exists
  directory = f'tmp/dpo_{selected_split}'
  os.makedirs(directory, exist_ok=True)

  with open(f'tmp/dpo_{selected_split}/all_prompts.txt', 'w') as file:
      for item in dataset['prompt']:
          file.write(item + '\n')

  with open(f'tmp/dpo_{selected_split}/all_chosen.txt', 'w') as file:
      for item in dataset['chosen']:
          file.write(item + '\n')

  with open(f'tmp/dpo_{selected_split}/all_rejected.txt', 'w') as file:
      for item in dataset['rejected']:
          file.write(item + '\n')

  # Ensure directories exist
  directories = [f'tmp/dpo_{selected_split}/prompt', f'tmp/dpo_{selected_split}/chosen', f'tmp/dpo_{selected_split}/rejected']
  for directory in directories:
      os.makedirs(directory, exist_ok=True)

  def write_files(data_list, directory):
      for index, item in enumerate(data_list):
          filename = f"{index:03d}.txt"
          filepath = os.path.join(directory, filename)
          with open(filepath, 'w') as file:
              file.write(item)

  # Write data to respective directories
  write_files(dataset['prompt'], f'tmp/dpo_{selected_split}/prompt')
  write_files(dataset['chosen'], f'tmp/dpo_{selected_split}/chosen')
  write_files(dataset['rejected'], f'tmp/dpo_{selected_split}/rejected')

