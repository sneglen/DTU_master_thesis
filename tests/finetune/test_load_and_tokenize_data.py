# standard libraries
import os

# third-party libraries
from transformers import AutoTokenizer

# local libraries
from tests.load_hydra_for_testing import load_hydra_config
from src.finetune import finetune_data_utils as ft_du

## Execution hint (from project root):
#  pytest -s tests/finetune

def test_load_and_tokenize_data(logger):

  cfg = load_hydra_config()
  cfg.finetuning.training_settings.indices.train = ['0','1','2']
  cfg.finetuning.training_settings.pct_of_datasize = 100
  cfg.finetuning.training_settings.shuffle = False  

  model_tag = 'munin'
  model_name = cfg.llm[model_tag].name

  logger.notice(f'\nLoading tokenizer from: {model_name}')

  tokenizer = AutoTokenizer.from_pretrained(model_name)

  ft_du.add_pad_token(tokenizer)

  train_dataset = ft_du.load_and_tokenize_data(cfg, tokenizer, key='QUOTES', selected_split='train')


  # Fetch the second article and quote by index (1)
  data_point = train_dataset[1] 

  # Decode the tokens to strings
  decoded_article = tokenizer.decode(data_point['input_ids'], skip_special_tokens=True)
  decoded_quote = tokenizer.decode(data_point['labels'], skip_special_tokens=True)

  # Retrieve raw token IDs
  raw_article_token_ids = data_point['input_ids']
  raw_quote_token_ids = data_point['labels']

## ARTICLE TEST
  # DECODED
  expected_article_first_part = "Read the following D"
  expected_article_last_part = "lokal tid.\n/ritzau/\n"

  assert decoded_article[:20] == expected_article_first_part, f"Decoded article does not match: {decoded_article[:20]} (ensure hydra cfg.data.train.indices: ['0-2'])"
  assert decoded_article[-20:] == expected_article_last_part, f"Decoded article does not match: {decoded_article[-20:]}"

  # RAW
  expected_article_token_ids_first_part = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  expected_article_token_ids_last_part = [14750, 668, 11619, 356, 5848, 357, 15606, 28719, 2558, 357, 305, 23356, 12158, 28723, 13, 28748, 24855, 581, 28748, 13]

  assert raw_article_token_ids[:20].tolist() == expected_article_token_ids_first_part, f"Raw article token IDs do not match: {raw_article_token_ids[:20].tolist()}"
  assert raw_article_token_ids[-20:].tolist() == expected_article_token_ids_last_part, f"Raw article token IDs do not match: {raw_article_token_ids[-20:].tolist()}"

## QUOTE TEST
  # DECODED
  expected_quote_first_part = "Forhåbentlig bliver "
  expected_quote_last_part = "er hende rigtig godt"

  assert decoded_quote[:20] == expected_quote_first_part, f"Decoded article does not match: {decoded_quote[:20]}"
  assert decoded_quote[-20:] == expected_quote_last_part, f"Decoded article does not match: {decoded_quote[-20:]}"

  # RAW
  expected_quote_token_ids_first_part = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  expected_quote_token_ids_last_part = [295, 262, 17081, 613, 12474, 267, 3225, 28725, 1683, 461, 641, 446, 2341, 400, 18773, 408, 8335, 326, 5525, 28707]

  assert raw_quote_token_ids[:20].tolist() == expected_quote_token_ids_first_part, f"Raw article token IDs do not match: {raw_quote_token_ids[:20].tolist()}"
  assert raw_quote_token_ids[-20:].tolist() == expected_quote_token_ids_last_part, f"Raw article token IDs do not match: {raw_quote_token_ids[-20:].tolist()}"


