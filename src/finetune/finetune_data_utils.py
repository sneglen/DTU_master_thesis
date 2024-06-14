# standard libraries
import logging
import random
import re
import os

# third-party libraries
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
import pandas as pd

# local libraries
import src.data.data_utils as du
import src.utils.hydra_helper as hh
from src.spe.annotator_instructions import get_instructions


logger = logging.getLogger(__name__)


def add_pad_token(tokenizer):
  if tokenizer.pad_token is None:
    logger.notice('<pad_token> = None. Needs to be set.')
    if tokenizer.eos_token is not None:
        logger.notice(f'<pad_token> = <eos_token> = {tokenizer.eos_token}')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        logger.error('Warning: <eos_token> = None. <pad_token> not set.')
  else:
    logger.notice(f'pad_token is already set. <pad_token> = {tokenizer.pad_token}')


class CustomDataset(TorchDataset):
  def __init__(self, tokenizer, articles, quotes):
    self.tokenizer = tokenizer
    self.articles = articles
    self.quotes = quotes
  
  def __len__(self):
    return len(self.articles)
  
  def __getitem__(self, idx):
    article = self.articles[idx]
    quote = self.quotes[idx]
    
    # Tokenize the input (articles) and label (quotes)
    input_ids = self.tokenizer(article, padding="max_length", truncation=True, max_length=2048, return_tensors="pt")
    labels = self.tokenizer(quote, padding="max_length", truncation=True, max_length=2048, return_tensors="pt")
    
    return {
      # Remove (squeeze out) batch dimension
      'input_ids': input_ids['input_ids'].squeeze(0),  
      'labels': labels['input_ids'].squeeze(0)         
    }


def load_and_tokenize_data(cfg, tokenizer, key, selected_split='train'):
  logger.notice('WARNING: load_and_tokenize_data() is deprecated. Use load_data_from_keys_in_chatML_format() instead.')
  logger.notice(f'Loading <{selected_split}> dataset...')

  pct_of_datasize = cfg.finetuning.training_settings.pct_of_datasize
  shuffle         = cfg.finetuning.training_settings.shuffle  
  if shuffle and selected_split == 'val':
    logger.notice('Shuffle disabled for validation dataset.')
    shuffle = False


  # Get datasplit to annotate (articles and indices)
  articles = du.load_articles(cfg.data.split[selected_split])
  DR_indices = hh.parse_article_DR_indices(cfg.finetuning.training_settings.indices[selected_split])

  seed = cfg.finetuning.get('random_seed', 42) # default seed if not provided
  random.seed(seed)

  if not DR_indices:
    logger.notice('No articles found.')
    return None
  
  # Expand articles into article-quote pairs
  expanded_articles_with_instructions = []
  expanded_quotes = []
  for index in DR_indices:
    article = articles[index]
    article_text = article['TEXT']
    quotes = article['QUOTES']
    
    for quote in quotes:
      # Fetch the instructional text for the article
      instructions = get_instructions(key=key, article_text=article_text, quote_text=None, per_dict=None, org_dict=None, src_dict=None)
      # Append {instructions, quotes} in corresponding lists
      expanded_articles_with_instructions.append(instructions)
      expanded_quotes.append(quote['TEXT'])

  logger.notice(f'Number of article-quote pairs: {len(expanded_articles_with_instructions)}')
  total_pairs = list(zip(expanded_articles_with_instructions, expanded_quotes))

  # Reduce dataset size
  if pct_of_datasize < 100:
    reduced_size = int(len(total_pairs) * (pct_of_datasize / 100))
    total_pairs = total_pairs[:reduced_size]
    logger.notice(f'Reduced dataset size to {len(total_pairs)} pairs.') 
  else:
    logger.notice(f'Dataset size: {len(total_pairs)} pairs.') 

  # Shuffle combined article-quote pairs
  if shuffle:
      random.shuffle(total_pairs)
      logger.notice('Dataset shuffled.')

  expanded_articles_with_instructions, expanded_quotes = zip(*total_pairs)
  dataset = CustomDataset(tokenizer, list(expanded_articles_with_instructions), list(expanded_quotes))

  return dataset


def create_conversation(sample: dict) -> dict[str, list[dict[str, str]]]:
  """This converts the sample to the standardised ChatML format.

  Args:
    sample:
      The data sample.

  Returns:
    The sample set up in the ChatML format.
  """
  return {
    "messages": [
      {"role": "system", "content": sample["system_prompt"]},
      {"role": "user", "content": sample["question"]},
      {"role": "assistant", "content": sample["response"]}
    ]
  }


def load_data_in_chatML_format(cfg, key, selected_split='train'):

  logger.notice(f'Loading <{selected_split}> dataset...')
  pct_of_datasize = cfg.finetuning.training_settings.pct_of_datasize

  shuffle = cfg.finetuning.training_settings.shuffle  
  if shuffle and selected_split == 'val':
    logger.notice('Shuffle disabled for validation dataset.')
    shuffle = False

  articles = du.load_articles(cfg.data.split[selected_split])
  DR_indices = hh.parse_article_DR_indices(cfg.finetuning.training_settings.indices[selected_split])

  seed = cfg.finetuning.get('random_seed', 42) # default seed if not provided
  random.seed(seed)

  if not DR_indices:
    logger.notice('No articles found.')
    return None
  
  # Expand articles into article-quote pairs
  expanded_articles_with_instructions = []
  expanded_quotes = []
  for index in DR_indices:
    article = articles[index]
    article_text = article['TEXT']
    quotes = article['QUOTES']
    
    for quote in quotes:
      # Fetch the instructional text for the article
      instructions = get_instructions(key=key, article_text=article_text, quote_text=None, per_dict=None, org_dict=None, src_dict=None)
      # Append {instructions, quotes} in corresponding lists
      expanded_articles_with_instructions.append(instructions)
      expanded_quotes.append(quote['TEXT'])

  logger.notice(f'Number of article-quote pairs: {len(expanded_articles_with_instructions)}')
  total_pairs = list(zip(expanded_articles_with_instructions, expanded_quotes))

  # Reduce dataset size
  if pct_of_datasize < 100:
    reduced_size = int(len(total_pairs) * (pct_of_datasize / 100))
    total_pairs = total_pairs[:reduced_size]
    logger.notice(f'Reduced dataset size to {len(total_pairs)} pairs.') 

  # Shuffle combined article-quote pairs
  if shuffle:
      random.shuffle(total_pairs)
      logger.notice('Dataset shuffled.')

  # Create DataFrame from total_pairs
  df = pd.DataFrame(total_pairs, columns=['question', 'response'])
  df['system_prompt'] = [''] * len(df)  # Add empty system prompts if necessary

  # Create 'messages' column
  df['messages'] = df.apply(lambda row: [
      {'role': 'system', 'content': row['system_prompt']},
      {'role': 'user', 'content': row['question']},
      {'role': 'assistant', 'content': row['response']}
  ], axis=1)

  # Convert to dataset
  chatml_dataset = Dataset.from_pandas(df)

  return chatml_dataset


def load_data_from_keys_in_chatML_format(cfg, keys, selected_split='train'):

  quote_keys = ['QUOTES', 'SRC', 'GEN', 'FUN', 'EMP', 'ROL']

  logger.notice(f'Loading <{selected_split}> dataset from keys: {keys}...')
  pct_of_datasize = cfg.finetuning.training_settings.pct_of_datasize

  shuffle = cfg.finetuning.training_settings.shuffle  
  if shuffle and selected_split == 'val':
    logger.notice('Shuffle disabled for validation dataset.')
    shuffle = False

  articles = du.load_articles(cfg.data.split[selected_split])
  DR_indices = hh.parse_article_DR_indices(cfg.finetuning.training_settings.indices[selected_split])

  seed = cfg.finetuning.get('random_seed', 42) # default seed if not provided
  random.seed(seed)

  if not DR_indices:
    logger.notice('No articles found.')
    return None
  
  # Expand articles and generate instructions+target pairs: 
  # 1) instructions (including article + others) 
  # 2) targets
  list_instructions_with_article = []
  list_target = []

  for index in DR_indices:
    logger.notice(f'** Loading article [{index}]...')
    article = articles[index]  
    article_text = article['TEXT']
    for key in keys:
      #logger.notice(f'   Loading key [{key}]')

      if key == 'PER' or key == 'ORG':
        per_json_dict = None
        org_json_dict = None
        quote_text    = ""
        src_json_dict = None
        instructions = get_instructions(key=key, 
                                        article_text = article_text, 
                                        quote_text   = quote_text, 
                                        per_dict     = per_json_dict, 
                                        org_dict     = org_json_dict, 
                                        src_dict     = src_json_dict)
        #target =  ', '.join(article[key])
        target =  '\n'.join(article[key])
        #target = '[' + target + ']'
      
        list_instructions_with_article.append(instructions)
        list_target.append(target)


      elif key in quote_keys:
        for quote_idx, quote in enumerate(article['QUOTES']):
          quote_text = quote['TEXT']
          per_json_dict = {'PER': article['PER']}
          org_json_dict = {'ORG': article['ORG']}
          src_json_dict = {'SRC': quote['SRC']}

          if key == 'SRC' and (not article['PER'] or article['PER'] == [] or article['PER'] == ['']):
            logger.notice(f'Skip generating SRC datapoint in quote[{quote_idx}]: No person annotated.')
            continue

          instructions = get_instructions(key=key, 
                                           article_text = article_text, 
                                           quote_text   = quote_text, 
                                           per_dict     = per_json_dict, 
                                           org_dict     = org_json_dict, 
                                           src_dict     = src_json_dict)
          
          # Fetch key target
          if key == 'QUOTES':
            target = quote_text

          elif key == 'SRC':
            if quote['SRC'] is not None:
              target = ', '.join(quote['SRC'])
            else:
              target = ''

          elif key == 'GEN':
            if quote['GEN'] is not None:
              target = quote['GEN']
            else:
              logger.notice('ERROR: GEN not found in quote')
              target = 'X'

          elif key == 'FUN':
            if quote['FUN'] is not None:
              target = quote['FUN']
            else:
              logger.notice('ERROR: FUN not found in quote')
              target = 'Other'

          elif key == 'ROL':
            if quote['ROL'] is not None:
              target = ', '.join(quote['ROL'])
            else:
              target = ''

          elif key == 'EMP':
            if quote['EMP'] is not None:
              target = ', '.join(quote['EMP'])
            else:
              target = ''

          list_instructions_with_article.append(instructions)
          list_target.append(target)

  total_pairs = list(zip(list_instructions_with_article, list_target))

  # Reduce dataset size
  original_pairs_count = len(total_pairs)
  if pct_of_datasize < 100:
    reduced_size = int(len(total_pairs) * (pct_of_datasize / 100))
    total_pairs = total_pairs[:reduced_size]
    logger.notice(f'Dataset size reduced from {original_pairs_count} to {len(total_pairs)} pairs.') 
  else:
    logger.notice(f'Dataset size of {original_pairs_count} pairs.') 


  # Shuffle combined article-quote pairs
  if shuffle:
      random.shuffle(total_pairs)
      logger.notice('Dataset shuffled.')

  # Create DataFrame from total_pairs
  df = pd.DataFrame(total_pairs, columns=['question', 'response'])
  df['system_prompt'] = [''] * len(df)  # Add empty system prompts if necessary

  # Create 'messages' column
  df['messages'] = df.apply(lambda row: [
      {'role': 'system', 'content': row['system_prompt']},
      {'role': 'user', 'content': row['question']},
      {'role': 'assistant', 'content': row['response']}
  ], axis=1)

  # Convert to dataset
  chatml_dataset_dict = Dataset.from_pandas(df)

  return chatml_dataset_dict


def get_random_fake_quotes(article_txt, max_num_quotes=3, seed=42):
  random.seed(seed)

  num_quotes = random.randint(1, max_num_quotes)

  # Split the article into sentences (delimiters: ".", "!", "?", "new line")
  sentences = re.split(r'(?<=[.!?])\s+', article_txt)

  # Filter out short sentences (with less than 3 words)
  long_sentences = [sentence.strip() for sentence in sentences if len(sentence.strip().split()) >= 3]     

  if len(long_sentences) < num_quotes:
      logger.notice(f'Could not get {num_quotes} fake quotes. The article contains only {len(sentences)} sentences.')
      num_quotes = len(long_sentences)
  
  # Select random valid_sentences
  random_fake_quotes = random.sample(long_sentences, num_quotes)
  
  formatted_random_fake_quotes = ', '.join([f'"{sentence}"' for sentence in random_fake_quotes])

  return formatted_random_fake_quotes


def save_dpo_data_entry_to_file(cfg, entry, filename):

  data_dir = hh.get_data_dir(cfg)
  dpo_data_dir =  os.path.join(data_dir, "dpo_data")

  # create dpo_data_dir if is does not exist
  os.makedirs(dpo_data_dir, exist_ok=True)

  full_filename = os.path.join(dpo_data_dir, filename)

  try:
    with open(full_filename, "w") as file:
      file.write(entry)
  except Exception as e:
    logger.error(f"Could not save raw response to file <{filename}>: {e}")


def format_quotes_with_text_key(list_of_quotes_str):

    if list_of_quotes_str == '':
        return ''
    
    # Split the quotes
    quotes = re.split(r'"\s*,\s*"', list_of_quotes_str.strip(' "'))
    
    # Format each quote
    formatted_quotes_with_TEXT = ', '.join([f'{{"TEXT": "{quote}"}}' for quote in quotes])
    
    return formatted_quotes_with_TEXT

def load_dpo_dataset(cfg):
  # hydra parameters
  dpo_keys  = cfg.finetuning.training_settings.dpo_keys
  shuffle   = cfg.finetuning.training_settings.shuffle  
  seed      = cfg.finetuning.random_seed
  
  PER_ORG_QUOTES_keys = ['PER', 'ORG', 'QUOTES']
  SRC_GEN_FUN_EMP_ROL_keys = ['SRC', 'GEN', 'FUN', 'EMP', 'ROL']

  logger.notice(f'Loading DPO dataset for keys: {dpo_keys}...')

  pct_of_datasize = cfg.finetuning.training_settings.pct_of_datasize

  target_articles = du.load_articles(cfg.data.split.train)
  rejected_articles = du.load_articles(cfg.data.split.dpo)

  if len(target_articles) != len(rejected_articles):
    logger.error(f'Error: Length mismatch of target and rejected articles: {len(target_articles)} vs. {len(rejected_articles)}\n'
                 'Ensure that the rejected pkl file includes all articles (ie. also 88).\n'
                 'To generate new rejected pkl file: <data/dpo/annotation_results/readme.txt>')
    return None


  DR_indices = hh.parse_article_DR_indices(cfg.finetuning.training_settings.indices.train)

  if not DR_indices:
    logger.notice('No articles found.')
    return None
  
  # Expand articles and generate {instructions+target+rejected} trios: 
  # 1) instructions (including article + others) 
  # 2) targets entries
  # 3) rejected entries
  list_instructions_with_article = []
  list_target_dict = []
  list_rejected_dict = []

  # FOR-LOOPS:
  # 1) All articles in <DR_indices>
  # 2) All specified <dpo_keys> to generate DPO data
  # 3) All quotes if <dpo_keys> containes <quote_keys>
  for index in DR_indices:
    logger.notice(f'** Loading article [{index}]...')
    target_article = target_articles[index]
    rejected_article = rejected_articles[index]

    article_text = target_article['TEXT']
    for dpo_key in dpo_keys:

      if dpo_key in PER_ORG_QUOTES_keys:
        # target_entry = target_quote_text
        # rejected_entry = rejected_quote_text
        # rejected_quote_text = rejected_quote['TEXT']    

        per_json_dict     = None
        org_json_dict     = None
        target_quote_text = ""
        src_json_dict     = None
        instructions = get_instructions(key=dpo_key, 
                                        article_text = article_text, 
                                        quote_text   = target_quote_text, 
                                        per_dict     = per_json_dict, 
                                        org_dict     = org_json_dict, 
                                        src_dict     = src_json_dict)
        #target =  ', '.join(article[key])
        #target =  '\n'.join(article[key])
        #target = '[' + target + ']'


        # format: target = '"entry 1", "entry 2", "entry 3"...'
        if dpo_key in ['PER', 'ORG']: 
          target_entry = ', '.join(f'"{entry}"' for entry in target_article[dpo_key])        
          rejected_entry = ', '.join(f'"{entry}"' for entry in rejected_article[dpo_key])  
        elif dpo_key == 'QUOTES':
          quotes_count = len(target_article[dpo_key])

          if quotes_count == 0:
            target_entry = ''
            # get random amount of fake quotes (sentences): 1-3
            rejected_entry = get_random_fake_quotes(target_article['TEXT'], cfg.data.dpo[dpo_key].max_retrieval, seed)
            
          else:
            target_entry = ''
            rejected_entry = ''

            target_quotes = [f'"{target_article[dpo_key][quote_idx]["TEXT"]}"' for quote_idx in range(quotes_count)]
            target_entry = ', '.join(target_quotes)

            rejected_quotes = [f'"{rejected_article[dpo_key][quote_idx]["TEXT"]}"' for quote_idx in range(quotes_count)]
            rejected_entry = ', '.join(rejected_quotes)

        else:
          logger.error(f'ERROR: Unknown DPO key: <{dpo_key}>')
          continue  

        # convert to dictionary format expected to be output by the LLM:
        if dpo_key in ['PER', 'ORG']:
          # '{"PER": ["entry 1", "entry 2", "entry 3"...]}'
          target_entry_dict = f'{{"{dpo_key}": [{target_entry}]}}'
          rejected_entry_dict = f'{{"{dpo_key}": [{rejected_entry}]}}'
        elif dpo_key == 'QUOTES':

          # filled: '{"TEXT": "quote 1"}, .... {"TEXT": "quote n"}'
          # empty: ''
          target_entry_with_text = format_quotes_with_text_key(target_entry)
          rejected_entry_with_text = format_quotes_with_text_key(rejected_entry)

          # filled: '{"QUOTES": [{"TEXT": "quote 1"}, .... {"TEXT": "quote n"}]}'
          # empty: '{"QUOTES": []}'
          target_entry_dict = f'{{"{dpo_key}": [{target_entry_with_text}]}}'
          rejected_entry_dict = f'{{"{dpo_key}": [{rejected_entry_with_text}]}}'

        else:
          logger.error(f'ERROR: Should not be possible to reach this: Unknown article DPO key: <{dpo_key}>')
          continue

        list_instructions_with_article.append(instructions)
        list_target_dict.append(target_entry_dict)
        list_rejected_dict.append(rejected_entry_dict)

        save_dpo_data_entry_to_file(cfg, instructions, f'dpo_{index}_{dpo_key}_i.txt')
        save_dpo_data_entry_to_file(cfg, target_entry_dict, f'dpo_{index}_{dpo_key}_t.txt')
        save_dpo_data_entry_to_file(cfg, rejected_entry_dict, f'dpo_{index}_{dpo_key}_r.txt')

      elif dpo_key in SRC_GEN_FUN_EMP_ROL_keys:
        # Go through the target and rejected quotes in parallel
        for quote_idx, (target_quote, rejected_quote) in enumerate(zip(target_article['QUOTES'], rejected_article['QUOTES'])):
          target_quote_text = target_quote['TEXT']

          per_json_dict = {'PER': target_article['PER']}
          org_json_dict = {'ORG': target_article['ORG']}
          src_json_dict = {'SRC': target_quote['SRC']}

          if dpo_key == 'SRC' and (not target_article['PER'] or target_article['PER'] == [] or target_article['PER'] == ['']):
            logger.notice(f'Skip generating SRC datapoint in quote[{quote_idx}]: No person annotated.')
            continue

          instructions = get_instructions(key=dpo_key, 
                                           article_text = article_text, 
                                           quote_text   = target_quote_text, 
                                           per_dict     = per_json_dict, 
                                           org_dict     = org_json_dict, 
                                           src_dict     = src_json_dict)
          
          if dpo_key == 'SRC':
            if target_quote[dpo_key] is not None and target_quote[dpo_key] != [''] and target_quote[dpo_key] != []:
              # Assumed there is only one SRC entity, hence[0]
              target_entry = f'"{target_quote[dpo_key][0]}"'
            else:
              target_entry = ''

            if rejected_quote[dpo_key] is not None and rejected_quote[dpo_key] != [''] and rejected_quote[dpo_key] != []:
              # Assumed there is only one SRC entity, hence[0]
              rejected_entry = f'"{rejected_quote[dpo_key][0]}"'
            else:
              rejected_entry = ''

          elif dpo_key == 'GEN':
            target_entry = f'"{target_quote[dpo_key]}"'
            rejected_entry = f'"{rejected_quote[dpo_key]}"'

          elif dpo_key == 'FUN':
            target_entry = f'"{target_quote[dpo_key]}"'
            rejected_entry = f'"{rejected_quote[dpo_key]}"'

          elif dpo_key == 'ROL':
            target_entry = ', '.join(f'"{entry}"' for entry in target_quote[dpo_key])
            rejected_entry = ', '.join(f'"{entry}"' for entry in rejected_quote[dpo_key])

          elif dpo_key == 'EMP':
            target_entry = ', '.join(f'"{entry}"' for entry in target_quote[dpo_key])
            rejected_entry = ', '.join(f'"{entry}"' for entry in rejected_quote[dpo_key])


          # convert to dictionary format expected to be output by the LLM:
          # GEN,FUN are not lists, so [] are removed
          if dpo_key in ['GEN', 'FUN']:
            # example: {"GEN": "X"}
            target_entry_dict = f'{{"{dpo_key}": {target_entry}}}'
            rejected_entry_dict = f'{{"{dpo_key}": {rejected_entry}}}'
          elif dpo_key in ['SRC', 'ROL', 'EMP']:
            # example: {"SRC": ["entry 1"]}
            target_entry_dict = f'{{"{dpo_key}": [{target_entry}]}}'
            rejected_entry_dict = f'{{"{dpo_key}": [{rejected_entry}]}}'
          else:
            logger.error(f'ERROR: Should not be possible to reach this: Unknown quote DPO key: <{dpo_key}>')
            continue

          list_instructions_with_article.append(instructions)
          list_target_dict.append(target_entry_dict)
          list_rejected_dict.append(rejected_entry_dict)                   

          save_dpo_data_entry_to_file(cfg, instructions, f'dpo_{index}_{dpo_key}_i.txt')
          save_dpo_data_entry_to_file(cfg, target_entry_dict, f'dpo_{index}_{dpo_key}_t.txt')
          save_dpo_data_entry_to_file(cfg, rejected_entry_dict, f'dpo_{index}_{dpo_key}_r.txt')

      else:
        logger.error(f'ERROR: Should not be possible to reach this: Unknown DPO key: <{dpo_key}>')
        continue

  dpo_pairs = list(zip(list_instructions_with_article, list_target_dict, list_rejected_dict))

  # Reduce dataset size
  original_pairs_count = len(dpo_pairs)
  if pct_of_datasize < 100:
    reduced_size = int(len(dpo_pairs) * (pct_of_datasize / 100))
    dpo_pairs = dpo_pairs[:reduced_size]
    logger.notice(f'Dataset size reduced from {original_pairs_count} to {len(dpo_pairs)} pairs.') 
  else:
    logger.notice(f'Dataset size of {original_pairs_count} pairs.') 


  # Shuffle combined article-quote pairs
  if shuffle:
      random.shuffle(dpo_pairs)
      logger.notice('Dataset shuffled.')

  # Initialize dictionary
  dpo_dataset_dict = {
      "prompt": [],
      "chosen": [],
      "rejected": []
  }

  # Populate dictionary from dpo_pairs
  for p, c, r in dpo_pairs:
      dpo_dataset_dict["prompt"].append(p)
      dpo_dataset_dict["chosen"].append(c)
      dpo_dataset_dict["rejected"].append(r)


  # Convert dictionary to Hugging Face Dataset
  dpo_dataset = Dataset.from_dict(dpo_dataset_dict)
  
  return dpo_dataset

