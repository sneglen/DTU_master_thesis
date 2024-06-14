# standard libraries

# third-party libraries

# local libraries
from tests.load_hydra_for_testing import load_hydra_config
from src.finetune import finetune_data_utils as ft_du

## Execution hint (from project root):
#  pytest -s tests/finetune

def test_load_data_in_chatML_format(logger):

  cfg = load_hydra_config()
  cfg.llm.tag.for_finetuning = 'munin'
  cfg.finetuning.training_settings.indices.train = ['0','1','2']
  cfg.finetuning.training_settings.pct_of_datasize = 100
  cfg.finetuning.training_settings.shuffle = False  

  dataset = ft_du.load_data_in_chatML_format(cfg, key='QUOTES', selected_split='train')
  ### print(dataset)
  # Dataset: <train>. Number of article-quote pairs: 6
  # Dataset shuffled.
  # Dataset({
  #     features: ['question', 'response', 'system_prompt', 'messages'],
  #     num_rows: 6
  # })
  #
  ### print(dataset[1]) (instructions and quote capped to [:20])
  # {'question': 'You are a helpful as',                                      <---- These are the instructions with the article
  # 'response': 'Forhåbentlig bliver ',                                       <---- This is the quote to be generated/predicted
  # 'system_prompt': '', 
  # 'messages': [ {'content': '',                     'role': 'system'}, 
  #               {'content': 'You are a helpful as', 'role': 'user'},        <---- These are the instructions with the article 
  #               {'content': 'Forhåbentlig bliver ', 'role': 'assistant'}]}  <---- This is the quote to be generated/predicted

## CHECK DATASET LENGTH
  EXP_dataset_len = 6
  assert len(dataset) == EXP_dataset_len, f"Actual length of train_dataset: {len(dataset)}, does not match expected length: {EXP_dataset_len} (ensure hydra cfg.data.train.indices: ['0-2'])"

  
## CHECK CONTENTS OF THE SECOND DATA POINT (system, user, assistant)
  messages = dataset[1]['messages'] 

  EXP_system_content    = "" 
  EXP_user_content      = "Read the following D"
  EXP_assistant_content = "Forhåbentlig bliver " 

  # Each item in 'messages' is a list of dictionaries with roles and contents
  system_content    = [msg['content'] for msg in messages if msg['role'] == 'system'][0][:20]
  user_content      = [msg['content'] for msg in messages if msg['role'] == 'user'][0][:20]
  assistant_content = [msg['content'] for msg in messages if msg['role'] == 'assistant'][0][:20]

  assert system_content     == EXP_system_content,    f"Expected system content does not match:     {system_content}"
  assert user_content       == EXP_user_content,      f"Expected user content does not match:       {user_content}"
  assert assistant_content  == EXP_assistant_content, f"Expected assistant content does not match:  {assistant_content}"