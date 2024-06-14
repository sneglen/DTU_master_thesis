## python src/finetune/data_investigation.py

# For loading the finetuning datasets
from datasets import load_dataset

RANDOM_SEED = 42

print('--------- BEGIN ---------')

dataset = load_dataset("kobprof/skolegpt-instruct", split="train")
print(f"Number of samples in dataset: {len(dataset):,}")

n_samples = 2
dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(n_samples))

## print dataset
for i, sample in enumerate(dataset):    
  print(f"\n****Sample {i+1}:\n {sample}")
  print(f"Sample keys: {sample.keys()}")

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


# Sample keys: dict_keys(['id', 'system_prompt', 'question', 'response', 'source'])
dataset = dataset.map(create_conversation, batched=False)
# Sample keys: dict_keys(['id', 'system_prompt', 'question', 'response', 'source', 'messages'])
#
# 'messages': [
# {'content': '', 													'role': 'system'}, 
# {'content': "Giv opgavedefinitionen...",  'role': 'user'}, 
# {'content': 'sKCCsdZmPFFm', 			        'role': 'assistant'}
# ]}


## print dataset
print('printing dataset...')
for i, sample in enumerate(dataset):    
  print(f"\n****Sample {i+1}:\n {sample}")
  print(f"Sample keys: {sample.keys()}")
  if 'messages' in sample:
    for j, message in enumerate(sample['messages']):
      print(f"Message {j+1} keys: {message.keys()}")

print('--------- END ---------')

