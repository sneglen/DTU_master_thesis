import yaml
import sys
import torch

def main(config_path='conf/llm/default.yaml'):
  # Load the configuration file
  with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

  model_tag = config['tag']['for_inference']

  model_info = config[model_tag]

  if not model_info:
    print(f"Error: Model '{model_tag}' not found in configuration.", file=sys.stderr)
    sys.exit(1)

  model_dir = model_info['dir']
  mfs = model_info.get('mem-fraction-static', 0.95)

  # get GPU count
  try:
    gpu_count = torch.cuda.device_count()
  except Exception as e:
    raise Exception(f"An error occurred while getting gpu count: {e}")
    gpu_count = 1      
    
  # separate parameters with a space.
  print(f"{model_dir} {mfs} {gpu_count}")

if __name__ == "__main__":
  main()
