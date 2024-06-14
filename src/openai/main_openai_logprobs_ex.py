from openai import OpenAI
import json
from math import exp
import numpy as np
from IPython.display import display, HTML
import os

client = OpenAI()

## Get completion from OpenAI API
def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-3.5-turbo-0125",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion

CLASSIFICATION_PROMPT = """You will be given a headline of a news article.
Classify the article into one of the following categories: Technology, Politics, Sports, and Art.
Return only the name of the category, and nothing else.
MAKE SURE your output is one of the four categories stated.
Article headline: {headline}"""


headlines = [
    "Tech Giant Unveils Latest Smartphone Model with Advanced Photo-Editing Features.",
    "Local Mayor Launches Initiative to Enhance Urban Public Transport.",
    "Tennis Champion Showcases Hidden Talents in Symphony Orchestra Debut",
]

## Without logprobs
if False:
  for headline in headlines:
      print(f"\nHeadline: {headline}")
      API_RESPONSE = get_completion(
          [{"role": "user", "content": CLASSIFICATION_PROMPT.format(headline=headline)}],
      )
      print(f"Category: {API_RESPONSE.choices[0].message.content}\n")
      print(f"Prompt T: {API_RESPONSE.usage.prompt_tokens}\n")
      print(f"Compl. T: {API_RESPONSE.usage.completion_tokens}\n")


## With logprobs
html_file_path = "src/openai/output.html" 
html_content = ""  # Initialize HTML content outside the loop

for headline in headlines:
    print(f"\nHeadline: {headline}")
    API_RESPONSE = get_completion(
        [{"role": "user", "content": CLASSIFICATION_PROMPT.format(headline=headline)}],
        logprobs=True,
        top_logprobs=2,
    )
    print(f"Prompt T: {API_RESPONSE.usage.prompt_tokens}\n")
    print(f"Compl. T: {API_RESPONSE.usage.completion_tokens}\n")
    top_two_logprobs = API_RESPONSE.choices[0].logprobs.content[0].top_logprobs
    for i, logprob in enumerate(top_two_logprobs, start=1):
        html_content += (
            f"<span style='color: cyan'>Output token {i}:</span> {logprob.token}, "
            f"<span style='color: darkorange'>logprobs:</span> {logprob.logprob}, "
            f"<span style='color: magenta'>linear probability:</span> {np.round(np.exp(logprob.logprob)*100,2)}%<br>"
        )

# Write the HTML content to a file
with open(html_file_path, "a") as html_file:
    html_file.write(html_content)
    
    print(f'HTML content saved to {html_file_path}\n')
