# standard libraries
import sys
import pickle
import logging
import numpy as np

# third-party libraries
from omegaconf import DictConfig
from collections import defaultdict
import matplotlib.pyplot as plt

# local libraries
from src.data.articles_dataset import ArticlesDataset
from transformers import AutoTokenizer

# path to "type_definitions/", where QuoteCollection is defined, needed to load pickle files.
sys.path.append('src/data/dr_lib/')

logger = logging.getLogger(__name__)

def read_pickle(fullpath_pickle: str) -> object: 
    """Reads a pickle file and returns the object."""
    with open(fullpath_pickle, 'rb') as file:
        return pickle.load(file)

def load_articles_for_figures(cfg: DictConfig, dataset_selection) -> ArticlesDataset:
  logger.notice(f'\n### Dataset: {dataset_selection} ###')

  if dataset_selection == 'train':
    articles_data = read_pickle(cfg.data.split.train)
    articles = ArticlesDataset(articles_data=articles_data)    

  elif dataset_selection == 'val':
    articles_data = read_pickle(cfg.data.split.val)
    articles = ArticlesDataset(articles_data=articles_data)    

  elif dataset_selection == 'test':
    articles_data = read_pickle(cfg.data.split.test)
    articles = ArticlesDataset(articles_data=articles_data)
    
  elif dataset_selection == 'dpo':
    articles = read_pickle(cfg.data.split.dpo)

  return articles

def main(cfg: DictConfig):

  MAX_FIG_WIDTH_mm  = 147  
  GOLDEN_HW_RATIO   = 1.62
  FONT_NAME         = 'DejaVu Sans' 
  FONT_SIZE         = 10
  DPI               = 300  
  fig_path          = 'docs/source/figures/'
  fig_print         = False

  # derived values
  MAX_FIG_WIDTH_in  = MAX_FIG_WIDTH_mm / 25.4

  fig1 = True # Histogram: PER-ORG-QUOTES list lengths in articles
  fig2 = False # Histogram: token counts of articles

## FIGURE: Histogram: PER-ORG-QUOTES list lengths in articles
  if fig1:
    fig_name = 'hist_per_org_quotes.png'
    articles = load_articles_for_figures(cfg, dataset_selection='train')

## ------------------ gather data ------------------------------------------------------------------

    quote_count = 0
    quote_count_M = 0
    quote_count_F = 0
    quote_count_X = 0
    quote_count_X_in_org = 0

    # Initialize maximum lengths
    max_lengths = {
        'PER': 0,
        'ORG': 0,
        'QUOTES': 0,
        'SRC': 0,
        'ROL': 0,
        'EMP': 0
    }

    # Initialize dictionaries to store the count of each length
    length_counts = {
        'PER': defaultdict(int),
        'ORG': defaultdict(int),
        'QUOTES': defaultdict(int),
        'SRC': defaultdict(int),
        'ROL': defaultdict(int),
        'EMP': defaultdict(int)
    }

    # Iterate over each article
    for article in articles:
        max_lengths['PER'] = max(max_lengths['PER'], len(article['PER']))
        max_lengths['ORG'] = max(max_lengths['ORG'], len(article['ORG']))
        max_lengths['QUOTES'] = max(max_lengths['QUOTES'], len(article['QUOTES']))

        # Update length counts
        length_counts['PER'][len(article['PER'])] += 1
        length_counts['ORG'][len(article['ORG'])] += 1
        length_counts['QUOTES'][len(article['QUOTES'])] += 1

        # Iterate over each quote
        for quote in article['QUOTES']:
            quote_count += 1
            if quote['GEN'] == 'X':
                quote_count_X += 1
                # chech if the list in quote['SRC'] has any element in article['ORG']
                #if any(src in article['PER'] for src in quote['SRC']):                
                if quote['SRC'] == []:
                   quote_count_X_in_org += 1
            elif quote['GEN'] == 'M':
                quote_count_M += 1
            elif quote['GEN'] == 'F':
                quote_count_F += 1

            max_lengths['SRC'] = max(max_lengths['SRC'], len(quote.get('SRC', [])))
            max_lengths['ROL'] = max(max_lengths['ROL'], len(quote.get('ROL', [])))
            max_lengths['EMP'] = max(max_lengths['EMP'], len(quote.get('EMP', [])))

            # Update length counts for quote attributes
            length_counts['SRC'][len(quote.get('SRC', []))] += 1
            length_counts['ROL'][len(quote.get('ROL', []))] += 1
            length_counts['EMP'][len(quote.get('EMP', []))] += 1

    # Print maximum lengths
    for category, length in max_lengths.items():
        print(f"{category}: {length}")

    # Print the histogram of list lengths for each category
    for category, counts in length_counts.items():
        print(f"\n{category} Length Counts:")
        for length, count in sorted(counts.items()):
            print(f"Length {length}: {count} times")

    print(f'\nTotal quotes: {quote_count}')
    print(f'quotes_M: {quote_count_M}')
    print(f'quotes_F: {quote_count_F}')
    print(f'quotes_X: {quote_count_X}')
    print(f'quotes_X_in_org: {quote_count_X_in_org}')
    

## ------------------ make plot ------------------------------------------------------------------
    # Prepare data for plotting
    total_articles = len(articles)

    per_lengths = list(length_counts['PER'].keys())
    token_counts_pct = [count / total_articles * 100 for count in length_counts['PER'].values()]
    org_lengths = list(length_counts['ORG'].keys())
    org_counts_pct = [count / total_articles * 100 for count in length_counts['ORG'].values()]
    quotes_lengths = list(length_counts['QUOTES'].keys())
    quotes_counts_pct = [count / total_articles * 100 for count in length_counts['QUOTES'].values()]

    all_lengths = sorted(set(per_lengths) | set(org_lengths) | set(quotes_lengths))

    # Map lengths to indices for positions
    per_positions = [all_lengths.index(length) for length in per_lengths]
    org_positions = [all_lengths.index(length) for length in org_lengths]
    quotes_positions = [all_lengths.index(length) for length in quotes_lengths]

    # Define the width of the bars
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_in, MAX_FIG_WIDTH_in / GOLDEN_HW_RATIO * 0.7)) 

    # Plot the bars
    ax.bar(np.array(per_positions) - bar_width, token_counts_pct, width=bar_width, color='blue', label='PER')
    ax.bar(np.array(org_positions), org_counts_pct, width=bar_width, color='orange', label='ORG')
    ax.bar(np.array(quotes_positions) + bar_width, quotes_counts_pct, width=bar_width, color='green', label='QUOTES')

    # Set the x-ticks to be in the center of the bars
    ax.set_xticks(range(len(all_lengths)))
    ax.set_xticklabels(all_lengths)
        
    # Add labels and title
    ax.set_xlabel('Length of lists', fontname=FONT_NAME, fontsize=FONT_SIZE)
    ax.set_ylabel('Percentage of Articles (%)', fontname=FONT_NAME, fontsize=FONT_SIZE)
    ax.legend()    
  
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.legend().get_texts()):
      item.set_fontname(FONT_NAME)
      item.set_fontsize(FONT_SIZE)

    # Display the plot
    fig_fullpath = fig_path + fig_name

    plt.grid(axis='y')
    if fig_print:
       plt.savefig(fig_fullpath, dpi=DPI, bbox_inches='tight')
    plt.show()

## FIGURE: Histogram: token counts of articles
  if fig2:
    fig_name = 'hist_article_tokens.png'
    
    articles = load_articles_for_figures(cfg, dataset_selection='train')
  
## ------------------ gather data ------------------------------------------------------------------

    model_name = cfg.llm[cfg.llm.tag.for_finetuning].name 
    logger.notice(f'Loading tokenizer from: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length  = 0
    max_length_idx = 0
    word_count_of_longest_article = 0
    token_count_of_longest_article = 0
    cum_length = 0
    cum_words_count = 0
    cum_token_count = 0

    token_counts = []
       
    for idx, article in enumerate(articles):
      text = article['TEXT']
      art_len = len(text)
      cum_length += art_len
      art_words = len(text.split())
      cum_words_count += art_words
      tokens = tokenizer.tokenize(text)
      art_tokens = len(tokens)
      cum_token_count += art_tokens

      token_counts.append(art_tokens)

      if art_len > max_length:
        max_length = art_len
        max_length_idx = idx
        word_count_of_longest_article = art_words
        token_count_of_longest_article = art_tokens

      print(f"Article[{idx}]: {art_len} characters, {art_words} words, {art_tokens} tokens")

    print(f"Art[{max_length_idx}] (with {word_count_of_longest_article} words and {token_count_of_longest_article} tokens) - Max char length: {max_length}")
    print(f"Average char length: {round(cum_length / len(articles))}")
    print(f"Average words: {round(cum_words_count / len(articles))}")
    print(f"Average tokens per article: {round(cum_token_count / len(articles))}")
    # Art[110] (with 2199 words and 5022 tokens) - Max char length: 13682
    # Average char length: 2248
    # Average words: 378
    # Average tokens per article: 828

## ------------------ make plot ------------------------------------------------------------------
    # Prepare data for plotting
    total_articles = len(articles)

    # set bin width
    bin_width = 300
    bins = np.arange(0, max(token_counts) + bin_width, bin_width)
    hist, bin_edges = np.histogram(token_counts, bins=bins)

    # Convert counts to percentages
    hist_pct = hist / total_articles * 100

    # Prepare data for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Define the width of the bars
    bar_width = 295

    fig, ax = plt.subplots(figsize=(MAX_FIG_WIDTH_in, MAX_FIG_WIDTH_in / GOLDEN_HW_RATIO * 0.7))

    # Plot the bars
    ax.bar(bin_centers, hist_pct, width=bar_width, color='blue')

    # Set the x-ticks to be in the center of the bars
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f'{int(edge)}' for edge in bin_edges])

    ax.set_xlim(0, 3200)

    # Add labels and title
    ax.set_xlabel('Token count', fontname=FONT_NAME, fontsize=FONT_SIZE)
    ax.set_ylabel('Percentage of Articles (%)', fontname=FONT_NAME, fontsize=FONT_SIZE)        

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname(FONT_NAME)
        item.set_fontsize(FONT_SIZE)

    # Display the plot
    fig_fullpath = fig_path + fig_name

    plt.grid(axis='y')
    if fig_print:
      plt.savefig(fig_fullpath, dpi=DPI, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
  main()