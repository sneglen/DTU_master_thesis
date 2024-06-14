from torch.utils.data import Dataset
import src.data.data_utils as du
import json

QUOTE_CATEGORIES = ('GEN', 'FUN', 'SRC', 'ROL', 'EMP')
ARTICLE_CATEGORIES = ('PER', 'ORG', 'TEXT', *QUOTE_CATEGORIES)

class ArticlesDataset(Dataset):
    def __init__(self, articles_data):
        self.articles_data = articles_data
        # Label map relevant for LLM fine-tuning (NER+RE tagging task)      
        self.label_map = {'PER': 0, 'ORG': 1}

        # Gender map: from Danish to English
        self.gender_da_to_en = {'M': 'M', 
                                'K': 'F', 
                                'X': 'X'}
        
        # Function map: from Danish to English
        self.function_da_to_en = {
            'Ekspert': 'Expert', 
            'Case': 'Case', 
            'Politiker': 'Politician', 
            'DR-kilde': 'DR source',
            'Interesseorganisation': 'Interest organization', 
            'Professionsekspert': 'Professional expert',
            'Myndighed': 'Authority', 
            'Andet': 'Other'
        }


    def __len__(self):
        # Get number of articles based on 'text' field
        return len(self.articles_data['text'])
    
    def _construct_article_record(self, idx):
        # LOCations and classifications omitted for simplicity.
        # (not relevant in the current project)
        article_record = {
            'headline': self.articles_data['overskrift'].iloc[idx],
            'url': self.articles_data['url'].iloc[idx],
            'TEXT': self.articles_data['text'].iloc[idx],
            'PER': self.articles_data['PER'].iloc[idx],
            'ORG': self.articles_data['ORG'].iloc[idx],
            'QUOTES': self.articles_data['citater'].iloc[idx],
        }
        return article_record


    def __getitem__(self, idx):
        """Returns the article data as a dictionary compatible with JSON serialization."""
        article_record = self._construct_article_record(idx)

        # Process 'QUOTES' if not empty
        if article_record['QUOTES'].citater:
            # Adjusted to handle Quote objects
            article_record['QUOTES'] = [{
                'TEXT': quote.text,
                'GEN': self.gender_da_to_en.get(quote.Køn.value, 'X'),            # 'X' if not found
                'FUN': self.function_da_to_en.get(quote.Funktion.value, 'Other'), # 'Other' if not found
                'SRC': list(quote.SRC) if hasattr(quote, 'SRC') and quote.SRC else [],
                'ROL': list(quote.ROL) if hasattr(quote, 'ROL') and quote.ROL else [],
                'EMP': list(quote.EMP) if hasattr(quote, 'EMP') and quote.EMP else [],
            } for quote in article_record['QUOTES'].citater]
        else:
            article_record['QUOTES'] = []

        return du.convert_sets_to_lists_and_sort(article_record)
    
    
    def print_raw_article(self, idx):
        """Prints the article data as is."""      
        # Ensure the index is within bounds
        if idx >= len(self.articles_data['TEXT']):
            print("Index out of bounds.")
            return

        article_record = self._construct_article_record(idx)

        # Print main entities except 'QUOTES'
        print("Article Data:")
        for key, value in article_record.items():
            if key != 'QUOTES':
                print(f"{key}: {value}\n")

        # Now print quotes 
        print("\nQUOTES:")
        if article_record['QUOTES']:
            for idx, quote in enumerate(article_record['QUOTES'].citater):
                print(f"quote({idx})[")
                for key, value in quote:
                    print(f"'{key}': '{value}',")
                print("]")
        else:
            print("No quotes available.")  


    def article_to_readable_json(self, idx):
        """Converts an article to a readable JSON string (sets are converted to sorted lists).

        Usage:
        json_str = articles.article_to_readable_json(article_index)

        where articles is of type <ArticlesDataset>
        """
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index is out of bounds.")

        article_record = self.__getitem__(idx)

        article_record_converted = du.convert_sets_to_lists_and_sort(article_record)
        # add indent=2  (JSON string more readable)
        json_str = json.dumps(article_record_converted, ensure_ascii=False, indent=2)

        return json_str     
