import os
from transformers import XLMTokenizer
import numpy as np
import os
import torch

# os.environ['TRANSFORMERS_CACHE'] = '/mloraw1/sfan/huggingface_cache'
# os.environ['TRANSFORMERS_CACHE'] = '/scratch/'
# export HF_DATASETS_CACHE='/scratch/hf_cache'

end_of_doc_token = '</s>'
languages = ['en', 'ar', 'zh-cn', 'zh-tw', 'nl', 'fr', 'de', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'es', 'th', 'tr', 'bg', 'ca', 'cs', 'da', 'el', 'et', 'fa', 'fi', 'he', 'hi', 'hr', 'hu', 'id', 'lt', 'lv', 'ms', 'no', 'ro', 'sk', 'sl', 'sr', 'sv', 'tl', 'uk', 'vi']


def get_wiki40b(subset='en', num_proc=40,
                return_torch=True):
    tknzr = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
    """ https://huggingface.co/datasets/wiki40b
    """
    # {
    #     "text": ...,
    #     "meta": {"url": "...", "timestamp": "...", "source": "...", "language": "...", ...},
    #     "red_pajama_subset": "common_crawl" | "c4" | "github" | "books" | "arxiv" | "wikipedia" | "stackexchange"
    # }
    WIKI_40B_PATH = "/scratch/homes/sfan/multilingual-wiki-data/notebook/"
    # WIKI_40B_PATH = os.path.join(os.path.dirname(__file__), "/scratch/homes/pagliard/curriculum/src/data/datasets/slim_redpajama")
    SUBSET_PATH = os.path.join(WIKI_40B_PATH, subset)
    train_path = os.path.join(SUBSET_PATH, f"{subset}_train.bin")
    # val_path = os.path.join(SUBSET_PATH, f"{subset}_validation.bin")
    test_path = os.path.join(SUBSET_PATH, f"{subset}_test.bin")
    
    train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
    # val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
    test_data = np.memmap(test_path, dtype=np.uint16, mode='r')
    print(f'Subset {subset}: train[{len(train_data)}] | val[{len(test_data)}]')
    if return_torch:
        train_data = torch.tensor(np.array(train_data, dtype=np.int32))
        # val_data = torch.tensor(np.array(val_data, dtype=np.int32))
        test_data = torch.tensor(np.array(test_data, dtype=np.int32))
    return {'train': train_data, 'val': test_data, 'test': test_data}
