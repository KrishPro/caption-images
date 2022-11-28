"""
Written by KrishPro @ KP

filename: `process_data.py`
"""

from tqdm import tqdm
from tokenizers import Tokenizer
import pandas as pd
import os

def main(data_dir: str):
    data: pd.DataFrame = pd.read_csv(os.path.join(data_dir, "results.csv"), sep="|").dropna()

    tokenizer: Tokenizer = Tokenizer.from_file('vocab/vocab.json')
    
    new_data = []
    
    for image_name, _, caption in tqdm(data.iloc, total=len(data)):
        new_data.append([image_name, tokenizer.encode(caption.strip()).ids])

    data: pd.DataFrame = pd.DataFrame(new_data)
    data.to_csv('data/data.csv', index=False)

if __name__ == '__main__':
    main("/home/krish/Datasets/flickr30k")