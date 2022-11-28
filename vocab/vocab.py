"""
Written by KrishPro @ KP

filename: `vocab.py`
"""


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from typing import List
import pandas as pd
import os

data_dir = "/home/krish/Datasets/flickr30k"


def gather_text(csv_path: str, output_path: str):
    data: pd.DataFrame = pd.read_csv(csv_path, sep='|').dropna()

    comments: List[str] = data[' comment'].tolist()

    comments = list(map(lambda comment: comment.strip(), comments))
    
    with open(output_path, 'w') as file:
        file.write('\n'.join(comments))


def create_vocab(files: List[str], output_path: str):

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    trainer = BpeTrainer(special_tokens=["[PAD]", "[SOS]", "[EOS]", "[UNK]"])

    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train(files=files, trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    tokenizer.save(output_path)


if __name__ == "__main__":
    gather_text(os.path.join(data_dir, 'results.csv'), '.tmp')

    create_vocab(['.tmp'], 'vocab/vocab.json')