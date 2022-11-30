"""
Written by KrishPro @ KP

filename: `process_ckpt.py`
"""

from typing import Dict
import torch

def main(input_path: str, output_path: str):
    model_params = ['d_model', 'n_heads', 'dim_feedforward', 'num_layers', 'tgt_vocab_size', 'dropout_p', 'pad_idx']
    
    ckpt: Dict = torch.load(input_path)

    state = {'dims': {k:v for k,v in ckpt['hyper_parameters'].items() if k in model_params}, 'state_dict': {k[4:]:v for k,v in ckpt['state_dict'].items()}}

    torch.save(state, output_path)

if __name__ == '__main__':
    main('/home/krish/Projects/newOutput/lightning_logs/version_0/checkpoints/epoch=17-step=40230.ckpt', 'model.ckpt')