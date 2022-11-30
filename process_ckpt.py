"""
Written by KrishPro @ KP

filename: `process_ckpt.py`
"""

from typing import Dict
import torch

def main(input_path: str, output_path: str):
    ckpt: Dict = torch.load(input_path)

    state = {'dims': ckpt['hyper_parameters'], 'state_dict': {k[4:]:v for k,v in ckpt['state_dict'].items()}}

    torch.save(state, output_path)

if __name__ == '__main__':
    main('/home/krish/Projects/caption-images/newOutput/lightning_logs/version_0/checkpoints/epoch=17-step=40230.ckpt', 'model.ckpt')