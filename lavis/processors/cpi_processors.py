import os
import torch
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from dgl.data.utils import save_graphs
from dgllife.utils import smiles_to_bigraph

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
import esm

SPECIAL_TOKENS_DICT = {'bos_token': "<bos>", 'eos_token': "<eos>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>"}

def load_json(file_path):
    assert file_path.split('.')[-1] == 'json'
    with open(file_path,'r') as file:
        data = json.load(file)
    return data

@registry.register_processor("proteins_processor")
class ProteinsFeatureProcessor(BaseProcessor):
    def __init__(self): 
        self.seqIndexDict = load_json('SeqIndex.json') 

    def get_attention_mask(self, seq):
        return torch.sum(seq != 1, dim=2) != 0
    
    def padding(self, ds):
        temp = []
        if len(ds[0].shape) == 3:
            ds = [torch.squeeze(d,0) for d in ds]
        nums = [d.shape[0] for d in list(ds)]
        max_len = max(nums)
        for d in ds:
            if d.shape[0] < max_len:
                zero = torch.zeros(max_len - d.shape[0], d.shape[1])
                data = torch.cat((d, zero), dim=0) # row padding
            else:
                data = d
            temp.append(data)
        t = torch.stack(temp)
        return t, nums

    def __call__(self, protein): 
        seqIndex = self.seqIndexDict[protein]
        representations = torch.load('proteins_emb_esm2/proteins%s.pt'% seqIndex) 
        rep = torch.Tensor(representations['representations'][33])
        rep = rep.unsqueeze(0)
        return rep

@registry.register_processor("smiles_processor")
class SmilesFeatureProcessor(BaseProcessor):
    def __init__(self): #, visual_ft, audio_ft
        self.num_atom_feat = 34

    def get_attention_mask(self, seq):
        return torch.sum(seq != 1, dim=2) != 0

    def __call__(self, smiles): # 传入SMILES，no embedding
        return smiles
