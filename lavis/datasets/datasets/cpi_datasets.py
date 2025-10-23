import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset

import json
import copy
import pandas as pd
import torch

class CPIDataset(BaseDataset):
    def __init__(self, protein_processor, smiles_processor, root, datatype = "others"):
        """
        protein_processor (string): protein processor
        smiles_processor (string): smiles processor
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_paths (string): Root directory of images (e.g. coco/images/)
        """
        self.datatype = datatype
        self.protein_processor = protein_processor
        self.smiles_processor = smiles_processor
        self.root = root
        
        data = pd.read_parquet(self.root)
        self.proteins = data['seq']
        self.smiles = data['canonical_smi']
        self.batch_flag = False

        if self.datatype == "add_neg1":
            self.NegProteins = data['seq1']
            self.NegSmiles = data['neg_canonical_smi1']

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        if self.datatype == "add_neg1":
            return {
                "proteins": self.proteins[index],
                "smiles": self.smiles[index],
                "negproteins": self.NegProteins[index],
                "negsmiles": self.NegSmiles[index]
            }
        else:
            return {"proteins": self.proteins[index], "smiles": self.smiles[index]} 


    def collater(self, samples): # esm type
        proteins_esm, smiles, batches, negProtein1, negSmiles1, negProtein2, negSmiles2 = [], [], [], [], [], [], []

        for i in samples:
            proteins_esm.append(self.protein_processor(i['proteins'].upper()))
            smiles.append(self.smiles_processor(i['smiles']))
            # if self.batch_flag == True:
            #     batches.append(i['batches'])
            if self.datatype == "add_neg1":
                negProtein1.append(self.protein_processor(i['negproteins'].upper()))
                negSmiles1.append(self.smiles_processor(i['negsmiles']))
                
        proteins,_ = self.protein_processor.padding(proteins_esm)
        if self.datatype == "add_neg1":
            negprotein1,_ = self.protein_processor.padding(negProtein1)
            
        samples = {}
        samples['proteins'] = proteins
        samples['smiles'] = smiles
        # if self.batch_flag == True:
        #     samples['batches'] = torch.Tensor(batches).long()

        if self.datatype == "add_neg1":
            samples["negproteins"] = negprotein1
            samples["negsmiles"] = negSmiles1

        return samples
