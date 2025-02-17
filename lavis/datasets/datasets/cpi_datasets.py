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
        # if 'batch' in data.columns:
        #     self.batch_flag = True
        #     self.batches = data['batch']

        if self.datatype == "add_neg1":
            self.NegProteins = data['seq1']
            self.NegSmiles = data['neg_canonical_smi1']
        # elif self.datatype == "add_neg2":
        #     self.NegProteins1 = data['seq1']
        #     self.NegSmiles1 = data['neg_canonical_smi1']
        #     self.NegProteins2 = data['seq2']
        #     self.NegSmiles2 = data['neg_canonical_smi2']

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, index):
        # if self.batch_flag == True:
        #     return {"proteins": self.proteins[index], "smiles": self.smiles[index], "batches": self.batches[index]} #"labels": self.labels[index], 
        
        if self.datatype == "add_neg1":
            return {
                "proteins": self.proteins[index],
                "smiles": self.smiles[index],
                # "batches": self.batches[index],
                "negproteins": self.NegProteins[index],
                "negsmiles": self.NegSmiles[index]
            }
        # elif self.datatype == "add_neg2":
        #     return {
        #         "proteins": self.proteins[index],
        #         "smiles": self.smiles[index],
        #         "batches": self.batches[index],
        #         "negproteins1": self.NegProteins1[index],
        #         "negsmiles1": self.NegSmiles1[index],
        #         "negproteins2": self.NegProteins2[index],
        #         "negsmiles2": self.NegSmiles2[index],
        #     }
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
            # elif self.datatype == "add_neg2":
            #     negProtein1.append(self.protein_processor(i['negproteins1'].upper()))
            #     negSmiles1.append(self.smiles_processor(i['negsmiles1']))
            #     negProtein2.append(self.protein_processor(i['negproteins2'].upper()))
            #     negSmiles2.append(self.smiles_processor(i['negsmiles2']))

        proteins,_ = self.protein_processor.padding(proteins_esm)
        if self.datatype == "add_neg1":
            negprotein1,_ = self.protein_processor.padding(negProtein1)
        elif self.datatype == "add_neg2":
            negprotein1,_ = self.protein_processor.padding(negProtein1)
            negprotein2,_ = self.protein_processor.padding(negProtein2)

        samples = {}
        samples['proteins'] = proteins
        samples['smiles'] = smiles
        # if self.batch_flag == True:
        #     samples['batches'] = torch.Tensor(batches).long()

        if self.datatype == "add_neg1":
            samples["negproteins"] = negprotein1
            samples["negsmiles"] = negSmiles1
        elif self.datatype == "add_neg2":
            samples["negproteins1"] = negprotein1
            samples["negsmiles1"] = negSmiles1
            samples["negproteins2"] = negprotein2
            samples["negsmiles2"] = negSmiles2
        return samples
