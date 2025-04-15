import torch
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.decomposition import PCA
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2_qformer_cpi import Blip2QformerCPI

def plot_heatmap(x, batch, prePath):
    f, ax = plt.subplots(figsize=(130,100))
    ax = sns.heatmap(x, annot=True,cmap = 'Blues',ax=ax,fmt ='.2f')#,vmin=0, vmax=1
    plt.title('score',fontdict={'size': 12})
    plt.savefig(prePath + '%s.png' % batch)
    plt.show()

@registry.register_model("blip2_portein_smiles_matching")
class Blip2ITMCPI(Blip2QformerCPI):
    def __init__(
        self,
        protein_model="esm",
        protein_embeddings=1280,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        freeze_protein_model =True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        type_neg = 0
    ):
        super().__init__(
            protein_model=protein_model,
            protein_embeddings=protein_embeddings,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            freeze_protein_model=freeze_protein_model,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
        )
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.index = 0

    def forward(self, samples, match_head="psm", test_path = ''):
        proteins = samples["proteins"]
        with self.maybe_autocast():
            proteins_embeds = proteins 
        proteins_atts = torch.ones(proteins_embeds.size()[:-1], dtype=torch.long).to(proteins.device)

        smiles = samples["smiles"] # print(smiles)
        t_smiles = self.tokenizer(smiles,truncation=True,padding="max_length",max_length=self.max_txt_len,return_tensors="pt").to(proteins.device)

        if match_head == "pcm":
            query_tokens = self.query_tokens.expand(proteins_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(proteins.device)
            attention_mask = torch.cat([query_atts, t_smiles.attention_mask], dim=1)

            output_itm = self.Qformer.bert(t_smiles.input_ids, query_embeds=query_tokens, attention_mask=attention_mask, encoder_hidden_states=proteins_embeds, encoder_attention_mask=proteins_atts, return_dict=True)
            itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
            itm_logit = self.itm_head(itm_embeddings)# print(itm_logit.shape) # ([100, 32, 2])
            itm_logit = itm_logit.mean(dim=1) # print(itm_logit.shape) torch.Size([100, 2])
            itm_logit = itm_logit[:, 1] 
            return itm_logit

        elif match_head == "pcc":
            query_tokens = self.query_tokens.expand(proteins_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(query_embeds=query_tokens, encoder_hidden_states=proteins_embeds, encoder_attention_mask=proteins_atts, return_dict=True)
            proteins_feats = F.normalize(self.protein_proj(query_output.last_hidden_state), dim=-1)

            t_smiles_output = self.Qformer.bert(t_smiles.input_ids,attention_mask=t_smiles.attention_mask,return_dict=True,)
            t_smiles_feat = F.normalize(self.text_proj(t_smiles_output.last_hidden_state[:, 0, :]), dim=-1)
            sims = torch.bmm(proteins_feats, t_smiles_feat.unsqueeze(-1)) # torch.Size([100, 32, 1])
            sim, _ = torch.max(sims, dim=1)
            return sim
        
