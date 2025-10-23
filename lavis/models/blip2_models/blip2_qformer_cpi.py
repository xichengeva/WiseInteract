import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2_cpi import (
    Blip2BaseCPI,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip2_models.mask_ratio_tensor import smiles_to_token_tensors
from lavis.models.blip2_models.blip_outputs import BlipOutput, BlipOutputFeatures

@registry.register_model("blip2_cpi")
@registry.register_model("blip2_feature_extractor_cpi")
class Blip2QformerCPI(Blip2BaseCPI):
    PRETRAINED_MODEL_CONFIG_DICT = {"pretrain": "configs/models/blip2/blip2_pretrain.yaml"}
    def __init__(
        self,
        protein_model="esm",
        protein_embeddings=1280,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        freeze_protein_model=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        type_neg = 0,
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        logging.info("constructing qformer...")
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, protein_embeddings, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.protein_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.pcm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        self.type_neg = type_neg

    def pcm_labels(self, proteins, smiles):
        proteins_embeds = proteins 
        proteins_atts = torch.ones(proteins_embeds.size()[:-1], dtype=torch.long).to(proteins.device)

        smiles = smiles
        t_smiles = self.tokenizer(smiles,truncation=True,padding="max_length",max_length=self.max_txt_len,return_tensors="pt").to(proteins.device)

        query_tokens = self.query_tokens.expand(proteins_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(proteins.device)###
        attention_mask = torch.cat([query_atts, t_smiles.attention_mask], dim=1)
        output_pcm = self.Qformer.bert(
            t_smiles.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=proteins_embeds,
            encoder_attention_mask=proteins_atts,
            return_dict=True,
        )
        pcm_embeddings = output_pcm.last_hidden_state[:, : query_tokens.size(1), :]
        pcm_logit = self.pcm_head(pcm_embeddings) 
        logits = pcm_logit.mean(dim=1) 
        return logits
    
    def getSpFromBatch(self, protein, smiles):
        protein_embeds = protein 
        protein_atts = torch.ones(protein_embeds.size()[:-1], dtype=torch.long).to(protein_embeds.device)
        query_tokens = self.query_tokens.expand(protein_embeds.shape[0], -1, -1) 
        query_output = self.Qformer.bert(query_embeds=query_tokens,encoder_hidden_states=protein_embeds,encoder_attention_mask=protein_atts,use_cache=True,return_dict=True)

        protein_feats = F.normalize(self.protein_proj(query_output.last_hidden_state), dim=-1)
        text_tokens = self.tokenizer(smiles,padding="max_length",truncation=True,max_length=self.max_txt_len,return_tensors="pt").to(protein_embeds.device)
        text_output = self.Qformer.bert(text_tokens.input_ids, attention_mask=text_tokens.attention_mask, return_dict=True)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        ###============== protein-smiles Contrastive ===================###
        protein_feats_all = concat_all_gather(protein_feats)  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(protein_feats.unsqueeze(1), text_feat_all.unsqueeze(-1))
        sim_q2t = sim_q2t.squeeze() #[batch_size, batch_size*num_gpu, num_query_tokens]
        sim_p2c, _ = sim_q2t.max(-1)
        sim_p2c = sim_p2c / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(text_feat.unsqueeze(1).unsqueeze(1), protein_feats_all.permute(0, 2, 1)).squeeze()
        sim_c2p, _ = sim_t2q.max(-1)
        sim_c2p = sim_c2p / self.temp
        return sim_p2c, sim_c2p

    def forward(self, samples):
        ###============== contrastive learning ===================###
        protein_embeds = samples['proteins'] # torch.Size([100, 1016, 1280])
        bs = protein_embeds.size(0)
        sim_p2c_part1, sim_c2p_part1 = self.getSpFromBatch(samples['proteins'], samples["smiles"])  

        rank = dist.get_rank()
        targets_part1 = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(protein_embeds.device) 
        loss_pcc_part1 = (
            F.cross_entropy(sim_p2c_part1, targets_part1, label_smoothing=0.1)
            + F.cross_entropy(sim_c2p_part1, targets_part1, label_smoothing=0.1)
        ) / 2
        if self.type_neg == 0:
            loss_pcc_part2 = 0
        elif self.type_neg == 1:
            sim_p2c_neg, sim_c2p_neg = self.getSpFromBatch(samples["negproteins"], samples["negsmiles"])
            sim_p2c_true = torch.cat([torch.diag(sim_p2c_part1),torch.diag(sim_p2c_neg)], dim=0)
            sim_c2p_true = torch.cat([torch.diag(sim_c2p_part1),torch.diag(sim_c2p_neg)], dim=0)
            targets_true = torch.zeros(bs*(self.type_neg + 1)).to(protein_embeds.device)
            targets_true[:bs] = 1
            loss_pcc_part2 = (
                F.binary_cross_entropy_with_logits(sim_p2c_true, targets_true) 
                + F.binary_cross_entropy_with_logits(sim_c2p_true, targets_true) 
            ) / 2
        loss_pcc = loss_pcc_part1 + loss_pcc_part2

        ###============== Matching ===================###
        if self.type_neg == 1:
            pcm0_logits = self.pcm_labels(samples['proteins'], samples['smiles'])
            pcm1_logits = self.pcm_labels(samples['negproteins'], samples['negsmiles'])
            logits = torch.cat([pcm0_logits, pcm1_logits], dim=0).to(protein_embeds.device)
            pcm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(bs, dtype=torch.long)], dim=0).to(protein_embeds.device)
            loss_pcm = F.cross_entropy(logits, pcm_labels)
        else:
            smiles = samples['smiles']
            mol_tokens = self.tokenizer(smiles,padding="max_length",truncation=True,max_length=self.max_txt_len,return_tensors="pt").to(protein_embeds.device)

            mol_input_ids_world = concat_all_gather(mol_tokens.input_ids)
            mol_attention_mask_world = concat_all_gather(mol_tokens.attention_mask)
            protein_embeds_world = all_gather_with_grad(protein_embeds)
            
            with torch.no_grad():   
                sim_c2p_part1[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
                sim_p2c_part1[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000) 
                weights_c2p = F.softmax(sim_c2p_part1, dim=1)
                weights_p2c = F.softmax(sim_p2c_part1, dim=1)

            # select a negative protein for each molecules
            protein_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_c2p[b], 1).item()
                protein_embeds_neg.append(protein_embeds_world[neg_idx])
            protein_embeds_neg = torch.stack(protein_embeds_neg, dim=0)

            # select a negative molecule for each protein
            mol_ids_neg = []
            mol_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_p2c[b], 1).item()
                mol_ids_neg.append(mol_input_ids_world[neg_idx])
                mol_atts_neg.append(mol_attention_mask_world[neg_idx])
            mol_ids_neg = torch.stack(mol_ids_neg, dim=0)
            mol_atts_neg = torch.stack(mol_atts_neg, dim=0)
            mol_ids_all = torch.cat([mol_tokens.input_ids, mol_tokens.input_ids, mol_ids_neg], dim=0)  # pos, pos, neg
            mol_atts_all = torch.cat([mol_tokens.attention_mask, mol_tokens.attention_mask, mol_atts_neg], dim=0)

            query_tokens_pcm = self.query_tokens.expand(mol_ids_all.shape[0], -1, -1)
            query_atts_pcm = torch.ones(query_tokens_pcm.size()[:-1], dtype=torch.long).to(protein_embeds.device)
            attention_mask_all = torch.cat([query_atts_pcm, mol_atts_all], dim=1)

            protein_embeds_all = torch.cat([protein_embeds, protein_embeds_neg, protein_embeds], dim=0)  # pos, neg, pos
            protein_atts_all = torch.ones(protein_embeds_all.size()[:-1], dtype=torch.long).to(protein_embeds.device)

            output_itm = self.Qformer.bert(
                mol_ids_all,
                query_embeds=query_tokens_pcm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=protein_embeds_all,
                encoder_attention_mask=protein_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_pcm.size(1), :]
            vl_output = self.pcm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            pcm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(protein_embeds.device)
            loss_pcm = F.cross_entropy(logits, pcm_labels)

        ##================= recoverying ========================##
        smiles = samples["smiles"]
        protein_atts = torch.ones(protein_embeds.size()[:-1], dtype=torch.long).to(protein_embeds.device) 
        query_tokens = self.query_tokens.expand(protein_embeds.shape[0], -1, -1) 
        query_output = self.Qformer.bert(query_embeds=query_tokens,encoder_hidden_states=protein_embeds,encoder_attention_mask=protein_atts,use_cache=True,return_dict=True)

        mol_tokens = smiles_to_token_tensors(smiles_list=smiles,tokenizer=self.tokenizer,max_length=self.max_txt_len)        
        decoder_input_ids = mol_tokens["masked_input_ids"].to(protein_embeds.device)
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = mol_tokens["labels"].to(protein_embeds.device)

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(protein_embeds.device)
        attention_mask = torch.cat([query_atts, mol_tokens["attention_mask"].to(protein_embeds.device)], dim=1) 
        mlm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
            is_decoder=True
        )
        loss_mlm = mlm_output.loss

        return BlipOutput(
            loss = loss_pcc + loss_pcm + loss_mlm,
            loss_pcc = loss_pcc,
            loss_pcm = loss_pcm,
            loss_mlm = loss_mlm,
        )

    @classmethod
    def from_config(cls, cfg):
        protein_embedding_size = cfg.get("protein_embedding_size", 1280)
        num_query_token = cfg.get("num_query_token", 0)
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        freeze_protein_model = cfg.get("freeze_protein", True)
        max_txt_len = cfg.get("max_txt_len", 32)
        type_neg = cfg.get("type_neg", 0)
        model = cls(
            protein_model="esm",
            protein_embeddings=protein_embedding_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            freeze_protein_model=freeze_protein_model,
            num_query_token=num_query_token, #
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            type_neg = type_neg,
        )
        model.load_checkpoint_from_config(cfg)
        return model

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        vit_num_layers = 33
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
            
        optim_params = list(parameter_group_vars.values())
        return optim_params