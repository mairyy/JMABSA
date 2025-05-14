from typing import Optional, Tuple
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
# from fastNLP.modules.torch import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch
import torch.nn.functional as F
from torch import nn
from src.model.modeling_bart import (PretrainedBartModel, BartEncoder,
                                     BartDecoder, BartModel,
                                     BartClassificationHead,
                                     _make_linear_from_emb,
                                     _prepare_bart_decoder_inputs)
from transformers import BartTokenizer, BertModel

from src.model.config import MultiModalBartConfig
from src.model.modules import MultiModalBartEncoder, MultiModalBartDecoder_span, Span_loss
import numpy as np
import torch.nn as nn
import math
from src.model.GCN import GCN
import random
import math
import copy
from src.model.GAT import GAT
import numpy as np
import os
from torchcrf import CRF

from src.model.RDGNN import RDGNN, Sim_GCN

class CustomDecoder(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_labels):
        super(CustomDecoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_labels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        
        x = F.softmax(x, dim=2)
        return x

class GateModule(nn.Module):
    def __init__(self, args):
        super(GateModule, self).__init__()
        self.linear_layer = nn.Linear(768 * 2, 768)
    
    def forward(self, syn_feature, sim_feature):
        cat_feature = torch.cat((syn_feature, sim_feature), -1)
        gate = F.softmax(self.linear_layer(cat_feature), dim=-1)
        gate = gate[:, :, 0].unsqueeze(2).repeat(1, 1, syn_feature.size(-1))
        mix_feature = gate * syn_feature + (1 - gate) * sim_feature
        
        return mix_feature

class MultiModalBartModel_AESC(PretrainedBartModel):
    def build_model(self,
                    args,
                    bart_model,
                    tokenizer,
                    label_ids,
                    config,
                    decoder_type=None,
                    copy_gate=False,
                    use_encoder_mlp=False,
                    use_recur_pos=False,
                    tag_first=False):
        if args.bart_init:
            model = BartModel.from_pretrained(bart_model)
            num_tokens, _ = model.encoder.embed_tokens.weight.shape
            print('num_tokens', num_tokens)

            model.resize_token_embeddings(
                len(tokenizer.unique_no_split_tokens) + num_tokens)
            encoder = model.encoder
            decoder = model.decoder

            padding_idx = config.pad_token_id
            encoder.embed_tokens.padding_idx = padding_idx

            # if use_recur_pos:
            #     decoder.set_position_embedding(label_ids[0], tag_first)

            _tokenizer = BartTokenizer.from_pretrained(bart_model)

            for token in tokenizer.unique_no_split_tokens:
                if token[:2] == '<<':  # 特殊字符
                    index = tokenizer.convert_tokens_to_ids(
                        tokenizer._base_tokenizer.tokenize(token))
                    if len(index) > 1:
                        raise RuntimeError(f"{token} wrong split")
                    else:
                        index = index[0]
                    assert index >= num_tokens, (index, num_tokens, token)
                    indexes = _tokenizer.convert_tokens_to_ids(
                        _tokenizer.tokenize(token[2:-2]))
                    embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                    for i in indexes[1:]:
                        embed += model.decoder.embed_tokens.weight.data[i]
                    embed /= len(indexes)
                    model.decoder.embed_tokens.weight.data[index] = embed
        else:
            raise RuntimeError("error init!!!!!!!")

        multimodal_encoder = MultiModalBartEncoder(config, encoder,
                                                   tokenizer.img_feat_id,
                                                   tokenizer.cls_token_id)
        return (multimodal_encoder, decoder)


    def __init__(self, config: MultiModalBartConfig, args, bart_model,
                 tokenizer, label_ids):
        super().__init__(config)
        self.config = config
        self.mydevice=args.device
        label_ids = sorted(label_ids)
        multimodal_encoder, share_decoder = self.build_model(
            args, bart_model, tokenizer, label_ids, config)

        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        self.causal_mask = causal_mask.triu(diagonal=1)
        self.encoder = multimodal_encoder

        self.gcn_on=args.gcn_on
        self.sentinet_on=args.sentinet_on
        self.dep_mode=args.dep_mode
        self.nn_attention_mode=args.nn_attention_mode
        self.sim_gcn_dropout=args.sim_gcn_dropout
        self.gcn_proportion=args.gcn_proportion
        self.abl_mode = args.abl_mode

        # add
        self.noun_linear=nn.Linear(768,768)
        self.multi_linear=nn.Linear(768,768)
        self.att_linear=nn.Linear(768*2,1)
        self.attention=Attention(4,768,768)
        self.linear=nn.Linear(768*2,1)
        self.linear2=nn.Linear(768*2,1)

        self.alpha_linear1=nn.Linear(768,768)
        self.alpha_linear2=nn.Linear(768,768)

        self.senti_value_linear=nn.Linear(1,768)

        self.RDGNN = RDGNN(args, args.K)
        self.Sim_GCN = Sim_GCN(args, 768, 768)

        if self.abl_mode == 'gate':
            self.fusion = nn.Linear(768*2, 768)
        else:
            self.fusion = GateModule(args)

        self.bart_enabled = args.bart_enabled

        if not self.bart_enabled:
            self.num_labels = len(args.label_dict)
        self.w_l = args.w_l
        self.task = args.task

        if args.task == 'AESC':
            if not self.bart_enabled:
                self.projection = nn.Linear(768, self.num_labels)
                self.crf = CRF(num_tags=self.num_labels, batch_first=True)
            else:
                self.decoder = MultiModalBartDecoder_span(self.config,
                                                  tokenizer,
                                                  share_decoder,
                                                  tokenizer.pad_token_id,
                                                  label_ids,
                                                  self.causal_mask,
                                                  self.gcn_on,
                                                  need_tag=True,
                                                  only_sc=False)
                self.span_loss_fct = Span_loss()
        elif args.task == 'SC':
            self.classifier = nn.Sequential(nn.Linear(768, int(768/2)), nn.ReLU(), nn.Linear(int(768/2), self.num_labels))

        # self.aesc_pred = CustomDecoder(768, 512, 256, self.num_labels)
        
    def get_noun_embed(self,feature,noun_mask):
        # print(feature.shape,noun_mask.shape)
        noun_mask = noun_mask.cpu()
        noun_num = [x.numpy().tolist().count(1) for x in noun_mask]
        noun_position=[np.where(np.array(x)==1)[0].tolist() for x in noun_mask]
        for i,x in enumerate(noun_position):
            assert len(x)==noun_num[i]
        max_noun_num = max(noun_num)

        # pad
        for i,x in enumerate(noun_position):
            if len(x)<max_noun_num:
                noun_position[i]+=[0]*(max_noun_num-len(x))
        noun_position=torch.tensor(noun_position).to(feature.device)
        noun_embed=torch.zeros(feature.shape[0],max_noun_num,feature.shape[-1]).to(feature.device)
        for i in range(len(feature)):
            noun_embed[i]=torch.index_select(feature[i],dim=0,index=noun_position[i])
            noun_embed[i,noun_num[i]:]=torch.zeros(max_noun_num-noun_num[i],feature.shape[-1])
        return noun_embed

    def prepare_state(self,
                      input_ids,
                      image_features,
                      noun_mask,
                      attention_mask=None,
                      syn_dep_adj_matrix=None,
                      syn_dis_adj_matrix=None,
                      sentiment_value=None,
                      pos_ids=None,
                      raw_token_ids=None,
                      first=None,
                      aspect_mask=None,
                      labels=None):

        dict = self.encoder(input_ids=input_ids,
                            image_features=image_features,
                            attention_mask=attention_mask,
                            output_hidden_states=True,
                            return_dict=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        encoder_mask = attention_mask
        src_embed_outputs = hidden_states[0]
        syn_feature, sim_feature = None, None

        if self.abl_mode != 'syntactic':
            syn_feature = self.RDGNN(encoder_outputs, syn_dep_adj_matrix, syn_dis_adj_matrix, False, False)

        if self.abl_mode != 'align':
            noun_embed=self.get_noun_embed(encoder_outputs,noun_mask)
            encoder_outputs=self.noun_attention(encoder_outputs,noun_embed,mode=self.nn_attention_mode)

        if self.sentinet_on:
           sentiment_value=nn.ZeroPad2d(padding=(51,0,0,0))(sentiment_value)
           sentiment_value =sentiment_value.unsqueeze(-1)
           sentiment_feature=self.senti_value_linear(sentiment_value)
           encoder_outputs = encoder_outputs + sentiment_feature

        if self.abl_mode != 'semantic':
            sim_feature = self.Sim_GCN(encoder_outputs, encoder_outputs, attention_mask, noun_mask)

        if self.abl_mode == 'semantic':
            mix_feature = syn_feature * self.gcn_proportion + dict.last_hidden_state
        elif self.abl_mode == 'syntactic':
            mix_feature = sim_feature * self.gcn_proportion + dict.last_hidden_state
        else:
            if self.abl_mode != 'gate':
                mix_feature = self.fusion(syn_feature, sim_feature) * self.gcn_proportion + dict.last_hidden_state
            else:
                mix_feature = (syn_feature + sim_feature) * self.gcn_proportion + dict.last_hidden_state

        if self.task == 'AESC':
            if not self.bart_enabled:
                predict = self.projection(mix_feature)
            else:
                predict = BartState(
                    encoder_outputs,
                    encoder_mask,
                    input_ids[:,51:],  #the text features start from index 51, the front are image features.
                    first,
                    src_embed_outputs,
                    mix_feature
                )
            return predict, syn_feature, sim_feature
        elif self.task == 'SC':
            outputs = mix_feature.mean(dim=1)
            return outputs, syn_feature, sim_feature 


    def noun_attention(self,encoder_outputs,noun_embed,mode='multi-head'):
        if mode=='cat':
            multi_features_rep = encoder_outputs.unsqueeze(2).repeat(1, 1, noun_embed.shape[1], 1)
            noun_features_rep = noun_embed.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1, 1)
            noun_features_rep = self.noun_linear(noun_features_rep)
            multi_features_rep = self.multi_linear(multi_features_rep)
            concat_features = torch.tanh(torch.cat([noun_features_rep, multi_features_rep], dim=-1))
            att = torch.softmax(self.att_linear(concat_features).squeeze(-1), dim=-1)
            att_features = torch.matmul(att, noun_embed)

            alpha = torch.sigmoid(self.linear(torch.cat([self.alpha_linear1(encoder_outputs), self.alpha_linear2(att_features)], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)

            encoder_outputs = torch.mul(1-alpha, encoder_outputs) + torch.mul(alpha, att_features)

            return encoder_outputs
        elif mode=='none':
            return encoder_outputs
        elif mode=='multi-head':
            # 多头注意力
            att_features=self.attention(encoder_outputs,noun_embed,noun_embed)
            alpha = torch.sigmoid(self.linear(torch.cat([encoder_outputs, att_features], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)
            encoder_outputs = torch.mul(1 - alpha, encoder_outputs) + torch.mul(alpha, att_features)
            return encoder_outputs
        elif mode=='cos_':
            multi_features_rep = encoder_outputs.unsqueeze(1).repeat(1, noun_embed.shape[1], 1,1)
            noun_features_rep = noun_embed.unsqueeze(2).repeat(1, 1,encoder_outputs.shape[1], 1)
            att=torch.cosine_similarity(multi_features_rep,noun_features_rep,dim=-1)
            att=att.max(1)[1]
            att_features=torch.zeros(encoder_outputs.shape).to(self.mydevice)
            for i in range(noun_embed.shape[0]):
                att_features[i]=torch.index_select(noun_embed[i],0,att[i])

            alpha = torch.sigmoid(self.linear(torch.cat([encoder_outputs, att_features], dim=-1)))
            alpha = alpha.repeat(1, 1, 768)
            encoder_outputs = torch.mul(alpha, encoder_outputs) + torch.mul(1-alpha, att_features)
            return encoder_outputs

    def contrastive_loss(self, feats_1, feats_2, temp=0.07):
        feats_1 = feats_1.mean(dim=1)
        feats_2 = feats_2.mean(dim=1 )
        sim_matrix = torch.matmul(feats_1, feats_2.T)

        i_logsoftmax = nn.functional.log_softmax(sim_matrix / temp, dim=1)
        j_logsoftmax = nn.functional.log_softmax(sim_matrix.T / temp, dim=1)

        i_diag = torch.diag(i_logsoftmax)
        loss_i = i_diag.mean()

        j_diag = torch.diag(j_logsoftmax)
        loss_j = j_diag.mean()

        con_loss = - (loss_i + loss_j) / 2 

        return con_loss
    
    def forward(
            self,
            input_ids,
            image_features,
            sentiment_value,
            noun_mask,
            attention_mask=None,
            syn_dep_adj_matrix=None,
            syn_dis_adj_matrix=None,
            aesc_infos=None,
            aspect_mask=None,
            labels=None,
            encoder_outputs: Optional[Tuple] = None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        if self.task == 'AESC':
            logits, syn_feature, sim_feature = self.prepare_state(input_ids=input_ids,
                    image_features=image_features,
                    noun_mask=noun_mask,
                    attention_mask=attention_mask,
                    syn_dep_adj_matrix=syn_dep_adj_matrix,
                    syn_dis_adj_matrix=syn_dis_adj_matrix,
                    sentiment_value=sentiment_value,
                    aspect_mask=aspect_mask,
                    labels=labels)

            if not self.bart_enabled:
                if labels != None:
                    log_likelihood = self.crf(logits[:, 51:, :], labels, mask=attention_mask[:, 51:].byte(), reduction='sum')

                    loss_crf = -log_likelihood
                    loss_con = self.contrastive_loss(syn_feature[:, 51:, :], sim_feature[:, 51:, :])

                    # print(loss_crf, loss_con, 1-self.w_l)
                    loss = self.w_l * loss_crf + (1-self.w_l) * loss_con
                    # loss = loss_crf
                    return loss
                else:
                    return self.crf.decode(logits[:, 51:, :], mask=attention_mask[:, 51:].byte())
            else:
                spans, span_mask = [
                    aesc_infos['labels'].to(input_ids.device),
                    aesc_infos['masks'].to(input_ids.device)
                ]

                logits = self.decoder(spans, logits, sentiment_value)
                # print(spans, spans.shape, logits, logits.shape)
                loss = self.span_loss_fct(spans[:, 1:], logits, span_mask[:, 1:])
                return loss

        elif self.task == 'SC':
            logits, syn_feature, sim_feature = self.prepare_state(input_ids=input_ids,
                    image_features=image_features,
                    noun_mask=noun_mask,
                    attention_mask=attention_mask,
                    syn_dep_adj_matrix=syn_dep_adj_matrix,
                    syn_dis_adj_matrix=syn_dis_adj_matrix,
                    sentiment_value=sentiment_value,
                    aspect_mask=aspect_mask,
                    labels=labels)
            
            predict = self.classifier(logits)
            # print(predict.shape)
            return predict  


class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first,
                 src_embed_outputs,mix_feature):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs
        self.mix_feature=mix_feature

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs,
                                                     indices)

        if self.mix_feature is not None:
            self.mix_feature = self._reorder_state(self.mix_feature,
                                                         indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(
                                layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new




class Attention(nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(Attention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, q,k,v):
        key = self.key_layer(k)
        query = self.query_layer(q)
        value = self.value_layer(v)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context
    
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):  # x: [B, S, D]
        weights = self.attn(x)              # [B, S, 1]
        weights = torch.softmax(weights, dim=1)
        output = (x * weights).sum(dim=1)   # [B, D]
        return output