import warnings
import numpy as np
import torch
from itertools import chain

class Collator:
    """
    The collator for all types of dataset.
    Remember to add the corresponding collation code after adding a new type of task.
    """
    def __init__(self, args,
                 tokenizer,
                 has_label=True,
                 aesc_enabled=False,
                 text_only=False,
                 trc_enabled=False,
                 crf_on=True,
                 lm_max_len=30,
                 max_img_num=49,
                 max_span_len=20):
        """
        :param tokenizer: ConditionTokenizer
        :param mlm_enabled: bool, if use mlm for language modeling. False for autoregressive modeling
        :param mrm_enabled: bool, if use mrm
        :param rp_enabled: bool, if use relation prediction (VG)
        :param ap_enabled: bool, if use attribute prediction (VG)
        :param mlm_probability: float, probability to mask the tokens
        :param mrm_probability: float, probability to mask the regions
        """
        self.args = args
        self._tokenizer = tokenizer
        self._has_label = has_label
        self._aesc_enabled = aesc_enabled
        self._trc_enabled=trc_enabled
        self._lm_max_len = lm_max_len
        self._max_img_num = max_img_num
        self._max_span_len = max_span_len
        self.text_only = text_only
        self.crf_on = crf_on
        if not has_label:
            raise ValueError(
                'mlm_enabled can not be true while has_label is false. MLM need labels.'
            )

    def _clip_text(self, text, length):
        tokenized = []
        for i, word in enumerate(text.split()):
            if i == 0:
                bpes = self._tokenizer._base_tokenizer.tokenize(word)
            else:
                bpes = self._tokenizer._base_tokenizer.tokenize(
                    word, add_prefix_space=True)
            bpes = self._tokenizer._base_tokenizer.convert_tokens_to_ids(bpes)
            tokenized.append(bpes)
        _tokenized = list(chain(*tokenized))
        return self._tokenizer.get_base_tokenizer().decode(_tokenized[:length])

    def __call__(self, batch):
        batch = [entry for entry in batch if entry is not None]
        image_features = [x['img_feat'] for x in batch]

        img_num = [self._max_img_num]*len(image_features)
        # img_num = None

        target = [x['sentence'] for x in batch]
        sentence = list(target)
        syn_dis_adj = [x['syn_dis_adj'] for x in batch]
        syn_dep_adj = [x['syn_dep_adj'] for x in batch]

        if not self._aesc_enabled:
            labels = [x['labels'] for x in batch]
        else:
            labels = None

        if self.args.task == 'AESC':
            encoded_conditions = self._tokenizer.encode_condition(
                img_num=img_num, sentence=sentence, text_only=self.text_only, syn_dis_adj=syn_dis_adj, syn_dep_adj=syn_dep_adj, _labels=labels)
        elif self.args.task == 'SC':
            aspects = [x['aspects'] for x in batch]
            encoded_conditions = self._tokenizer.encode_condition(
                img_num=img_num, sentence=sentence, text_only=self.text_only, syn_dis_adj=syn_dis_adj, syn_dep_adj=syn_dep_adj, aspects=aspects)
            
        input_ids = encoded_conditions['input_ids']
        output = {}

        output['input_ids'] = input_ids
        output['attention_mask'] = encoded_conditions['attention_mask']
        output['image_features'] = image_features
        output['input_ids'] = input_ids
        output['sentiment_value']=encoded_conditions['sentiment_value']

        output['noun_mask']=encoded_conditions['noun_mask']
        output['syn_dep_matrix'] = encoded_conditions['syn_dep_adj_matrix']
        output['syn_dis_matrix'] = encoded_conditions['syn_dis_adj_matrix']

        if not self._aesc_enabled:
            output['aspect_mask'] = encoded_conditions['aspect_mask']
        else:
            output['aspect_mask'] = None

        if self.args.task == 'AESC':
            output['task'] = 'AESC'
            if self._aesc_enabled:
                output['AESC'] = self._tokenizer.encode_aesc(
                    target, [x['aesc_spans'] for x in batch],  # target = [x['sentence'] for x in batch]
                    self._max_span_len)
            else:
                output['AESC'] = encoded_conditions['labels']
        elif self.args.task == 'SC':
            output['task'] = 'SC'
            output['SC'] = torch.tensor(labels)
        # output['AESC'] = self._tokenizer.encode_aesc(
        #                                             target, [x['aesc_spans'] for x in batch],  # target = [x['sentence'] for x in batch]
        #                                             self._max_span_len)
            

        output['image_id'] = [x['image_id'] for x in batch]
        return output