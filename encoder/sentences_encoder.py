import logging
import random

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SentencesEncoder(nn.Module):
    def __init__(self, pretrain_path, batch_size,num_nodes, blank_padding=True, mask_entity=False):
        super().__init__()
        self.blank_padding = blank_padding
        self.hidden_size = 768
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.mask_entity = mask_entity
        self.max_length = 256
        
        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = BertModel.from_pretrained("./bert_pretrained")
        self.tokenizer = BertTokenizer.from_pretrained("./bert_pretrained")
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, token, att_mask):
        hidden = self.bert(token, attention_mask=att_mask)
        return hidden['pooler_output']

    def tokenize(self, bag, data):
        max_len = 0
        indexed_tokens = []
        att_mask = []
        if len(bag) > self.num_nodes:
            # random_list = random.sample(range(0, len(bag)), 10)
            # for i in range(len(random_list)):
            #     sample_bag.append(bag[random_list[i]])
            sample_bag = bag[:self.num_nodes]
        else:
            sample_bag = bag

        for it, sent_id in enumerate(sample_bag):
            item = data[sent_id]
            if 'text' in item:
                sentence = item['text']
                is_token = False
            else:
                sentence = item['token']
                is_token = True
            pos_head = item['h']['pos']
            pos_tail = item['t']['pos']

            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False

            if not is_token:
                sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
                ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
                sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
                ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
                sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            else:
                sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
                ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
                sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
                ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
                sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))
                sent_temp = " ".join(sentence)

            if self.mask_entity:
                #print("Mask")
                ent0 = ['[unused5]'] if not rev else ['[unused6]']
                ent1 = ['[unused6]'] if not rev else ['[unused5]']
            else:
                ent0 = ['[unused1]'] + ent0 + ['[unused2]'] if not rev else ['[unused3]'] + ent0 + ['[unused4]']
                ent1 = ['[unused3]'] + ent1 + ['[unused4]'] if not rev else ['[unused1]'] + ent1 + ['[unused2]']

            re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2
            crr_indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
            if max_len == 0:
                max_len = len(crr_indexed_tokens)


                # Padding
            if self.blank_padding:
                while len(crr_indexed_tokens) < self.max_length:
                    crr_indexed_tokens.append(0)  # 0 is id for [PAD]
                crr_indexed_tokens = crr_indexed_tokens[:self.max_length]
            crr_indexed_tokens = torch.tensor(crr_indexed_tokens).long().unsqueeze(0)  # (1, L)
            indexed_tokens.append(crr_indexed_tokens)
             # Attention mask
            crr_att_mask = torch.zeros(crr_indexed_tokens.size()).long() # (1, L)
            crr_att_mask[0, :max_len] = 1
            att_mask.append(crr_att_mask)
        return indexed_tokens, att_mask
