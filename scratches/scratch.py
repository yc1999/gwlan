# from transformers import BertTokenizer, BertForMaskedLM
# import torch

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
# labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]

# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits
# print(logits)

# from transformers import BertTokenizer, BertLMHeadModel, BertConfig
# import torch

# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
# config = BertConfig.from_pretrained("bert-base-cased")
# config.is_decoder = True
# model = BertLMHeadModel.from_pretrained("bert-base-cased", config=config)

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# prediction_logits = outputs.logits
# print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
# outputs = model.generate(input_ids=inputs.input_ids)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# print(prediction_logits.size())

import torch
import torch.nn as nn
transformer_model = nn.Transformer(nhead=8, num_encoder_layers=6, batch_first=True)

src = torch.rand((2, 5, 512))
tgt = torch.rand((2, 10, 512))

src_key_padding_mask = torch.zeros((2,5), dtype=torch.bool)
src_key_padding_mask[:, 3:] = True

tgt_key_padding_mask = torch.zeros((2,10), dtype=torch.bool)
tgt_key_padding_mask[:, 8:] = True

# memory_key_padding_mask = torch.zeros((2,5), dtype=torch.bool)
# memory_key_padding_mask[0][3:] = True
tgt_mask = nn.Transformer.generate_square_subsequent_mask(10)

out = transformer_model(src, tgt, 
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)

print(out.size())

# from transformers import AutoModel, AutoTokenizer, BertModel

# model = AutoModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# sent1 = "I am very good!"
# sent2 = "I am very sad!"
# inputs = tokenizer([sent1, sent2], return_tensors="pt")
# outputs = model(**inputs)

# import math
# import random

# def gen_bi_context_span_from_sent(tgt_len, pos):
#     max_span_length = int(math.ceil(tgt_len * 0.25))
#     left_shift = min(random.randint(0, pos), max_span_length)
#     right_shift = min(random.randint(1, tgt_len - pos), max_span_length + 1)
#     lhs = max(1, pos - left_shift)
#     rhs = min(pos + right_shift, tgt_len)
#     return lhs, rhs

# def gen_bi_context_mask(tgt_len, pos):
#     if tgt_len <= 1:
#         return [0] * tgt_len
#     span = gen_bi_context_span_from_sent(tgt_len, pos)
#     masked = [0] * tgt_len
#     for word_id in range(span[0], span[1]):
#         masked[word_id] = 1
#     return masked

# def get_pos_with_mask(tgt_mask):
#     row_pos = []
#     n = 0
#     tgt_len = len(tgt_mask)
#     for j in range(tgt_len):
#         if tgt_mask[j]:
#             if j > 0 and tgt_mask[j - 1]:
#                 row_pos.append(row_pos[-1])
#             else:
#                 row_pos.append(n)
#                 n += 1
#         else:
#             row_pos.append(n)
#             n += 1
#     return row_pos

# print(get_pos_with_mask([0,0,0,0,1,1,1,0,0]))

# import torch

# x = torch.arange(40).reshape(4, 10)
# indices = torch.tensor(
#     [[2, 3],
#     [6, 7],
#     [0, 1],
#     [4, 5],]
# )
# print(torch.gather(x, 1, indices))
# batch_size = 8
# max_seq_len = 9
# hidden_size = 6
# x = torch.empty(batch_size, max_seq_len, hidden_size)
# for i in range(batch_size):
#   for j in range(max_seq_len):
#     for k in range(hidden_size):
#       x[i,j,k] = i + j*10 + k*100
# print(x)