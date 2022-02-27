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
transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=True)

src = torch.rand((2, 10, 512))
tgt = torch.rand((2, 20, 512))

src_key_padding_mask = torch.zeros((2,10), dtype=torch.bool)
src_key_padding_mask[0][5:] = True

tgt_key_padding_mask = torch.zeros((2,20), dtype=torch.bool)
tgt_key_padding_mask[0][10:] = True

memory_key_padding_mask = torch.zeros((2,10), dtype=torch.bool)
memory_key_padding_mask[0][5:] = True

out = transformer_model(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

print(out.size())

# from transformers import AutoModel, AutoTokenizer, BertModel

# model = AutoModel.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# sent1 = "I am very good!"
# sent2 = "I am very sad!"
# inputs = tokenizer([sent1, sent2], return_tensors="pt")
# outputs = model(**inputs)
