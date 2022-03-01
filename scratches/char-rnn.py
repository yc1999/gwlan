# char-rnn就是将字母的embeeding输入rnn，得到最终的结果
from unicodedata import bidirectional
import torch
from torch.nn import GRU

rnn = GRU(input_size=10, hidden_size=20, bidirectional=True, batch_first=True)
input = torch.randn(5, 3, 10)
output, hn = rnn(input)
print(output.size())
print(output)
print(hn.size())