"""This file is for data processing
"""
import json
from collections import defaultdict
from tqdm import tqdm
import random
random.seed(42)

tgt_len_lst = []
masked_len_lst = []

def get_vocab(src_lang_file, tgt_lang_file=None):
    # 原始的词表非常大，de + en = 1.7M
    # 分别各取top 50k作为词表就好

    # src
    vocab = defaultdict(int)
    with open(src_lang_file, "r", encoding="utf-8") as f:
        for line in f.read().splitlines():
            word_lst = line.split()
            for word in word_lst:
                vocab[word] += 1
    # 按照value进行排序
    vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)[:50000]
    vocab = [("<mask>", len(vocab))] + vocab
    vocab = [("<pad>", len(vocab))] + vocab
    vocab = [("<unk>", len(vocab))] + vocab
    vocab = [("<eos>", len(vocab))] + vocab
    vocab = [("<sos>", len(vocab))] + vocab

    new_vocab = dict()
    for idx, pair in enumerate(vocab):
        new_vocab[ pair[0] ] = idx
    with open("./dataset/en2de/vocab/vocab.50K.en", "w", encoding="utf-8") as f:
        json.dump(new_vocab, f)       
    
    
    # tgt
    vocab = defaultdict(int)
    with open(tgt_lang_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            word_lst = line.split()
            for word in word_lst:
                vocab[word] += 1
    # 按照value进行排序
    vocab = sorted(vocab.items(), key=lambda item: item[1], reverse=True)[:50000]
    vocab = [("<mask>", len(vocab))] + vocab
    vocab = [("<pad>", len(vocab))] + vocab
    vocab = [("<unk>", len(vocab))] + vocab
    vocab = [("<eos>", len(vocab))] + vocab
    vocab = [("<sos>", len(vocab))] + vocab

    new_vocab = dict()
    for idx, pair in enumerate(vocab):
        new_vocab[ pair[0] ] = idx
    vocab = dict(vocab)
    with open("./dataset/en2de/vocab/vocab.50K.de", "w", encoding="utf-8") as f:
        json.dump(new_vocab, f)

def lowcase_file(src_file, tgt_file):
    """lowcase the sentences in file.
    """
    with open(src_file, mode="r") as f:
        with open(tgt_file, mode="w") as g:
            lines = f.readlines()
            for line in lines:
                g.write(line.lower())

def write_to_file(src, tgt, masked, type, suggestion, src_file, tgt_file, masked_file, type_file, suggestion_file):
    with open(src_file, "a") as f:
        f.write(src + "\n")

    with open(tgt_file, "a") as f:
        f.write(tgt + "\n")

    with open(masked_file, "a") as f:
        f.write(masked + "\n")
    
    with open(type_file, "a") as f:
        f.write(type + "\n")

    with open(suggestion_file, "a") as f:
        f.write(suggestion + "\n")

def construct_train_zero(src, tgt, src_file, masked_file, type_file, suggestion_file):
    word_lst = tgt.split()
    # random.randint(a,b)是双端inclusive的~
    word_masked_idx = random.randint(0, len(word_lst)-1)
    word = tgt[word_masked_idx]
    while len(word) <= 4:
        word_masked_idx = random.randint(0, len(word_lst)-1)
        word = tgt[word_masked_idx]

    type_masked_idx = random.randint(1, len(word)-1)    
    type = word[:type_masked_idx]
    masked = "<mask>"
    write_to_file(src, masked, type, word, src_file, masked_file, type_file, suggestion_file)

def construct_train_prefix(src, tgt, src_file, masked_file, type_file, suggestion_file):
    word_lst = tgt.split()
    word_masked_idx = random.randint(0, len(word_lst)-1)
    word = tgt[word_masked_idx]
    while len(word) <= 4:
        word_masked_idx = random.randint(0, len(word_lst)-1)
        word = tgt[word_masked_idx]

    type_masked_idx = random.randint(1, len(word)-1)    
    type = word[:type_masked_idx]

    # 对word_masked_idx之前的字符进行
    pl1 = random.randint(0, word_masked_idx-2)
    pl2 = random.randint(pl1+1, word_masked_idx-1)
    prefix = " ".join(word_lst[pl1 : pl2+1])
    masked = " ".join([prefix, "<mask>"])
    write_to_file(src, masked, type, word, src_file, masked_file, type_file, suggestion_file)

def construct_train_suffix(src, tgt, src_file, masked_file, type_file, suggestion_file):
    word_lst = tgt.split()
    word_masked_idx = random.randint(0, len(word_lst)-1)
    word = tgt[word_masked_idx]
    while len(word) <= 4:
        word_masked_idx = random.randint(0, len(word_lst)-1)
        word = tgt[word_masked_idx]

    type_masked_idx = random.randint(1, len(word)-1)    
    type = word[:type_masked_idx]

    # 对word_masked_idx之前的字符进行
    pr1 = random.randint(word_masked_idx+1, len(word_lst)-2)
    pr2 = random.randint(pr1+1, len(word_lst)-1)
    suffix = " ".join(word_lst[pr1 : pr2+1])
    masked = " ".join(["<mask>", suffix])
    write_to_file(src, masked, type, word, src_file, masked_file, type_file, suggestion_file)    

def construct_train_bi(src, tgt, src_file, tgt_file, masked_file, type_file, suggestion_file):
    tgt_word_lst = tgt.split()
    
    # 如果全是小于等于4的字母
    contains_large = sum([len(word) > 4 for word in tgt_word_lst])
    if contains_large > 0:
        word_masked_idx = random.randint( 0, len(tgt_word_lst)-1 )
        word = tgt_word_lst[word_masked_idx]
        while len(word) <= 4:
            word_masked_idx = random.randint(0, len(tgt_word_lst)-1)
            word = tgt_word_lst[word_masked_idx]
    else:
        word_masked_idx = random.randint( 0, len(tgt_word_lst)-1 )
        word = tgt_word_lst[word_masked_idx]
        while len(word) < 1:
            word_masked_idx = random.randint(0, len(tgt_word_lst)-1)
            word = tgt_word_lst[word_masked_idx]

    if len(word) != 1:
        type_masked_idx = random.randint(1, len(word)-1)    
        type = word[:type_masked_idx]
    else:
        type = word
    
    # 对word_masked_idx之前的字符进行
    if word_masked_idx == 0:
        prefix = ""
    else:
        pl1 = 0
        pl2 = random.randint(max(0, word_masked_idx-7), word_masked_idx-1)
        # pl1, pl2 = min(pl1, pl2), max(pl1, pl2)
        prefix = " ".join(tgt_word_lst[pl1 : pl2+1])

    if word_masked_idx == len(tgt_word_lst)-1:
        suffix = ""
    else:
        pr1 = random.randint(word_masked_idx+1, min(word_masked_idx+7,len(tgt_word_lst)-1))
        pr2 = len(tgt_word_lst)-1
        # pr1, pr2 = min(pr1, pr2), max(pr1, pr2)
        suffix = " ".join(tgt_word_lst[pr1 : pr2+1])

    masked = " ".join([prefix, "<mask>", suffix])
    write_to_file(src, tgt, masked, type, word, src_file, tgt_file, masked_file, type_file, suggestion_file)

    tgt_len_lst.append(len(tgt.split()))
    masked_len_lst.append(len(masked.split()))

if __name__ == "__main__":
    # get_vocab("./dataset/wmt14_dev_data/bi_context_raw_data/dev.en2de.en", "./dataset/wmt14_dev_data/bi_context_raw_data/dev.de2en.de")
    # lowcase_file("dataset/en2de/train/upcase/train.de", "dataset/en2de/train/lowcase/train.de")
    # get_vocab("./dataset/en2de/train/lowcase/train.en", "./dataset/en2de/train/lowcase/train.de")

    # 构建bi-context的数据集
    with open("./dataset/en2de/train/lowcase/train.en") as f:
        src_lines = f.read().splitlines()
    with open("./dataset/en2de/train/lowcase/train.de") as g:
        tgt_lines = g.read().splitlines()
    for idx in tqdm(range(len(src_lines))):
        # print(idx)
        src = src_lines[idx]
        tgt = tgt_lines[idx]
        construct_train_bi(src, tgt, 
                            "./dataset/en2de/train/train.en2de.de.out.bi_context.src",
                            "./dataset/en2de/train/train.en2de.de",
                            "./dataset/en2de/train/train.en2de.de.out.bi_context.masked",
                            "./dataset/en2de/train/train.en2de.de.out.bi_context.type",
                            "./dataset/en2de/train/train.en2de.de.out.bi_context.suggestion"
                            )
    avg = (sum(tgt_len_lst) - sum(masked_len_lst)) / len(tgt_len_lst)
    print(avg)
         