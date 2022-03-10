"""
This data model is specific for the WPM(Word Predict Model),
which is proposed by the paper: GWLAN: A General Word-Level AutocompletioN for Computer-Aided Translation
"""
from argparse import ArgumentParser
from ast import arg
import copy
import json
import random
from tqdm import tqdm
import math
import os
from typing import Optional
import sys

import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
import numpy as np
from torch.utils.data import DataLoader
sys.path.append("./")
from utils.tokenizer import WPMTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class WPMDataModel(pl.LightningDataModule):
    def __init__(self, args):
        self.args = args
        self.src_tokenizer = WPMTokenizer("./dataset/en2de/vocab/vocab.50K.en")
        self.masked_tokenizer = WPMTokenizer("./dataset/en2de/vocab/vocab.50K.de")
        self.max_seq_len = args.max_seq_len

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_dataset = WPMDataset(self.args, "train")
            self.val_dataset = WPMDataset(self.args, "dev")
        else:
            self.test_dataset = WPMDataset(self.args, "test")
        return

    def train_dataloader(self):
        #TODO：shuffle=False, use only the first 400,000 pairs
        train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=self.args.train_batch_size, num_workers=4, collate_fn=self.collate_fn,
            pin_memory=True
        )
        return train_dataloader
 
    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=4, collate_fn=self.collate_fn,
            pin_memory=True
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=4, collate_fn=self.collate_fn,
            pin_memory=True
        )
        return test_dataloader

    def collate_fn(self, examples):
        """In `decay` setting, batch is examples not features, so we should first convert examples to features.
        """
        #batch_size = len(batch)
        batch = WPMDataset.convert_examples_to_features(self.src_tokenizer, self.masked_tokenizer, examples)

        src_input_ids = [ torch.tensor(f.src_input_ids, dtype=torch.int) for f in batch ]
        src_attention_mask = [ torch.tensor(f.src_attention_mask, dtype=torch.bool) for f in batch ]
        masked_input_ids = [ torch.tensor(f.masked_input_ids, dtype=torch.int) for f in batch ]
        masked_attention_mask = [ torch.tensor(f.masked_attention_mask, dtype=torch.bool) for f in batch ]
        masked_position_ids = [ torch.tensor(f.masked_position_ids, dtype=torch.int) for f in batch]
        labels = [ torch.tensor(f.labels, dtype=torch.int64) for f in batch ]
        types = [f.type for f in batch]
        type_mask = [torch.tensor(f.type_mask, dtype=torch.bool) for f in batch]

        src_input_ids = pad_sequence(src_input_ids, batch_first=True, padding_value=self.src_tokenizer.pad_token_id)    # 3
        src_attention_mask = pad_sequence(src_attention_mask, batch_first=True, padding_value=True)    # attention_mask set to be True
        masked_input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=self.masked_tokenizer.pad_token_id)  # 3
        masked_attention_mask = pad_sequence(masked_attention_mask, batch_first=True, padding_value=True)  # attention_mask set to be True
        masked_position_ids = pad_sequence(masked_position_ids, batch_first=True, padding_value=127)  # position mask is 127, this is used in RelativePositionalEncoder
        labels = pad_sequence(labels, batch_first=True, padding_value=-100) # labels should be -100
        type_mask = pad_sequence(type_mask, batch_first=True, padding_value=False)

        assert src_input_ids.size() == src_attention_mask.size()
        assert masked_input_ids.size() == masked_attention_mask.size() == labels.size() == masked_position_ids.size()

        results = {
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attention_mask,
            "masked_input_ids": masked_input_ids,
            "masked_attention_mask": masked_attention_mask,
            "masked_position_ids": masked_position_ids,
            "labels": labels,
            # "types": types,
            "type_mask": type_mask,
        }

        return results


    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("WPMDataModel")
        
        ## Required parameters
        parser.add_argument(
            "--train_dir",
            type=str,
            required=True,
            help="The train data dir. Should contain the train files for the task"
        )
        parser.add_argument(
            "--valid_dir",
            type=str,
            required=True,
            help="The valid data dir. Should contain the valid files for the task"
        )
        parser.add_argument(
            "--test_dir",
            type=str,
            required=True,
            help="The test data dir. Should contain the test files for the task"
        )
        parser.add_argument(
            "--model_name_or_path", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name selected in the list"
        )
        
        parser.add_argument(
            "--max_seq_len",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        )
        parser.add_argument("--vocab_size", type=int, required=True, help="vocab size")
        parser.add_argument("--d_model", type=int, required=True, help="dimension of the model")
        parser.add_argument("--learning_rate", type=float, required=True, help="learning rate for optimizer.")
        parser.add_argument("--warmup_ratio", type=float, default=-1, help="warmup_ratio for training with warm up.")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout ratio for model.")
        parser.add_argument("--scheduler", type=str, default="linear", required=True, help="learning rate scheduler.")
        parser.add_argument("--warmup_steps", type=int, default=-1, help="warmup steps for training.")

        parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
        parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        parser.add_argument("--train_dataset_ratio", type=float, default=1.0)

        parser.add_argument("--log_dir", default="./log", type=str, help="The log directory")
        parser.add_argument("--ckpt_dir", default=".", type=str, help="The ckpt directory")
        parser.add_argument("--ckpt_path", default=".", type=str)
        parser.add_argument("--save_dir", default="./save", type=str)
        parser.add_argument("--pred_file", default="pred.json", type=str, help="The pred file")

        parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
        parser.add_argument("--do_test", action="store_true", help="Whether to run predictions on the test set.")
        
        parser.add_argument("--setting", type=str, default="debug", choices=["debug", "run"], help="debug or run")
        return parent_parser


class WPMDataset(Dataset):
    def __init__(self, args, mode):
        super().__init__()
        self.args = args
        self.examples = self.load_and_cache_examples(args, mode)
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == "train":
            example = self.examples[index]
            # dynamic mask here
            example = self.construct_train_bi_context_example(example)
            return example
        elif self.mode == "dev":
            return self.examples[index]
        elif self.mode == "test":
            return self.examples[index]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.examples)        

    def load_and_cache_examples(self, args, mode):
        """Load dataset from cache or create dataset
        """
        examples = self.read_examples_from_file(mode)
        # sample examples
        if mode == "train":
            if args.train_dataset_ratio != 1.0:
                examples = random.sample(examples, int(len(examples) * args.train_dataset_ratio ))
            print("Num of examples are: ",len(examples))
            return examples

        elif mode == "dev" or mode == "test":
            return examples
        else:
            raise ValueError("no mode")

    def read_examples_from_file(self, mode):
        if mode == "train":
            file_path = self.args.train_dir
        elif mode == "dev":
            file_path = self.args.valid_dir
        elif mode == "test":
            file_path = self.args.test_dir
        else:
            raise ValueError(f"{mode} not found...")
        
        if mode == "train":
            with open(file_path + ".src") as f:
                srcs = f.read().splitlines()
            with open(file_path + ".tgt") as f:
                tgts = f.read().splitlines()
            examples = []
            for i, (src, tgt) in enumerate( zip(srcs, tgts) ):
                guid = f"{mode}-{i}"
                examples.append( InputExample(guid=guid, src=src, masked=None, type=None, suggestion=None, tgt=tgt) )
            return examples
        
        elif mode == "dev" or mode == "test":
            with open(file_path + ".src") as f:
                srcs = f.read().splitlines()
            with open(file_path + ".masked") as f:
                maskeds =  f.read().splitlines()
            with open(file_path + ".type") as f:
                types = f.read().splitlines()
            with open(file_path + ".suggestion") as f:
                suggestions = f.read().splitlines()
            
            examples = []
            for i, (src, masked, type, suggestion) in enumerate(zip(srcs, maskeds, types, suggestions)):
                guid = f"{mode}-{i}"
                examples.append(InputExample(guid=guid, src=src, masked=masked, type=type, suggestion=suggestion, tgt=None))
            return examples
 
        else:
            raise NotImplementedError

    def construct_train_bi_context_example(self, example):
        tgt = example.tgt
        tgt_word_lst = tgt.split()
        tgt_len = len(tgt_word_lst)

        # select a pivot from the tgt_word_lst
        pivot = random.randint( 0 , len(tgt_word_lst)-1 )
        max_span_length = int( math.ceil( tgt_len*0.25) )
        left_shift = min( random.randint(0, pivot), max_span_length )
        right_shift = min( random.randint(0, len(tgt_word_lst)-1-pivot), max_span_length )
        left = pivot - left_shift
        right = pivot + right_shift
        
        # mask the span [left, right]
        masked_tgt_word_lst = copy.deepcopy(tgt_word_lst)
        for idx in range(left, right+1):
            masked_tgt_word_lst[idx] = "<mask>"
        assert len(tgt_word_lst) == len(masked_tgt_word_lst)
        masked = " ".join(masked_tgt_word_lst)
        

        # tgt_word_lst = tgt.split()
        
        # # 如果全是小于等于4的字母
        # contains_large = sum([len(word) > 4 for word in tgt_word_lst])
        # if contains_large > 0:
        #     word_masked_idx = random.randint( 0, len(tgt_word_lst)-1 )
        #     word = tgt_word_lst[word_masked_idx]
        #     while len(word) <= 4:
        #         word_masked_idx = random.randint(0, len(tgt_word_lst)-1)
        #         word = tgt_word_lst[word_masked_idx]
        # else:
        #     word_masked_idx = random.randint( 0, len(tgt_word_lst)-1 )
        #     word = tgt_word_lst[word_masked_idx]
        #     while len(word) < 1:
        #         word_masked_idx = random.randint(0, len(tgt_word_lst)-1)
        #         word = tgt_word_lst[word_masked_idx]

        # if len(word) != 1:
        #     type_masked_idx = random.randint(1, len(word)-1)    
        #     type = word[:type_masked_idx]
        # else:
        #     type = word
        
        # # 对word_masked_idx之前的字符进行
        # if word_masked_idx == 0:
        #     prefix = ""
        # else:
        #     pl1 = 0
        #     pl2 = random.randint(max(0, word_masked_idx-7), word_masked_idx-1)
        #     # pl1, pl2 = min(pl1, pl2), max(pl1, pl2)
        #     prefix = " ".join(tgt_word_lst[pl1 : pl2+1])

        # if word_masked_idx == len(tgt_word_lst)-1:
        #     suffix = ""
        # else:
        #     pr1 = random.randint(word_masked_idx+1, min(word_masked_idx+7,len(tgt_word_lst)-1))
        #     pr2 = len(tgt_word_lst)-1
        #     # pr1, pr2 = min(pr1, pr2), max(pr1, pr2)
        #     suffix = " ".join(tgt_word_lst[pr1 : pr2+1])

        # masked = " ".join( list(filter(None, [prefix, "<mask>", suffix])) ) # filter empty string

        # For each tgt-lang sentence, we will mask each word in the random selected span! (Because I found that mask only one word is not very satisfying)
        return InputExample(guid=example.guid, src=example.src, masked=masked, type=None, suggestion=None, tgt=example.tgt)

    @staticmethod
    def convert_examples_to_features(src_tokenizer, masked_tokenizer, examples):
        """Loads a data file into a list of `InputBatch`s
            Encoder part is : src
            Decoder part is : <sos> + masked + <eos>
        """
        features = []
        src_len_lst = []
        masked_len_lst = []
        stat_info = dict()

        for (ex_index, example) in enumerate( examples ):
            # if ex_index % 1000 == 0:
            #     print("#"*15)
            #     print(f"Writing example {ex_index} of {len(examples)}")
            #     print("#"*15)

            src_encoded_results = src_tokenizer.tokenize(example.src)
            src_input_ids = src_encoded_results["input_ids"]
            src_attention_mask = src_encoded_results["attention_mask"]
            src_seq_len = len(src_input_ids)
            src_len_lst.append(src_seq_len)

            masked_encoded_results = masked_tokenizer.tokenize(example.masked)
            masked_input_ids = [ masked_tokenizer.sos_token_id ] + masked_encoded_results["input_ids"] + [ masked_tokenizer.eos_token_id ]
            masked_attention_mask = [False] + masked_encoded_results["attention_mask"] + [False]

            # masked_postion_ids version 1
            masked_position_ids = [0 for _ in range(len(masked_input_ids))]
            position_id = 0
            for idx, token_id in enumerate(masked_input_ids):
                if token_id == masked_tokenizer.mask_token_id and ( (idx == 0) or (masked_input_ids[idx-1] != masked_tokenizer.mask_token_id)):
                    masked_attention_mask[idx] = True
                    masked_position_ids[idx] = position_id
                    position_id += 1
                elif token_id == masked_tokenizer.mask_token_id and masked_input_ids[idx-1] == masked_tokenizer.mask_token_id:
                    masked_attention_mask[idx] = True
                    masked_position_ids[idx] = masked_position_ids[idx-1]
                else:
                    masked_position_ids[idx] = position_id
                    position_id += 1
            masked_seq_len = len(masked_input_ids)
            masked_len_lst.append(masked_seq_len)
            # masked_position_ids version 2
            # masked_index = masked_input_ids.index(masked_tokenizer.mask_token_id) 
            # masked_position_ids = [0 for _ in range(len(masked_input_ids))]
            # position_id = 0
            # for idx in range( len(masked_position_ids) ):
            #     if idx == masked_index:
            #         masked_position_ids[idx] = position_id
            #     elif idx == masked_index - 1:
            #         masked_position_ids[idx] = position_id
            #     else:
            #         masked_position_ids[idx] = position_id
            #         position_id += 1

            # masked_position_ids version 3
            # masked_position_ids = [2 for _ in range(len(masked_input_ids))]
            # masked_position_ids[:masked_index]  = [0 for _ in range(masked_index)]
            # masked_position_ids[masked_index] = 1
        

            type = example.type # a str reprensents human typed characters
            if type != None:
                # for dev and test
                suggestion = masked_tokenizer.convert_token_to_id(example.suggestion)
                labels = [  -100 if token_id != masked_tokenizer.mask_token_id else suggestion for token_id in masked_input_ids ]
                type_mask = [ word.startswith(type) for word in masked_tokenizer.vocabs ]
            else:
                # for training
                suggestion = masked_tokenizer.convert_token_to_id(example.suggestion)
                labels = [ masked_tokenizer.sos_token_id ]  + masked_tokenizer.tokenize(example.tgt)["input_ids"] + [ masked_tokenizer.eos_token_id ]
                assert len(masked_input_ids) == len(labels)
                for idx, token_idx in enumerate(masked_input_ids):
                    if token_idx != masked_tokenizer.mask_token_id:
                        labels[idx] = -100
                type_mask = [False for _ in masked_tokenizer.vocabs]

            features.append(
                InputFeature(
                    guid=example.guid,
                    src_input_ids=src_input_ids,
                    src_attention_mask=src_attention_mask,
                    src_seq_len=src_seq_len,
                    masked_input_ids=masked_input_ids,
                    masked_attention_mask=masked_attention_mask,
                    masked_position_ids=masked_position_ids,
                    masked_seq_len=masked_seq_len,
                    labels=labels,
                    type=type,
                    suggestion=suggestion,
                    type_mask=type_mask,
                )
            )
        
        stat_info["src_len"] = dict()
        stat_info["src_len"]["mean"] = np.mean(src_len_lst)
        stat_info["src_len"]["std"] = np.std(src_len_lst, ddof=1)
        stat_info["src_len"]["max"] = max(src_len_lst)

        stat_info["masked_len"] = dict()
        stat_info["masked_len"]["mean"] = np.mean(masked_len_lst)
        stat_info["masked_len"]["std"] = np.std(masked_len_lst, ddof=1)
        stat_info["masked_len"]["max"] = max(masked_len_lst)

        # print(stat_info)

        return features

class InputExample:
    """A single example for token classification"""
    def __init__(self, guid, src, masked, type, suggestion, tgt):
        """
        Args:
            guid: Unique id for the example.
            src: source sentence
            masked: masked target sentence(bi-context, prefix, suffix or zero-context)
            type: human typed characters
            suggestion: ground truth label
            tgt: target sentence without mask
        """
        self.guid = guid
        self.src = src
        self.masked = masked
        self.type = type
        self.suggestion = suggestion
        self.tgt = tgt
    
    def __repr__(self):
        return str( self.to_json_string() )
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps( self.to_dict(), indent=True, sort_keys=True ) + "\n"

class InputFeature:
    def __init__(self, guid, src_input_ids, src_attention_mask, src_seq_len, masked_input_ids, masked_attention_mask, masked_seq_len, masked_position_ids,
                type, suggestion, labels, type_mask):
        self.guid = guid
        self.src_input_ids = src_input_ids
        self.src_attention_mask = src_attention_mask
        self.src_seq_len = src_seq_len
        self.masked_input_ids = masked_input_ids
        self.masked_attention_mask = masked_attention_mask
        self.masked_position_ids = masked_position_ids
        self.masked_seq_len = masked_seq_len
        self.type = type
        self.suggestion = suggestion
        self.labels = labels
        self.type_mask = type_mask

    def __repr__(self):
        return str( self.to_json_string() )
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps( self.to_dict(), indent=True, sort_keys=True ) + "\n"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = WPMDataModel.add_data_specific_args(parent_parser=parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    wpm_data_model = WPMDataModel(args)
    wpm_data_model.setup('fit')

    train_dataloader = wpm_data_model.train_dataloader()
    # for i in range(3):
    for batch in train_dataloader:
            # batch = next( iter(train_dataloader) )
        print(batch)
    print("yes")
    