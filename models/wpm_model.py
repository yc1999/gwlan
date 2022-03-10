from audioop import bias
from utils.tokenizer import WPMTokenizer
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
from modules.positional_encoder import RelativePositionalEncoder, SinusoidalPostionalEncoder
from models.base_model import BaseModel
import torch
import torch.nn as nn
import os
from typing import Dict

class WPMModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        
        self.embeddings = WPMEmbeddings(args)
        self.transformer = nn.Transformer(batch_first=True, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048)  # This is the default params in torch.nn.Transformer document

        #   分类头参考自huggingface BertMaskLM 
        self.cls = nn.Sequential(
            nn.Linear(args.d_model, args.d_model),
            nn.LayerNorm(args.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.d_model, args.vocab_size)
        )
        # self.cls = nn.Linear(args.d_model, args.vocab_size, bias=False)
        # shared weight
        # self.cls.weight = self.embeddings.masked_token_encoder.weight 

        self.loss_fn = nn.CrossEntropyLoss()
        #TODO:需要同一对参数进行初始化吗？nn.Embedding有自己的初始化函数~

        #TODO:怎么使用LayerNorm，ReLU和Dropout
        # LayerNorm用于归一化，ReLU()用于非线性化，Dropout用于防止过拟合。
        self.masked_tokenizer = WPMTokenizer("./dataset/en2de/vocab/vocab.50K.de")

    def forward(self, src_input_ids, src_attention_mask, masked_input_ids, masked_attention_mask, masked_position_ids):
        """
        Args:
            src_input_ids: (batch_size, src_len)
            src_attention_mask: (batch_size, src_len)
            masked_input_ids: (batch_size, masked_len)
            masked_attention_mask: (batch_size, masked_len)
        """
        src_embeddings, masked_embeddings = self.embeddings(src_input_ids=src_input_ids, masked_input_ids=masked_input_ids, masked_position_ids=masked_position_ids)
        
        # logits: (batch_size, masked_len, d_model)
        hidden_states = self.transformer.forward(src=src_embeddings, tgt=masked_embeddings, src_key_padding_mask=src_attention_mask, tgt_key_padding_mask=masked_attention_mask, memory_key_padding_mask=src_attention_mask)
        
        # usually, forward() function is used to calculate logits~ not loss, we should calculate loss in training_step()
        # logits: (batch_size, masked_len, vocab_size)
        logits = self.cls(hidden_states)

        return logits

    def get_inputs(self, batch):
        inputs = dict()
        inputs["src_input_ids"] = batch["src_input_ids"]
        inputs["src_attention_mask"] = batch["src_attention_mask"]
        inputs["masked_input_ids"] = batch["masked_input_ids"]
        inputs["masked_attention_mask"] = batch["masked_attention_mask"]
        inputs["masked_position_ids"] = batch["masked_position_ids"]
        return inputs

    def training_step(self, batch, batch_idx):
        inputs = self.get_inputs(batch)
        logits = self.forward(**inputs)
        labels = batch["labels"]
        train_loss = self.loss_fn.forward(logits.view(-1, self.args.vocab_size), labels.view(-1))
        self.log("train_loss", train_loss.detach().cpu(), prog_bar=True)
        
        # prediction
        # index = (labels != -100)
        # labels = labels[index].cpu().numpy()
        # logits = logits[index]
        # type_mask = batch["type_mask"]
        # logits_mask = torch.logical_not(type_mask)
        # logits = logits.masked_fill(logits_mask, -float("inf"))
        # preds = torch.max(logits, dim=-1).indices.cpu().numpy()
        # true = sum(labels == preds)
        # size = preds.shape[0]
        # self.log("train_acc", true/size, prog_bar=True)
        
        # log gradient norm
        # parameters = [p for p in self.parameters() if p.grad is not None]
        # if len(parameters) == 0:
        #     total_norm = 0.0
        # else:
        #     total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters])).detach().cpu()
        # self.log("grad", total_norm)

        
        # for name,params in self.named_parameters():
        #     if params.requires_grad == False:
        #         #print(params.requires_grad)
        #         tensorboard = self.logger.experiment
        #         log_str = "{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(name, params.detach().min(), params.detach().max(), params.detach().mean(), params.detach().std())
        #         tensorboard.add_text(name, log_str)
        #         print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(name, params.detach().min(), params.detach().max(), params.detach().mean(), params.detach().std()))

        return train_loss
    
    def validation_step(self, batch, batch_idx):
        inputs = self.get_inputs(batch)
        
        # labels: (batch_size, masked_seq_len)
        labels =  batch["labels"]
        
        # logits: (batch_size, masked_seq_len, vocab_size)
        logits = self.forward(**inputs)    

        # masked_position_ids: (batch_size, masked_seq_len)
        # masked_position_ids = batch["masked_position_ids"]
        index = (labels != -100)
        # labels: (batch_size, )
        labels = labels[index].cpu().numpy()
        
        # logits: (batch_size, vocab_size)
        logits = logits[index]

        # type_mask: (batch_size, vocab_size)
        type_mask = batch["type_mask"]
        logits_mask = torch.logical_not(type_mask)
        # preds: (batch_size, )
        logits = logits.masked_fill(logits_mask, -float("inf"))
        preds = torch.max(logits, dim=-1).indices.cpu().numpy()

        true = sum(labels == preds)
        size = preds.shape[0]
        return true, size

    def validation_epoch_end(self, outputs) :
        true = sum([result[0] for result in outputs ])
        size = sum([result[1] for result in outputs ])
        valid_acc = true / size
        self.log("valid_acc", valid_acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs = self.get_inputs(batch)
        
        # labels: (batch_size, masked_seq_len)
        labels =  batch["labels"]
        
        # logits: (batch_size, masked_seq_len, vocab_size)
        logits = self.forward(**inputs)    

        # masked_position_ids: (batch_size, masked_seq_len)
        # masked_position_ids = batch["masked_position_ids"]
        index = (labels != -100)
        # labels: (batch_size, )
        labels = labels[index].cpu().numpy()
        
        # logits: (batch_size, vocab_size)
        logits = logits[index]

        # type_mask: (batch_size, vocab_size)
        type_mask = batch["type_mask"]
        logits_mask = torch.logical_not(type_mask)
        # preds: (batch_size, )
        logits = logits.masked_fill(logits_mask, -float('inf'))
        preds = torch.max(logits, dim=-1).indices.cpu().numpy()

        true = sum(labels == preds)
        size = preds.shape[0]
        
        # masked_tokenizer = WPMTokenizer("./dataset/en2de/vocab/vocab.50K.de")
        with open(self.args.pred_file, "a") as f:
            for idx, pred in enumerate(preds):
                f.write(self.masked_tokenizer.decoder[labels[idx]] + " " +self.masked_tokenizer.decoder[pred] + "\n")
                # f.write(self.masked_tokenizer.decoder[labels[idx]] + "\n")

        return true, size
        # return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) :
        true = sum([result[0] for result in outputs])
        size = sum([result[1] for result in outputs])
        test_acc = true / size
        self.log("test_acc", test_acc, on_epoch=True, prog_bar=True)

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        """Override this method to change the default behaviour of ``log_grad_norm``.

        If clipping gradients, the gradients will not have been clipped yet.

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        Example::

            # DEFAULT
            def log_grad_norm(self, grad_norm_dict):
                self.log_dict(grad_norm_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        """
        self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        # #TODO: optimizer的learning rate需要跟Attention is All You Need 论文里面对齐~ 可能需要自己写optimizer
        # filter requires_grad() Parameter
        optimizer = Adam( filter(lambda p : p.requires_grad, self.parameters()), lr=self.args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
        if self.args.scheduler == "linear":
            if self.args.warmup_steps == -1:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.total_step * self.args.warmup_ratio, num_training_steps = self.total_step)
            else:
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps = self.total_step)
        elif self.args.scheduler == "noam":
            scheduler = self.get_noam_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps = self.total_step)

        return [
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", # this means that scheduler.step() while to called in pl.LightningModule.optimize_step()
                    "frequency": 1
                }
            }
        ]

    @staticmethod
    def update_output_dirs(args):
        if args.setting == "debug":
            args.save_dir = "./save/tmp"
        else:
            args.save_dir = os.path.join(args.save_dir, args.model_name_or_path)

        # parameters not strongly associated with model
        data_paras = "seed: {} - max_epochs: {} - max_steps: {} - warmup_ratio: {} - warmup_steps: {} - train_dataset_ratio: {} - train_bacth_size: {} - gpus: {} - accumulate_grad_batches: {} - lr: {} -scheduler: {} - dropout: {} - gradient_clip_val: {}".format(args.seed, args.max_epochs, args.max_steps, 
                                                    args.warmup_ratio, args.warmup_steps,
                                                    args.train_dataset_ratio, args.train_batch_size, 
                                                    args.gpus, args.accumulate_grad_batches,
                                                    args.learning_rate, args.scheduler,
                                                    args.dropout, args.gradient_clip_val, )
        args.save_dir = os.path.join(args.save_dir, data_paras)
        
        # parameters strongly associated with model
        # model_paras = "use_label_fusion: {} - use_attn: {} - add_label_info: {} - use_res: {}".format(args.use_label_fusion, args.use_attn, args.add_label_info, args.use_res)
        # model_paras = os.path.join(model_paras, time.strftime("%Y%m%d-%H%M", time.localtime()))
        args.log_dir = os.path.join(args.save_dir, args.log_dir)
        args.ckpt_dir = os.path.join(args.save_dir, args.ckpt_dir)
        args.pred_file = os.path.join(args.save_dir, args.pred_file)
        return args

class WPMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and ~~token_type~~ embeddings."""
    def __init__(self, args):
        super().__init__()
        
        self.src_LayerNorm = nn.LayerNorm(normalized_shape=512) # eps is set to default->0.00001
        self.src_dropout = nn.Dropout(args.dropout)

        self.masked_LayerNorm = nn.LayerNorm(normalized_shape=512) # eps is set to default->0.00001
        self.masked_dropout = nn.Dropout(args.dropout)
        
        # token encoder
        self.src_token_encoder = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.d_model, padding_idx=3)
        self.masked_token_encoder = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=args.d_model, padding_idx=3)

        # position encoder
        self.src_pos_encoder = SinusoidalPostionalEncoder(max_seq_len=args.max_seq_len, d_model=512)
        # ~~self.masked_pos_encoder = RelativePositionalEncoder(d_model=512)~~
        self.masked_pos_encoder = self.src_pos_encoder

        # src_position ids
        self.register_buffer("src_position_ids", torch.arange(args.max_seq_len).expand((1,-1)))
    
    def forward(self, src_input_ids, masked_input_ids, masked_position_ids):
        # get src_position_ids
        src_input_shape = src_input_ids.size()
        src_seq_len = src_input_shape[1]
        src_position_ids = self.src_position_ids[:, :src_seq_len]

        # get src_embeddings
        src_input_embeddings = self.src_token_encoder(src_input_ids)
        src_position_embeddings = self.src_pos_encoder(src_position_ids)
        src_embeddings = src_input_embeddings + src_position_embeddings
        src_embeddings = self.src_LayerNorm(src_embeddings)
        src_embeddings = self.src_dropout(src_embeddings)

        # get masked_embeddings
        masked_input_embeddings = self.masked_token_encoder(masked_input_ids)
        masked_position_embeddings = self.masked_pos_encoder(masked_position_ids)
        masked_embeddings = masked_input_embeddings + masked_position_embeddings
        masked_embeddings = self.masked_LayerNorm(masked_embeddings)
        masked_embeddings = self.masked_dropout(masked_embeddings)

        return src_embeddings, masked_embeddings



    
