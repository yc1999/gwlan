import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

class BaseModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('BaseModel')
        return parent_parser
    
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters(self.args)
    
    def get_noam_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        
        def lr_lambda(current_step: int):
            # print("current_step: " , current_step)
            if current_step == 0:
                return 0.0
            ratio = (self.args.d_model ** (-0.5)) * (min( current_step ** (-0.5), current_step * num_warmup_steps**(-1.5) ))
            return ratio
        
        return LambdaLR(optimizer, lr_lambda, last_epoch)
        
    def setup(self, stage):
        if stage == "fit":
            if self.trainer.max_steps == -1:
                train_loader = self.trainer.datamodule.train_dataloader()
                self.total_step = int( ( self.trainer.max_epochs * len(train_loader) )  / (self.trainer.gpus * self.trainer.accumulate_grad_batches) )
            else:
                self.total_step = self.trainer.max_steps
            print(f"Total optimization step is {self.total_step}")