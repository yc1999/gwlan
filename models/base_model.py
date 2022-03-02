import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('BaseModel')
        return parent_parser
    
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters(self.args)
    
    def setup(self, stage):
        if stage == "fit":
            train_loader = self.trainer.datamodule.train_dataloader()
            self.total_step = int( (self.trainer.max_epochs * len(train_loader) * self.trainer.limit_train_batches) / (self.trainer.gpus * self.trainer.accumulate_grad_batches) )
            print(f"Total optimization step is {self.total_step}")