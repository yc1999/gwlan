import argparse

import torch
from datamodels.wpm_data_model import WPMDataModel
from models.base_model import BaseModel
from models.wpm_model import WPMModel
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def load_callbacks(args):
    callbacks = []
    
    checkpoint = ModelCheckpoint(dirpath=args.ckpt_dir, save_top_k=1, monitor='valid_acc', mode='max', filename='{epoch:02d}--{valid_acc:.4f}')
    callbacks.append(checkpoint)
    
    # early_stop = EarlyStopping(monitor='valid_acc', mode='max', patience=20)
    # callbacks.append(early_stop)

    lr_monitor = LearningRateMonitor()
    callbacks.append(lr_monitor)

    return callbacks   

def main(args):
    # Set Random Seeds
    seed_everything(args.seed)
    torch.set_num_threads(10)

    if args.model_name_or_path == "wpm":
        Model = WPMModel
        DataModel = WPMDataModel
    else:
        raise ValueError("model type not found : {args.model_name_or_path}")
    
    # Callbacks
    logger = loggers.TensorBoardLogger(save_dir=args.log_dir, name="")
    callbacks = load_callbacks(args)

    # Trainer
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
    
    model = Model(args)
    data_model = DataModel(args)

    # Training and Testing
    if args.do_train == True:
        trainer.fit(model, data_model)
        checkpoint_path = callbacks[0].best_model_path


if __name__ == "__main__":
    total_parser = argparse.ArgumentParser()

    # Args for data model
    total_parser = WPMDataModel.add_data_specific_args(total_parser)

    # Args for Training
    total_parser = Trainer.add_argparse_args(total_parser)

    # Args for model
    total_parser = BaseModel.add_model_specific_args(total_parser)

    args = total_parser.parse_args()

    main(args)