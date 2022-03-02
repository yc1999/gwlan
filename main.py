import argparse
import torch
from datamodels.wpm_data_model_decay import WPMDataModel
from models.base_model import BaseModel
from models.wpm_model import WPMModel
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def load_callbacks(args):
    callbacks = []
    
    checkpoint = ModelCheckpoint(dirpath=args.ckpt_dir, save_last=True, save_top_k=1, monitor='valid_acc', mode='max', filename='{epoch:02d}--{valid_acc:.4f}')
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

    args = Model.update_output_dirs(args) 
    # Callbacks
    logger = loggers.TensorBoardLogger(save_dir=args.log_dir, name="")
    callbacks = load_callbacks(args)

    # Trainer
    trainer = Trainer.from_argparse_args(args, logger=logger, callbacks=callbacks)
    
    model = Model(args)
    data_model = DataModel(args)

    print("The following is the parameters of our model:")
    for name, param in model.named_parameters():
        print('-->name:', name, '-->grad_requirs:', param.requires_grad)

    # return

    # Training and Testing
    if args.do_train == True:
        trainer.fit(model, data_model)
        checkpoint_path = callbacks[0].best_model_path
    else:
        # 直接load最佳的checkpoint
        checkpoint_path = args.ckpt_path

    if args.do_test == True:
        print(f"Load best checkpoint from {checkpoint_path}...")
        test_results = trainer.test(model=model, ckpt_path=checkpoint_path, datamodule=data_model)

        print("test_acc:{}".format(test_results[0]["test_acc"]))


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