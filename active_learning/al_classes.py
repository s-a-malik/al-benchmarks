"""Base model classes for Active Learning pipelines
"""

import os
import time
from typing import List
from copy import deepcopy

import wandb

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset

from active_learning.data import Collate
from active_learning.query_strats import get_query_strat
from active_learning.model_classes import get_model
from active_learning import utils


class ActiveLearner():
    """General class for model used to acquire data.
    TODO this could inherit from EvalModel. 
    """
    def __init__(self, config: dict, num_classes: int, labelled_idx: List[int], unlabelled_idx: List[int]):
        """Params:
        - config (dict): full experiment config
        - num_classes (int): number of classes for classification (currently derived from dataset)
        - labelled_idx (List(int)): indices of labelled pool examples (seeded)
        - unlabelled_idx (List(int)): indices of unlabelled pool examples
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() \
                        else torch.device("cpu")
        print(f"The model will run on the {self.device} device")
        self.config = config
        self.num_classes = num_classes
        self.experiment_type = config["experiment"]["type"]
        self.optim_config = config["acq_model"]["model_hypers"]["optim"]
        self.collate = Collate(config["acq_model"])
        # strategy
        self.query_strat = get_query_strat(
            strat_config=config["query_strategy"],
            collate=self.collate,
            batch_size=self.optim_config["batch_size"],
            num_workers=self.optim_config["num_workers"],
            device=self.device)
        
        # initialise model and save initial weights
        self.acq_model = None
        self.init_weights = None
        self.acq_model = self._init_net()
        self.optimizer, self.scheduler, self.criterion = utils.init_optim(self.optim_config, self.acq_model)
        
        # data
        self.val_prop = self.config['dataset']['val_prop']
        self.labelled_idx = labelled_idx
        self.unlabelled_idx = unlabelled_idx    # could just get via labelled pool

    def _init_net(self):
        """Randomly initialise the acquisition model
        """
        if self.acq_model is None:
            self.acq_model = get_model(self.config["acq_model"], self.experiment_type, self.num_classes)
            self.init_weights = deepcopy(self.acq_model.state_dict())
        else:
            # use cached init weights
            self.acq_model.load_state_dict(self.init_weights)
        
        self.acq_model.to(self.device)
        return self.acq_model

    def al_step(self, train_data, dev_data):
        """Carry out one iteration of Active Learning training:
        - Use AL strategy to add newly labelled data to labelled pool
        - train acq_model with labelled pool
        Params:
        - train_data (Dataset): full training pool
        - dev_data (Dataset): full dev set
        Returns:
        - labelled_idx (list): indices of pool in the labelled set
        - unlabelled_idx (list): indices of pool not in labelled set
        - train_losses (List[float]): train loss at each epoch
        - val_losses (List[float]): val loss at each epoch
        - best_loss (float): best val loss achieved
        - best_acc (float): best val acc achieved
        """
        # # put on gpu again 
        # self.acq_model.to(self.device)

        # get new indices of labelled and unlabelled pools
        print("Choosing data to label...")
        start_time = time.time()
        self.labelled_idx, self.unlabelled_idx = self.query_strat.get_query(
            self.acq_model,
            train_data,
            self.labelled_idx,
            self.unlabelled_idx
        )
        print(f"Chosen new labels. Time taken: {time.time()-start_time} seconds")

        # train on labelled pool only 
        train_loader = DataLoader(
            train_data,
            batch_size=self.optim_config["batch_size"],
            sampler=SubsetRandomSampler(self.labelled_idx),
            num_workers=self.optim_config["num_workers"],
            collate_fn=self.collate
        )

        # Change val set size as a function of training set size.
        val_subset = Subset(dev_data, list(range(int(len(self.labelled_idx)*self.val_prop))))
        val_loader = DataLoader(
            val_subset,
            batch_size=self.optim_config["batch_size"],
            shuffle=False,
            num_workers=self.optim_config["num_workers"],
            collate_fn=self.collate,
        )

        train_losses, val_losses, best_loss, best_acc = self.fit(train_loader, val_loader)

        return self.labelled_idx, self.unlabelled_idx, train_losses, val_losses, best_loss, best_acc

    def fit(self, train_loader, val_loader):
        """Train on current labelled pool and log training curves.
        Params:
        - train_loader (DataLoader): dataloader for batches from labelled pool
        - val_loader (DataLoader): dataloader for batches from val dataset
        Returns:
        - train_losses (List[float]): train loss at each epoch
        - val_losses (List[float]): val loss at each epoch
        - best_loss (float): best val loss achieved
        - best_acc (float): best val acc achieved
        """

        # set up directories and logs for this training run
        scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
        checkpoint_file = os.path.join(scratch_dir, "acq_model_ckpt.pth.tar")
        best_file = os.path.join(scratch_dir, "acq_model_best.pth.tar")  
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # reinitialise network and optimizer
        self.acq_model = self._init_net()
        self.optimizer, self.scheduler, self.criterion = utils.init_optim(self.optim_config, self.acq_model)

        # TODO optionally train online (without reinitialising) 

        # initial validation loss
        best_loss, best_acc = utils.evaluate(
            self.acq_model,
            val_loader,
            self.optimizer,
            self.scheduler,
            self.criterion,
            self.device,
            task="val"
        )
        best_epoch = 0
        # initially save first as best
        checkpoint_dict = {"epoch": best_epoch,
                            "state_dict": self.acq_model.state_dict(),
                            "best_loss": best_loss,
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "config": self.config}
        torch.save(checkpoint_dict, best_file)
        # try, except to stop training partway through
        try: 
            # training loop
            print("\nFitting acquisition model...")
            for epoch in range(self.optim_config["epochs"]):
                # train
                train_loss, train_acc = utils.evaluate(
                    self.acq_model,
                    train_loader,
                    self.optimizer,
                    self.scheduler,
                    self.criterion,
                    self.device,
                    task="train"
                )
                # validation
                with torch.no_grad():
                    val_loss, val_acc = utils.evaluate(
                        self.acq_model,
                        val_loader,
                        self.optimizer,
                        self.scheduler,
                        self.criterion,
                        self.device,
                        task="val"
                    )

                print(f"\nEpoch {epoch+1}/{self.optim_config['epochs']}: \ntrain/loss: {train_loss}, train/acc: {train_acc}"
                      f"\nval/loss: {val_loss}, val/acc: {val_acc}\n")

                # TODO log other metrics on train and val set while training?
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_accs)

                # model checkpointing
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    best_acc = val_acc
                    best_epoch = epoch
                    # save checkpoint if best
                    checkpoint_dict = {"epoch": epoch,
                                       "state_dict": self.acq_model.state_dict(),
                                       "best_loss": best_loss,
                                       "optimizer": self.optimizer.state_dict(),
                                       "scheduler": self.scheduler.state_dict(),
                                       "config": self.config}
                    torch.save(checkpoint_dict, best_file)

                # early stopping
                if epoch - best_epoch > self.optim_config["patience"]:
                    break
        except KeyboardInterrupt:
            pass

        # save last checkpoint
        checkpoint_dict = {"epoch": epoch,
                           "state_dict": self.acq_model.state_dict(),
                           "best_loss": best_loss,
                           "optimizer": self.optimizer.state_dict(),
                           "scheduler": self.scheduler.state_dict(),
                           "config": self.config}
        torch.save(checkpoint_dict, checkpoint_file)

        return train_losses, val_losses, best_loss, best_acc


class EvalModel():
    """Generic evaluation model to wrap over neural networks to ease training and eval.
    TODO this could be a base class of ActiveLearner. fit and _init_net methods are the same.
    """
    def __init__(self, model_config: dict, num_classes: int, val_prop: int, experiment_type: str, model_id: int):
        self.device = torch.device("cuda") if torch.cuda.is_available() \
                        else torch.device("cpu")
        self.config = model_config
        self.num_classes = num_classes
        self.experiment_type = experiment_type
        self.model_id = model_id
        self.collate = Collate(model_config)
        self.optim_config = self.config["model_hypers"]["optim"]
        self.val_prop = val_prop
        self.eval_model = None
        self.init_weights = None
        self.eval_model = self._init_net()
        self.optimizer, self.scheduler, self.criterion = utils.init_optim(self.optim_config, self.eval_model) 

    def _init_net(self):
        """Randomly initialise the evaluation model
        """
        if self.eval_model is None:
            self.eval_model = get_model(self.config, self.experiment_type, self.num_classes)
            self.init_weights = deepcopy(self.eval_model.state_dict())
        else:
            # use cached init weights
            self.eval_model.load_state_dict(self.init_weights)
        self.eval_model.to(self.device)
        return self.eval_model

    def fit(self, train_loader, val_loader):
        """Train on current labelled pool.
        Params:
        - train_loader (DataLoader): dataloader for batches from labelled pool
        - val_loader (DataLoader): dataloader for batches from val dataset
        Returns:
        - train_losses (List[float]): train loss at each epoch
        - val_losses (List[float]): val loss at each epoch
        - best_loss (float): best val loss achieved
        - best_acc (float): best val acc achieved
        """
        # set up directories and logs for this training run
        scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
        checkpoint_file = os.path.join(scratch_dir, f"eval_model_{self.model_id}_ckpt.pth.tar")
        best_file = os.path.join(scratch_dir, f"eval_model_{self.model_id}_best.pth.tar")
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # reinitialise network and optimizer
        self.eval_model = self._init_net()
        self.optimizer, self.scheduler, self.criterion = utils.init_optim(self.optim_config, self.eval_model)

        best_loss, best_acc = utils.evaluate(
            self.eval_model,
            val_loader,
            self.optimizer,
            self.scheduler,
            self.criterion,
            self.device,
            task="val"
        )
        best_epoch = 0
        checkpoint_dict = {"epoch": best_epoch,
                            "state_dict": self.eval_model.state_dict(),
                            "best_loss": best_loss,
                            "optimizer": self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "config": self.config}
        torch.save(checkpoint_dict, best_file)

        try:
            # training loop
            print(f"\nFitting eval model {self.model_id}...\n")
            for epoch in range(self.optim_config["epochs"]):
                
                # train
                train_loss, train_acc = utils.evaluate(
                    self.eval_model,
                    train_loader,
                    self.optimizer,
                    self.scheduler,
                    self.criterion,
                    self.device,
                    task="train"
                )

                # validation
                with torch.no_grad():
                    val_loss, val_acc = utils.evaluate(
                        self.eval_model,
                        val_loader,
                        self.optimizer,
                        self.scheduler,
                        self.criterion,
                        self.device,
                        task="val"
                    )

                print(f"\nEpoch {epoch+1}/{self.optim_config['epochs']}: \ntrain/loss: {train_loss}, train/acc: {train_acc}"
                        f"\nval/loss: {val_loss}, val/acc: {val_acc}\n")

                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                # model checkpointing
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    best_acc = val_acc
                    best_epoch = epoch

                    # save checkpoint if best
                    checkpoint_dict = {"epoch": epoch,
                                       "state_dict": self.eval_model.state_dict(),
                                       "best_loss": best_loss,
                                       "optimizer": self.optimizer.state_dict(),
                                       "scheduler": self.scheduler.state_dict(),
                                       "config": self.config}
                    torch.save(checkpoint_dict, best_file)
                # if is_best or epoch == 0:
                #     shutil.copyfile(checkpoint_file, best_file)

                # early stopping
                if epoch - best_epoch > self.optim_config["patience"]:
                    break
        except KeyboardInterrupt:
            pass

        # save last checkpoint
        checkpoint_dict = {"epoch": epoch,
                           "state_dict": self.eval_model.state_dict(),
                           "best_loss": best_loss,
                           "optimizer": self.optimizer.state_dict(),
                           "scheduler": self.scheduler.state_dict(),
                           "config": self.config}
        torch.save(checkpoint_dict, checkpoint_file)
        return train_losses, val_losses, best_loss, best_acc
