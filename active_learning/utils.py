"""General purpose utility functions and classes
"""

import ast
from torch.utils import data
from tqdm.autonotebook import trange

import torch
import torch.nn as nn

import transformers

from active_learning import metrics


class AverageMeter(object):
    """Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate(model, generator, optimizer, scheduler, criterion, device, task="train"):
    """Generic evaluation function for model. Iterates through generator once (1 epoch)
    Params:
    - model (nn.Module): model to eval
    - generator (DataLoader): data to eval on
    - optimizer (optim): torch optimizer for model
    - scheduler: LR scheduler
    - criterion: loss function
    - device: Cuda/CPU 
    - task (str) = [train, val, test]: determines whether to take gradient steps etc.
    Returns:
    if test: results (test_pred, test_true, test_f1, test_prec, test_rec)
    if train/val: average loss and accuracy over batch
    """
    # TODO generic enough to use for all models? Might need to add "experiment_type" argument

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if task == "test":
        model.eval()
        test_true = []
        test_pred = []
    elif task == "val":
        model.eval()
    elif task == "train":
        model.train()
    else:
        raise NameError("Only train, val or test is allowed as task")

    with trange(len(generator)) as t:
        for batch in generator:
            x, y = batch
            # TODO would need all inputs as a dict (general enough?)
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            output = model(x)
            # predictions
            preds = torch.argmax(output, dim=-1)
            acc = metrics.accuracy(y, preds)
            acc_meter.update(acc.data.cpu().item(), y.size(0))

            if task == "test":
                # collect the model outputs
                test_true += y.detach().cpu().tolist()
                test_pred += preds.detach().cpu().tolist()
            else:
                # compute loss
                loss = criterion(output, y)
                loss_meter.update(loss.data.cpu().item(), y.size(0))
                if task == "train":
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

            t.update()

    if task == "test":
        prec_rec_f1_sup = metrics.cls_metrics(test_true, test_pred, num_classes=model.num_classes, average="macro")
        # per class metrics
        prec_rec_f1_sup_per_class = metrics.cls_metrics(test_true, test_pred, num_classes=model.num_classes, average=None)

        return acc_meter.avg, prec_rec_f1_sup, prec_rec_f1_sup_per_class, test_pred, test_true
    else:
        return loss_meter.avg, acc_meter.avg


def init_optim(optim_config, net):
    """initialise the optimizer and loss function of a network given the config
    Params:
    - optim_config (dict): optimiser config
    - net (nn.Module): model to optimise
    """
    if optim_config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(net.parameters(), 
                                     lr=optim_config["lr"], 
                                     weight_decay=optim_config["L2_reg"])
    elif optim_config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=optim_config["lr"],
                                    weight_decay=optim_config["L2_reg"],
                                    momentum=optim_config["momentum"])
    elif optim_config["optimizer"] == "adamw":
        optimizer = transformers.AdamW(net.parameters(),
                                       lr=optim_config["lr"],
                                       weight_decay=optim_config["L2_reg"])
    else:
        NotImplementedError()

    if optim_config["loss"] == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif optim_config["loss"] == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        NotImplementedError()

    # lr scheduler
    if "scheduler" in optim_config:
        warmup = optim_config.get("warmup_steps", 0)

        if optim_config["scheduler"] == "linear_warm_up":
            # TODO how to choose num_training_steps? Dependant on batch size, num epochs and data size.
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup,
                num_training_steps=optim_config.get("training_steps", optim_config["epochs"]*100)
            )
        elif optim_config["scheduler"] == "constant_warm_up":
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup
            )
        # TODO exp decay schedule
        elif optim_config["scheduler"] == "constant":
            scheduler = transformers.get_constant_schedule(optimizer)
        else:
            raise NotImplementedError()
    else:
        # if no scheduler defined, default is constant.
        scheduler = transformers.get_constant_schedule(optimizer)

    return optimizer, scheduler, criterion


def load_checkpoint(model, optimizer, scheduler, device, checkpoint_file: str):
    """Loads a model checkpoint.
    Params:
    - model (nn.Module): initialised model
    - optimizer (nn.optim): initialised optimizer
    - scheduler: initialised scheduler
    - device (torch.device): device model is on
    Returns:
    - model with loaded state dict
    - optimizer with loaded state dict
    - scheduler with loaded state dict
    """
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    print(f"Loaded {checkpoint_file}, "
          f"trained to epoch {checkpoint['epoch']+1} with best loss {checkpoint['best_loss']}")

    return model, optimizer, scheduler


def update_config(config, args):
    """Update config dict with command line arguments
    Params:
    - config (dict): default config dict
    - args (argparse): command line arguments
    Returns:
    - config (dict): updated config dict
    """
    if args.log is not None:
        config["experiment"]["log"] = True if args.log == "true" else False    
    
    # if using a cluster config but normal experiment_runner, default to 1 run
    if "num_repetitions" not in config["experiment"]:
        config["experiment"]["num_repetitions"] = args.num_repetitions if args.num_repetitions is not None else 1

    # change dataset
    data_config = config["dataset"]
    data_config["name"] = args.dataset if args.dataset is not None else data_config["name"]
    data_config["seed_size"] = args.seed_size if args.seed_size is not None else data_config["seed_size"]
    data_config["seed_balance"] = args.seed_balance if args.seed_balance is not None else data_config["seed_balance"]
    data_config["pool_balance"] = args.pool_balance if args.pool_balance is not None else data_config["pool_balance"]
    data_config["dev_balance"] = args.dev_balance if args.dev_balance is not None else data_config["dev_balance"]
    data_config["imbalance_prop"] = args.imbalance_prop if args.imbalance_prop is not None else data_config.get("imbalance_prop")
    data_config["imbalance_cls"] = args.imbalance_cls if args.imbalance_cls is not None else data_config.get("imbalance_cls")
    data_config["seed"] = args.seed if args.seed is not None else data_config["seed"]
    data_config["val_prop"] = args.val_prop if args.val_prop is not None else data_config["val_prop"]
    data_config["debug"] = True if (args.debug == "true") else data_config["debug"]

    # easily change query strategy
    query_config = config["query_strategy"]
    query_config["query_type"] = args.query_type if args.query_type is not None else query_config["query_type"]
    if args.num_subsample is not None:
        # prevents subsample key being created if not used
        query_config["num_subsample"] = args.num_subsample
    query_config["query_size"] = args.query_size if args.query_size is not None else query_config["query_size"]
    query_config["num_queries"] = args.num_queries if args.num_queries is not None else query_config["num_queries"]
    query_config["num_mc_iters"] = args.num_mc_iters if args.num_mc_iters is not None else query_config["num_mc_iters"]

    # model hypers (for sweep)
    hyper_config = config["acq_model"]["model_hypers"]
    hyper_config["optim"]["batch_size"] = args.batch_size if args.batch_size is not None else hyper_config["optim"]["batch_size"]
    hyper_config["optim"]["lr"] = args.lr if args.lr is not None else hyper_config["optim"]["lr"]
    hyper_config["optim"]["L2_reg"] = args.L2_reg if args.L2_reg is not None else hyper_config["optim"]["L2_reg"]
    hyper_config["optim"]["scheduler"] = args.scheduler if args.scheduler is not None else hyper_config["optim"]["scheduler"]
    hyper_config["optim"]["optimizer"] = args.optimizer if args.optimizer is not None else hyper_config["optim"]["optimizer"]
    hyper_config["optim"]["warmup_steps"] = args.warmup_steps if args.warmup_steps is not None else hyper_config["optim"]["warmup_steps"]
    if "dropout" in hyper_config["architecture"]:
        hyper_config["architecture"]["dropout"] = args.dropout if args.dropout is not None else hyper_config["architecture"]["dropout"]
    if "hid_dim" in hyper_config["architecture"]:
        hyper_config["architecture"]["hid_dim"] = args.hid_dim if args.hid_dim is not None else hyper_config["architecture"]["hid_dim"]
    if "hid_dims" in hyper_config["architecture"]:
        hyper_config["architecture"]["hid_dims"] = ast.literal_eval(args.hid_dims) if args.hid_dims is not None else hyper_config["architecture"]["hid_dims"]
    if "head_dim" in hyper_config["architecture"]:
        hyper_config["architecture"]["head_dim"] = args.head_dim if args.head_dim is not None else hyper_config["architecture"]["head_dim"]
    if "fine_tune" in hyper_config["architecture"]:
        fine_tune = args.fine_tune if args.fine_tune is not None else hyper_config["architecture"]["fine_tune"]
        hyper_config["architecture"]["fine_tune"] = True if (fine_tune == "true") else hyper_config["architecture"]["fine_tune"]

    # update config
    config["query_strategy"] = query_config
    config["dataset"] = data_config
    config["acq_model"]["model_hypers"] = hyper_config

    # change experiment name if params changed
    if args.name:
        config["experiment"]["name"] = args.name
    else:
        if args.query_type:
            config["experiment"]["name"] = f"{args.query_type}-{config['experiment']['name']}"
        if args.imbalance_prop:
            config["experiment"]["name"] = f"imbprop-{args.imbalance_prop}-{config['experiment']['name']}"
        if args.imbalance_cls:
            config["experiment"]["name"] = f"imbcls-{args.imbalance_cls}-{config['experiment']['name']}"

    return config
