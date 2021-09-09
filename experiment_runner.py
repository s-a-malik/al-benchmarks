"""experiment_runner.py
Main experiment runner.
Takes config file as input and runs multiple experiments sequentially.
"""

import os
import sys
import yaml
import argparse

import wandb

from torch.utils.data import DataLoader, SubsetRandomSampler, Subset

from active_learning.data import init_data
from active_learning.al_classes import ActiveLearner, EvalModel
from active_learning import utils


def parse_args():
    """Loads config file
    Returns:
    - config (dict): Full config information for the experiment
    """

    parser = argparse.ArgumentParser(description="AL Benchmarks")
    parser.add_argument("--config",
                        type=str,
                        default="experiments/debug.yaml",
                        help="Path to config file")
    # change dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--seed_size", type=int, default=None)
    parser.add_argument("--seed_balance", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val_prop", type=float, default=None)
    parser.add_argument("--debug", type=str, default=None)
    # change query type easily
    parser.add_argument("--query_type", type=str, default=None)
    parser.add_argument("--num_subsample", type=int, default=None)
    parser.add_argument("--query_size", type=int, default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--num_mc_iters", type=int, default=None)
    parser.add_argument("--num_repetitions", type=int, default=None)
    # for sweeps
    parser.add_argument("--L2_reg", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--fine_tune", type=str, default=None)
    parser.add_argument("--head_dim", type=int, default=None)
    parser.add_argument("--hid_dim", type=int, default=None)
    parser.add_argument("--hid_dims", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # update config with sweep params
    config = utils.update_config(config, args)
    return config


def main(config):
    """Run experiment detailed in the config file
    Distributes tasks to Ray cluster.
    Logs metrics and checkpoints to wandb.
    """
    # set up directory structure and logs
    wandb_group = f"{config['experiment']['name']}_{wandb.util.generate_id()}"  # this makes it easier to average
    experiment_type = config['experiment']['type']
    results_path = "./results"      # currently not saving actual predictions
    data_path = "./data"
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    os.environ['HF_DATASETS_CACHE'] = './data'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false' 
    os.environ['WANDB_MODE'] = 'online' if config['experiment']['log'] else 'offline' 
    # repeat over multiple runs
    for run in range(config["experiment"]["num_repetitions"]):
        # init wandb
        run = wandb.init(
            entity="ucl-msc-al-benchmarks",
            project="al-benchmarks",
            group=wandb_group,
            job_type=experiment_type,
            reinit=True
        )
        wandb.config.update(config)
        print(f"Saving model checkpoints to {os.getenv('SCRATCH_DIR', wandb.run.dir)}") 

        # init dataset
        train, val, test, num_classes, labelled_idx, unlabelled_idx = init_data(config["dataset"])
        val_prop = config["dataset"]["val_prop"]
        # number of acquisition steps (counting the seed as the first one)
        num_al_batches = config["query_strategy"]["num_queries"]

        # init active learning model
        al_model = ActiveLearner(config, num_classes, labelled_idx, unlabelled_idx)
        # init eval models
        eval_models = [EvalModel(model_config, num_classes, val_prop, experiment_type, model_id=i) for (i, model_config) in enumerate(config["eval_models"])]

        al_test_loader = DataLoader(
            test,
            batch_size=al_model.optim_config["batch_size"],
            shuffle=False,
            num_workers=al_model.optim_config["num_workers"],
            collate_fn=al_model.collate
        )
        if num_al_batches == 0:
            # train and test on full dataset
            al_train_loader = DataLoader(
                    train,
                    batch_size=al_model.optim_config["batch_size"],
                    num_workers=al_model.optim_config["num_workers"],
                    collate_fn=al_model.collate
                ) 
            al_val_loader = DataLoader(
                    val,
                    batch_size=al_model.optim_config["batch_size"],
                    num_workers=al_model.optim_config["num_workers"],
                    collate_fn=al_model.collate
                ) 
            al_train_losses, al_val_losses, al_best_val_loss, al_best_val_acc = al_model.fit(al_train_loader, al_val_loader)
            # load best val loss checkpoint.
            scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
            best_file = os.path.join(scratch_dir, "acq_model_best.pth.tar")
            al_model.acq_model, al_model.optimizer, al_model.scheduler = utils.load_checkpoint(
                al_model.acq_model, al_model.optimizer, al_model.scheduler, al_model.device, best_file)
            
            # eval on test set
            al_acc, al_prec_rec_f1_sup, al_prec_rec_f1_sup_per_class, _, _ = utils.evaluate(
                al_model.acq_model,
                al_test_loader,
                al_model.optimizer,
                al_model.scheduler,
                al_model.criterion,
                al_model.device,
                task="test"
            )
            print(f"acquisition model\n"
                        f"test acc: {al_acc}, test prec: {al_prec_rec_f1_sup[0]},"
                        f"test rec: {al_prec_rec_f1_sup[1]}, test f1: {al_prec_rec_f1_sup[2]}\n")
            wandb.log({
                "num_labelled": len(train),
                "fraction_labelled": 1,
                "acq_model/test_acc": al_acc,
                "acq_model/test_prec": al_prec_rec_f1_sup[0],
                "acq_model/test_rec": al_prec_rec_f1_sup[1],
                "acq_model/test_f1": al_prec_rec_f1_sup[2],
                "acq_model/test_prec_pc": al_prec_rec_f1_sup_per_class[0],
                "acq_model/test_rec_pc": al_prec_rec_f1_sup_per_class[1],
                "acq_model/test_f1_pc": al_prec_rec_f1_sup_per_class[2],
                "acq_model/test_sup_pc": al_prec_rec_f1_sup_per_class[3],
                "acq_model/train_curve": al_train_losses,
                "acq_model/val_curve": al_val_losses,
                "acq_model/val_loss": al_best_val_loss,
                "acq_model/val_acc": al_best_val_acc,
                }, step=0)    

            # # take off gpu
            # al_model.acq_model = al_model.acq_model.cpu()

            for model_idx, model in enumerate(eval_models):
                eval_train_loader = DataLoader(
                    train,
                    batch_size=model.optim_config["batch_size"],
                    num_workers=model.optim_config["num_workers"],
                    collate_fn=model.collate
                )
                eval_val_loader = DataLoader(
                    val,
                    batch_size=model.optim_config["batch_size"],
                    shuffle=False,
                    num_workers=model.optim_config["num_workers"],
                    collate_fn=model.collate,
                )
                eval_test_loader = DataLoader(
                    test,
                    batch_size=model.optim_config["batch_size"],
                    shuffle=False,
                    num_workers=model.optim_config["num_workers"],
                    collate_fn=model.collate
                )
                eval_train_losses, eval_val_losses, eval_best_val_loss, eval_best_val_acc = model.fit(eval_train_loader, eval_val_loader)

                # load best val loss checkpoint.
                scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
                best_file = os.path.join(scratch_dir, f"eval_model_{model_idx}_best.pth.tar")
                model.eval_model, model.optimizer, model.scheduler = utils.load_checkpoint(
                    model.eval_model, model.optimizer, model.scheduler, model.device, best_file)
                # eval on test set
                eval_acc, eval_prec_rec_f1_sup, eval_prec_rec_f1_sup_per_class, _, _ = utils.evaluate(
                    model.eval_model,
                    eval_test_loader,
                    model.optimizer,
                    model.scheduler,
                    model.criterion,
                    model.device,
                    task="test"
                )

                print(f"eval model {model_idx}\n"
                        f"test acc: {eval_acc}, test prec: {eval_prec_rec_f1_sup[0]},"
                        f"test rec: {eval_prec_rec_f1_sup[1]}, test f1: {eval_prec_rec_f1_sup[2]}\n")

                wandb.log({
                    "num_labelled_val": len(val),
                    "fraction_labelled_val": 1,
                    f"eval_model_{model_idx}/test_acc": eval_acc,
                    f"eval_model_{model_idx}/test_prec": eval_prec_rec_f1_sup[0],
                    f"eval_model_{model_idx}/test_rec": eval_prec_rec_f1_sup[1],
                    f"eval_model_{model_idx}/test_f1": eval_prec_rec_f1_sup[2],
                    f"eval_model_{model_idx}/test_prec_pc": eval_prec_rec_f1_sup_per_class[0],
                    f"eval_model_{model_idx}/test_rec_pc": eval_prec_rec_f1_sup_per_class[1],
                    f"eval_model_{model_idx}/test_f1_pc": eval_prec_rec_f1_sup_per_class[2],
                    f"eval_model_{model_idx}/test_sup_pc": eval_prec_rec_f1_sup_per_class[3],
                    f"eval_model_{model_idx}/train_curve": eval_train_losses,
                    f"eval_model_{model_idx}/val_curve": eval_val_losses,
                    f"eval_model_{model_idx}/val_loss": eval_best_val_loss,
                    f"eval_model_{model_idx}/val_acc": eval_best_val_acc,
                    }, step=0)

                # # take off gpu
                # model.eval_model = model.eval_model.cpu()

        else:
            # active learning training loop
            # try, except to stop partway through 
            try:
                for al_batch_idx in range(num_al_batches):
                    # acquire data and train acquisition model
                    labelled_idx, unlabelled_idx, al_train_losses, al_val_losses, al_best_val_loss, al_best_val_acc = al_model.al_step(train, val)

                    # load best val loss checkpoint.
                    scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
                    best_file = os.path.join(scratch_dir, "acq_model_best.pth.tar")
                    al_model.acq_model, al_model.optimizer, al_model.scheduler = utils.load_checkpoint(
                        al_model.acq_model, al_model.optimizer, al_model.scheduler, al_model.device, best_file)
                    
                    # eval on test set
                    al_acc, al_prec_rec_f1_sup, al_prec_rec_f1_sup_per_class, _, _ = utils.evaluate(
                        al_model.acq_model,
                        al_test_loader,
                        al_model.optimizer,
                        al_model.scheduler,
                        al_model.criterion,
                        al_model.device,
                        task="test"
                    )

                    print(f"Acquisition step: {al_batch_idx+1}, acquisition model\n"
                            f"labelled: {len(labelled_idx)} examples, {len(labelled_idx)*100 / len(train)}%\n"
                            f"test acc: {al_acc}, test prec: {al_prec_rec_f1_sup[0]},"
                            f"test rec: {al_prec_rec_f1_sup[1]}, test f1: {al_prec_rec_f1_sup[2]}\n")

                    wandb.log({
                        "num_labelled": len(labelled_idx),
                        "fraction_labelled": len(labelled_idx) / len(train),
                        "acq_model/test_acc": al_acc,
                        "acq_model/test_prec": al_prec_rec_f1_sup[0],
                        "acq_model/test_rec": al_prec_rec_f1_sup[1],
                        "acq_model/test_f1": al_prec_rec_f1_sup[2],
                        "acq_model/test_prec_pc": al_prec_rec_f1_sup_per_class[0],
                        "acq_model/test_rec_pc": al_prec_rec_f1_sup_per_class[1],
                        "acq_model/test_f1_pc": al_prec_rec_f1_sup_per_class[2],
                        "acq_model/test_sup_pc": al_prec_rec_f1_sup_per_class[3],
                        "acq_model/train_curve": al_train_losses,
                        "acq_model/val_curve": al_val_losses,
                        "acq_model/val_loss": al_best_val_loss,
                        "acq_model/val_acc": al_best_val_acc,
                        "labelled_idx": labelled_idx
                        }, step=al_batch_idx)
                    # # take off gpu 
                    # al_model.acq_model = al_model.acq_model.cpu()

                    # train and eval all evaluation models and save to log
                    for model_idx, model in enumerate(eval_models):
                        eval_train_loader = DataLoader(
                            train,
                            batch_size=model.optim_config["batch_size"],
                            sampler=SubsetRandomSampler(labelled_idx),
                            num_workers=model.optim_config["num_workers"],
                            collate_fn=model.collate
                        )
                        # change val set size as a function of training set size.
                        val_subset = Subset(val, list(range(int(len(labelled_idx)*model.val_prop))))
                        eval_val_loader = DataLoader(
                            val_subset,
                            batch_size=model.optim_config["batch_size"],
                            shuffle=False,
                            num_workers=model.optim_config["num_workers"],
                            collate_fn=model.collate,
                        )
                        eval_test_loader = DataLoader(
                            test,
                            batch_size=model.optim_config["batch_size"],
                            shuffle=False,
                            num_workers=model.optim_config["num_workers"],
                            collate_fn=model.collate
                        )

                        eval_train_losses, eval_val_losses, eval_best_val_loss, eval_best_val_acc = model.fit(eval_train_loader, eval_val_loader)

                        # TODO hyperparameter tuning on val set?

                        # load best val loss checkpoint.
                        scratch_dir = os.getenv('SCRATCH_DIR', wandb.run.dir)   # if given a scratch dir save models here
                        best_file = os.path.join(scratch_dir, f"eval_model_{model_idx}_best.pth.tar")
                        model.eval_model, model.optimizer, model.scheduler = utils.load_checkpoint(
                            model.eval_model, model.optimizer, model.scheduler, model.device, best_file)

                        # eval on test set
                        eval_acc, eval_prec_rec_f1_sup, eval_prec_rec_f1_sup_per_class, _, _ = utils.evaluate(
                            model.eval_model,
                            eval_test_loader,
                            model.optimizer,
                            model.scheduler,
                            model.criterion,
                            model.device,
                            task="test"
                        )

                        print(f"Acquisition step: {al_batch_idx+1}, eval model {model_idx}\n"
                            f"labelled: {len(labelled_idx)}, {len(labelled_idx)*100 / len(train)}%\n"
                            f"val set size: {len(val_subset)}, {len(val_subset)*100 / len(val)}%\n"
                            f"test acc: {eval_acc}, test prec: {eval_prec_rec_f1_sup[0]}, "
                            f"test rec: {eval_prec_rec_f1_sup[1]}, test f1: {eval_prec_rec_f1_sup[2]}\n")

                        wandb.log({
                            "num_labelled_val": len(val_subset),
                            "fraction_labelled_val": len(val_subset) / len(val),
                            f"eval_model_{model_idx}/test_acc": eval_acc,
                            f"eval_model_{model_idx}/test_prec": eval_prec_rec_f1_sup[0],
                            f"eval_model_{model_idx}/test_rec": eval_prec_rec_f1_sup[1],
                            f"eval_model_{model_idx}/test_f1": eval_prec_rec_f1_sup[2],
                            f"eval_model_{model_idx}/test_prec_pc": eval_prec_rec_f1_sup_per_class[0],
                            f"eval_model_{model_idx}/test_rec_pc": eval_prec_rec_f1_sup_per_class[1],
                            f"eval_model_{model_idx}/test_f1_pc": eval_prec_rec_f1_sup_per_class[2],
                            f"eval_model_{model_idx}/test_sup_pc": eval_prec_rec_f1_sup_per_class[3],
                            f"eval_model_{model_idx}/train_curve": eval_train_losses,
                            f"eval_model_{model_idx}/val_curve": eval_val_losses,
                            f"eval_model_{model_idx}/val_loss": eval_best_val_loss,
                            f"eval_model_{model_idx}/val_acc": eval_best_val_acc,
                            }, step=al_batch_idx)
                        # # take off gpu
                        # model.eval_model = model.eval_model.cpu()
            except KeyboardInterrupt:
                pass
        run.finish()


if __name__ == "__main__":
    # load args from config
    config = parse_args()
    # run experiment
    main(config)
