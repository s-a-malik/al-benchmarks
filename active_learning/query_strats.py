"""Active learning query strategy classes.
TODO lots of repeated code here, to make a base class (or use an existing framework)
and function wrappers/checks (c.f. baal https://baal.readthedocs.io/en/stable/index.html)
"""

from tqdm.autonotebook import trange
import random

import numpy as np

from scipy.special import xlogy, softmax

from sklearn.metrics import pairwise_distances

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from baal import ModelWrapper
from baal.bayesian.dropout import patch_module
from baal.active.heuristics import BALD, BatchBALD

from active_learning.metrics import get_metrics
# from metrics import get_metrics


def get_query_strat(strat_config, collate, batch_size, num_workers, device):
    """Use config to return the appropriate acquisition strategy class
    """
    if strat_config["query_type"] == "random":
        return RandomStrat(strat_config)
    elif strat_config["query_type"] in ["lc", "entropy", "margin"]:
        return UncertaintySamplingStrat(strat_config, collate, batch_size, num_workers, device)
    elif strat_config["query_type"] in ["mc-lc", "mc-entropy"]:
        return MCDropoutUncertaintySamplingStrat(strat_config, collate, batch_size, num_workers, device)
    elif strat_config["query_type"] == "bald": 
        return BALDStrat(strat_config, collate, batch_size, num_workers, device)
    elif strat_config["query_type"] == "batchbald": 
        return BatchBALDStrat(strat_config, collate, batch_size, num_workers, device)
    elif strat_config["query_type"] == "coreset": 
        return CoreSetStrat(strat_config, collate, batch_size, num_workers, device)
    else:
        raise NotImplementedError(f'query strat {strat_config["query_type"]} not implemented')
    
    # TODO DAL

    # TODO BADGE

    # TODO Posterior entropy


class CoreSetStrat():
    """CoreSet query strategy
    TODO this could be integrated with UncertaintyStrat
    """
    def __init__(self, strat_config, collate, batch_size, num_workers, device):
        """Params:
        - strat_config (dict): query strategy config
        - collate (class): collate function for pool dataloader
        - batch_size (int): batch size for pool dataloader
        - device: cuda/cpu for model
        """
        self.num_query = strat_config["query_size"]
        if "num_subsample" in strat_config:
            self.num_subsample = strat_config["num_subsample"]
        else:
            self.num_subsample = None
        self.collate = collate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.num_steps = 0
        self.min_distances = None

    def score(self, model, pool, labelled):
        """Params:
        - model (nn.Module): acquisition model
        - pool (Dataset): unlabelled training dataset
        - labelled (Dataset): labelled training dataset
        Returns:
        - indices (np.array): indices to add from unlabelled_pool
        """
        # TODO this could be refactored into functions
        pool_loader = DataLoader(
            pool,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )
        labelled_loader = DataLoader(
            labelled,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )
        pool_features = []
        labelled_features = []
        # get embeddings
        model.eval()
        with torch.no_grad():
            with trange(len(pool_loader)) as t:
                for batch in pool_loader:
                    x, _ = batch
                    x = {k: v.to(self.device) for k, v in x.items()}
                    _, feats = model(x, feats=True)
                    pool_features += feats.detach().cpu().tolist()
                    t.update()
            with trange(len(labelled_loader)) as t:
                for batch in labelled_loader:
                    x, _ = batch
                    x = {k: v.to(self.device) for k, v in x.items()}
                    _, feats = model(x, feats=True)
                    labelled_features += feats.detach().cpu().tolist()
                    t.update()
        pool_features = np.array(pool_features)
        labelled_features = np.array(labelled_features)
        # compute coreset
        indices = self.get_coreset(pool_features, labelled_features)
        return indices

    def get_coreset(self, pool_features, labelled_features):
        """Returns indices of points that minimizes the maximum distance of any point to a center.
        Ref: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
        Implements the k-Center-Greedy method in
        Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
        Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017
        Distance metric defaults to l2 distance.
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        TODO Can be extended to a robust k centers algorithm that ignores a certain number of
        outlier datapoints.  Resulting centers are solution to multiple integer program.
        Params:
        - pool_features (np.array): embedded unlabelled pool data
        - labelled_features (np.array): embedded labelled pool data
        Returns:
        - new_batch (list): indices of points selected to minimize distance to cluster centers
        """
        new_batch = []
        self.update_distances(pool_features, labelled_features, reset_dist=True)
        for _ in range(self.num_query):
            # choose furthest point
            ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in new_batch
            # update distances with this point
            self.update_distances(pool_features, pool_features[ind, :].reshape(1,-1), reset_dist=False)
            new_batch.append(ind)
        print(f"Maximum distance from cluster centers is {max(self.min_distances)}")

        return new_batch

    def update_distances(self, pool_features, labelled_features, reset_dist=False):
        """Update min distances given cluster centers.
        Params:
        - pool_features: unlabelled data points
        - labelled_features: labelled data points (cluster centers)
        - reset_dist: whether to reset min_distances.
        """
        if reset_dist:
            self.min_distances = None
        # Update min_distances for unlabelled examples given new cluster center.
        dist = pairwise_distances(pool_features, labelled_features, metric='euclidean', force_all_finite=True)
        if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
        else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def get_query(self, model, train_data, labelled_idx, unlabelled_idx):
        """Return updated labelled pool (indices)
        Params:
        - model (nn.Module): model used to acquire data
        - train_data (Dataset): full training data
        - labelled_idx (list): indices of labelled datapoints in train_data
        - unlabelled_idx (list): indices of unlabelled pool
        """
        self.num_steps += 1
        # if this is the first step, then just return the seed set
        if self.num_steps == 1:
            return labelled_idx, unlabelled_idx

        if self.num_subsample is not None:
            num_subsample = min(self.num_subsample, len(unlabelled_idx))
            subsample_idx = random.sample(unlabelled_idx, k=num_subsample)
        else:
            subsample_idx = unlabelled_idx

        pool = Subset(train_data, subsample_idx)
        labelled = Subset(train_data, labelled_idx)
        # get ranking of datapoints
        idx_to_add = self.score(model, pool, labelled)

        # choose top scoring datapoints to label
        new_labelled_idx = labelled_idx + [subsample_idx[i] for i in idx_to_add]
        new_unlabelled_idx = [j for j in range(len(train_data)) if j not in new_labelled_idx]

        return new_labelled_idx, new_unlabelled_idx


class BatchBALDStrat():
    """BatchBALD query strategy
    https://arxiv.org/abs/1906.08158 
    """
    def __init__(self, strat_config, collate, batch_size, num_workers, device):
        """Params:
        - strat_config (dict): query strategy config
        - collate (class): collate function for pool dataloader
        - batch_size (int): batch size for pool dataloader
        - device: cuda/cpu for model
        """
        self.num_query = strat_config["query_size"]
        if "num_subsample" in strat_config:
            self.num_subsample = strat_config["num_subsample"]
        else:
            self.num_subsample = None
        self.num_mc_iters = strat_config["num_mc_iters"]
        self.collate = collate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.num_steps = 0

    def score(self, model, pool_dataset):
        """Params:
        - model (nn.Module): acquisition model
        - pool_dataset (Dataset): unlabelled training dataset
        Returns:
        - indices (np.array): indices from pool in batch
        """
        # patch module to use dropout at test time
        mc_model = patch_module(model, inplace=False)
        wrapper = ModelWrapper(mc_model, criterion=None, replicate_in_memory=False)
        # num_draw is number to draw from history, paper suggests 40000 // num_classes
        # num_draws = 40000//model.num_classes
        num_draws = 1000
        heuristic = BatchBALD(num_samples=self.num_query, num_draw=num_draws)

        # Shape: (len(pool), num_classes, num_mc_iters)
        predictions = wrapper.predict_on_dataset(
            pool_dataset,
            batch_size=self.batch_size,
            iterations=self.num_mc_iters,
            use_cuda=False if self.device.type == "cpu" else True,
            workers=self.num_workers,
            collate_fn=self.collate)
        # need to have full prediction pool here (cannot split it up)
        indices = heuristic.get_uncertainties(predictions)

        return indices

    def get_query(self, model, train_data, labelled_idx, unlabelled_idx):
        """Return updated labelled pool (indices)
        Params:
        - model (nn.Module): model used to acquire data
        - train_data (Dataset): full training data
        - labelled_idx (list): indices of labelled datapoints in train_data
        - unlabelled_idx (list): indices of unlabelled pool
        """
        self.num_steps += 1
        # if this is the first step, then just return the seed set
        if self.num_steps == 1:
            return labelled_idx, unlabelled_idx
        
        if self.num_subsample is not None:
            num_subsample = min(self.num_subsample, len(unlabelled_idx))
            subsample_idx = random.sample(unlabelled_idx, k=num_subsample)
        else:
            subsample_idx = unlabelled_idx
        # initialise dataloader. Loads data in order of unlabelled idx
        pool = Subset(train_data, subsample_idx)

        # get ranking of unlabelled datapoints from batchBALD
        ranking = self.score(model, pool)

        # choose top scoring datapoints to label
        num_query = min(self.num_query, len(subsample_idx))
        idx_to_add = ranking[:num_query]    # take in order given
        new_labelled_idx = labelled_idx + [subsample_idx[i] for i in idx_to_add]
        new_unlabelled_idx = [j for j in range(len(train_data)) if j not in new_labelled_idx]
        return new_labelled_idx, new_unlabelled_idx


class BALDStrat():
    """BALD query strategy
    https://arxiv.org/pdf/1112.5745.pdf
    """
    def __init__(self, strat_config, collate, batch_size, num_workers, device):
        """Params:
        - strat_config (dict): query strategy config
        - train_size (int): size of training data pool
        - collate (class): collate function for pool dataloader
        - batch_size (int): batch size for pool dataloader
        - device: cuda/cpu for model
        """
        self.num_query = strat_config["query_size"]
        if "num_subsample" in strat_config:
            self.num_subsample = strat_config["num_subsample"]
        else:
            self.num_subsample = None
        self.num_mc_iters = strat_config["num_mc_iters"]
        self.collate = collate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.num_steps = 0

    def score(self, model, pool_dataset):
        """Params:
        - model (nn.Module): acquisition model
        - pool_dataset (Dataset): unlabelled training dataset
        Returns:
        - scores (np.array): acquisition score for each data point in unlabelled_pool
        """
        # patch module to use dropout at test time
        mc_model = patch_module(model, inplace=False)
        wrapper = ModelWrapper(mc_model, criterion=None, replicate_in_memory=False)
        heuristic = BALD()

        # Shape: (len(pool), num_classes, num_mc_iters)
        predictions = wrapper.predict_on_dataset(
            pool_dataset,
            batch_size=self.batch_size,
            iterations=self.num_mc_iters,
            use_cuda=False if self.device.type == "cpu" else True,
            workers=self.num_workers,
            collate_fn=self.collate)
        # compute BALD score
        scores = heuristic.get_uncertainties(predictions)

        return scores

    def get_query(self, model, train_data, labelled_idx, unlabelled_idx):
        """Return updated labelled pool (indices)
        Params:
        - model (nn.Module): model used to acquire data
        - train_data (Dataset): full training data
        - labelled_idx (list): indices of labelled datapoints in train_data
        - unlabelled_idx (list): indices of unlabelled pool
        """
        self.num_steps += 1
        # if this is the first step, then just return the seed set
        if self.num_steps == 1:
            return labelled_idx, unlabelled_idx

        if self.num_subsample is not None:
            num_subsample = min(self.num_subsample, len(unlabelled_idx))
            subsample_idx = random.sample(unlabelled_idx, k=num_subsample)
        else:
            subsample_idx = unlabelled_idx
        # initialise dataloader. Loads data in order of unlabelled idx
        pool = Subset(train_data, subsample_idx)

        # get scores on unlabelled datapoints
        scores = self.score(model, pool)
        # TODO get some metrics on the scores/plot?

        # choose top scoring datapoints to label
        num_query = min(self.num_query, len(subsample_idx))
        idx_to_add = np.argsort(scores)[-num_query:]
        new_labelled_idx = labelled_idx + [subsample_idx[i] for i in idx_to_add]
        new_unlabelled_idx = [j for j in range(len(train_data)) if j not in new_labelled_idx]
        return new_labelled_idx, new_unlabelled_idx


class MCDropoutUncertaintySamplingStrat():
    """Uncertainty sampling with expected probabilities over MC iterations 
    """
    def __init__(self, strat_config, collate, batch_size, num_workers, device):
        """Params:
        - strat_config (dict): query strategy config
        - train_size (int): size of training data pool
        - collate (class): collate function for pool dataloader
        - batch_size (int): batch size for pool dataloader
        - device: cuda/cpu for model
        """
        self.score_fn = get_metrics(strat_config["query_type"][4:]) # assuming query_type is "mc-<normal_query>"
        self.num_query = strat_config["query_size"]
        if "num_subsample" in strat_config:
            self.num_subsample = strat_config["num_subsample"]
        else:
            self.num_subsample = None
        self.num_mc_iters = strat_config["num_mc_iters"]
        self.collate = collate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.num_steps = 0

    def score(self, model, pool_dataset):
        """Params:
        - model (nn.Module): acquisition model
        - pool_dataset (Dataset): unlabelled training dataset
        Returns:
        - scores (np.array): acquisition score for each data point in unlabelled_pool
        """
        # patch module to use dropout at test time
        mc_model = patch_module(model, inplace=False)
        wrapper = ModelWrapper(mc_model, criterion=None, replicate_in_memory=False)

        # Shape: (len(pool), num_classes, num_mc_iters)
        predictions = wrapper.predict_on_dataset(
            pool_dataset,
            batch_size=self.batch_size,
            iterations=self.num_mc_iters,
            use_cuda=False if self.device.type == "cpu" else True,
            workers=self.num_workers,
            collate_fn=self.collate
        )
        # convert to probabilities TODO make this more general
        predictions = softmax(predictions, 1)
        # average over the MC iters
        expected_p = np.mean(predictions, axis=-1)
        # compute score
        scores = self.score_fn(expected_p)

        return scores

    def get_query(self, model, train_data, labelled_idx, unlabelled_idx):
        """Return updated labelled pool (indices)
        Params:
        - model (nn.Module): model used to acquire data
        - train_data (Dataset): full training data
        - labelled_idx (list): indices of labelled datapoints in train_data
        - unlabelled_idx (list): indices of unlabelled pool
        """
        self.num_steps += 1
        # if this is the first step, then just return the seed set
        if self.num_steps == 1:
            return labelled_idx, unlabelled_idx

        if self.num_subsample is not None:
            num_subsample = min(self.num_subsample, len(unlabelled_idx))
            subsample_idx = random.sample(unlabelled_idx, k=num_subsample)
        else:
            subsample_idx = unlabelled_idx
        # initialise dataloader. Loads data in order of unlabelled idx
        pool = Subset(train_data, subsample_idx)

        # get scores on unlabelled datapoints
        scores = self.score(model, pool)

        # choose top scoring datapoints to label
        num_query = min(self.num_query, len(subsample_idx))
        idx_to_add = np.argsort(scores)[-num_query:]
        new_labelled_idx = labelled_idx + [subsample_idx[i] for i in idx_to_add]
        new_unlabelled_idx = [j for j in range(len(train_data)) if j not in new_labelled_idx]
        return new_labelled_idx, new_unlabelled_idx


class UncertaintySamplingStrat():
    """General non-bayesian uncertainty sampling strategy
    TODO could also get this in the BALD framework
    """
    def __init__(self, strat_config, collate, batch_size, num_workers, device):
        """Params:
        - strat_config (dict): query strategy config
        - collate (class): collate function for pool dataloader
        - batch_size (int): batch size for pool dataloader
        - num_workers (int): number of dataloader workers
        - device: cuda/cpu for model
        """
        self.score_fn = get_metrics(strat_config["query_type"])
        self.num_query = strat_config["query_size"]
        if "num_subsample" in strat_config:
            self.num_subsample = strat_config["num_subsample"]
        else:
            self.num_subsample = None
        self.collate = collate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.num_steps = 0

    def score(self, model, pool):
        """Params:
        - model (nn.Module): acquisition model
        - pool_loader (Dataset): unlabelled training dataset
        Returns:
        - scores (np.array): acquisition score for each data point in unlabelled_pool
        """
        pool_loader = DataLoader(
            pool,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate
        )
        outputs = []
        # get model outputs
        model.eval()
        with torch.no_grad():
            with trange(len(pool_loader)) as t:
                for batch in pool_loader:
                    x, _ = batch
                    x = {k: v.to(self.device) for k, v in x.items()}
                    output = model(x)
                    # softmax to get probabilities
                    output = F.softmax(output, dim=-1)
                    outputs += output.detach().cpu().tolist()
                    t.update()
        outputs = np.array(outputs)
        # compute score
        scores = self.score_fn(outputs)
        return scores

    def get_query(self, model, train_data, labelled_idx, unlabelled_idx):
        """Return updated labelled pool (indices)
        Params:
        - model (nn.Module): model used to acquire data
        - train_data (Dataset): full training data
        - labelled_idx (list): indices of labelled datapoints in train_data
        - unlabelled_idx (list): indices of unlabelled pool
        """
        # if this is the first step, then just return the seed set
        self.num_steps += 1
        if self.num_steps == 1:
            return labelled_idx, unlabelled_idx

        if self.num_subsample is not None:
            num_subsample = min(self.num_subsample, len(unlabelled_idx))
            subsample_idx = random.sample(unlabelled_idx, k=num_subsample)
        else:
            subsample_idx = unlabelled_idx
        # initialise dataloader. Loads data in order of unlabelled idx
        pool = Subset(train_data, subsample_idx)
        # get scores on unlabelled datapoints
        scores = self.score(model, pool)

        # choose top scoring datapoints to label
        num_query = min(self.num_query, len(subsample_idx))
        idx_to_add = np.argsort(scores)[-num_query:]
        new_labelled_idx = labelled_idx + [subsample_idx[i] for i in idx_to_add]
        new_unlabelled_idx = [j for j in range(len(train_data)) if j not in new_labelled_idx]
        return new_labelled_idx, new_unlabelled_idx


class RandomStrat():
    """Random query strategy.
    """
    def __init__(self, strat_config: dict):
        """Params:
        - strat_config (dict): query strategy config
        - train_size (int): size of training data pool
        """
        self.num_query = strat_config["query_size"]
        self.num_steps = 0

    def get_query(self, model, train_data, labelled_idx, unlabelled_idx):
        """Return updated labelled pool (indices)
        Params:
        - model (nn.Module): model used to acquire data
        - train_data (Dataset): full training data
        - labelled_idx (list): indices of labelled datapoints in train_data
        - unlabelled_idx (list): indices of unlabelled pool
        """
        # if this is the first step, then just return the seed set
        self.num_steps += 1
        if self.num_steps == 1:
            return labelled_idx, unlabelled_idx

        # sample randomly from unlabelled_idx
        num_query = min(self.num_query, len(unlabelled_idx))
        new_labelled_idx = labelled_idx + np.random.choice(unlabelled_idx, size=num_query, replace=False).tolist()
        # TODO inefficient?
        new_unlabelled_idx = [j for j in range(len(train_data)) if j not in new_labelled_idx]
        return new_labelled_idx, new_unlabelled_idx


if __name__ == "__main__":   
    import torch.nn as nn

    # dropout test
    model = nn.Sequential(
            nn.Linear(10, 20), 
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(10,4))
    inputs = torch.rand((3, 10))
    model.eval()    # dropout removed
    print(model(inputs))
    print(model(inputs))
    # dropout_model = patch_module(model)     # acts in place, need to specific inplace=False
    dropout_model = patch_module(model, inplace=False) 
    dropout_model.eval()
    print(model(inputs))
    print(model(inputs))
    print(dropout_model(inputs))
    print(dropout_model(inputs))  

    # patch module to use dropout at test time
    mc_model = patch_module(model, inplace=False)
    wrapper = ModelWrapper(mc_model, criterion=None, replicate_in_memory=False)
    pool_dataset = torch.utils.data.TensorDataset(torch.rand((20, 10)), torch.rand((20,4)))
    print(pool_dataset[0],pool_dataset[1])
    # Shape: (len(pool), num_classes, num_mc_iters)
    predictions = wrapper.predict_on_dataset(
        pool_dataset,
        batch_size=4,
        iterations=20,
        use_cuda=False,
        workers=0,
        collate_fn=None
    )
    print(predictions)
    # average over the MC iters
    expected_p = np.mean(predictions, axis=-1)  
    print(expected_p)
    expected_entropy = - np.mean(np.sum(xlogy(predictions, predictions), axis=1),
                                    axis=-1)  # [batch size, ...]
    print(expected_entropy)
    entropy_expected_p = - np.sum(xlogy(expected_p, expected_p),
                                    axis=1)  # [batch size, ...]
    print(entropy_expected_p)
    bald_acq = entropy_expected_p - expected_entropy
    print(bald_acq)

    bald_scores = BALD().get_uncertainties(predictions)
    print("BALD score", bald_scores)

    # compute score
    scores = get_metrics("entropy")(expected_p)
    print("entropy", scores)
    
    probabilities = F.softmax(torch.tensor(predictions), dim=1).numpy()
    print(probabilities)
    # average over the MC iters
    expected_p = np.mean(probabilities, axis=-1)  
    print(expected_p)

    expected_entropy = - np.mean(np.sum(xlogy(probabilities, probabilities), axis=1),
                                    axis=-1)  # [batch size, ...]
    print(expected_entropy)
    entropy_expected_p = - np.sum(xlogy(expected_p, expected_p),
                                    axis=1)  # [batch size, ...]
    print(entropy_expected_p)
    bald_acq = entropy_expected_p - expected_entropy
    print("BALD score manual softmax", bald_acq)
    print("BALD score", bald_scores)

    # compute score
    scores = get_metrics("entropy")(expected_p)
    print("entropy manual softmax", scores)
    print(bald_acq == bald_scores)
