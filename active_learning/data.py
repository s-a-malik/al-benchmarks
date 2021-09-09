"""Dataloader classes and utilities.
"""
import os
import random

import torch
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset

from sklearn.model_selection import train_test_split as split

import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS

from datasets import load_dataset
from transformers import AutoTokenizer

from active_learning.model_classes import load_embedding_model
# from model_classes import load_embedding_model


def init_data(dataset_config: dict):
    """Download (or load from disk) and apply dataset specific preprocessing.
    Params:
    - dataset_config (dict): dataset config dict
    Returns:
    - train (Dataset): Full training dataset
    - dev (Dataset):
    - test (Dataset):
    - num_classes (int): number of classes
    - labelled_pool (List(int)): indices of seed labelled datapoints in train
    - unlabelled_pool (List(int)): unlabelled indices of train
    """
    # train and dev will be in random order, test may be ordered according to labels
    if dataset_config["name"] == "CoLA":
        train, dev, test, num_classes = load_cola(dataset_config)
    elif dataset_config["name"] == "AGNews":
        train, dev, test, num_classes = load_ag_news(dataset_config)
    elif dataset_config["name"] == "DBPedia":
        train, dev, test, num_classes = load_dbpedia(dataset_config)
    elif dataset_config["name"] == "YRF":
        train, dev, test, num_classes = load_yrf(dataset_config)
    else:
        raise NameError(f"Dataset {dataset_config['name']} not implemented.")
    # etc.

    # shrink size if debugging
    if dataset_config["debug"]:
        # choose a random subset using huggingface select function
        train = train.select(random.sample(range(len(train)), k=200))
        dev = dev.select(random.sample(range(len(dev)), k=40))
        test = test.select(random.sample(range(len(test)), k=200))

    # create class imbalance
    random.seed(dataset_config["seed"])
    if dataset_config["pool_balance"] == "balanced":
        pass
    elif dataset_config["pool_balance"] == "imbalanced":
        train = train.filter(lambda example: create_imbalanced_dataset(example, dataset_config["imbalance_prop"], dataset_config['imbalance_cls']))
    else:
        NameError(f"pool_balance = {dataset_config['pool_balance']} not allowed")

    if dataset_config["dev_balance"] == "balanced":
        pass
    elif dataset_config["dev_balance"] == "imbalanced":
        dev = dev.filter(lambda example: create_imbalanced_dataset(example, dataset_config["imbalance_prop"], dataset_config['imbalance_cls']))
    else:
        NameError(f"dev_balance = {dataset_config['dev_balance']} not allowed")

    # get seed labelled pool indices (using the same seed data every time)
    random.seed(dataset_config["seed"])
    if dataset_config["seed_balance"] == "balanced":
        # this is random (will have some variance vs pool)
        indices = list(range(len(train)))
        unlabelled_pool_idx, labelled_pool_idx = split(
            indices,
            random_state=dataset_config["seed"],
            test_size=dataset_config["seed_size"]
        )
    elif dataset_config["seed_balance"] == "stratified":
        # this is the same as the underlying train set (which may be unbalanced)
        indices = list(range(len(train)))
        unlabelled_pool_idx, labelled_pool_idx = split(
            indices,
            random_state=dataset_config["seed"],
            test_size=dataset_config["seed_size"],
            stratify=train['label']
        )
    elif dataset_config["seed_balance"] == "imbalanced":
        # artificially sample an imbalanced seed set from the pool
        unlabelled_pool_idx, labelled_pool_idx = create_imbalanced_seed(
            train,
            num_classes,
            dataset_config["seed_size"],
            dataset_config['imbalance_prop'],
            dataset_config['imbalance_cls']
        )
    else:
        raise NameError(f"seed_balance = {dataset_config['seed_balance']} not allowed")

    return train, dev, test, num_classes, labelled_pool_idx, unlabelled_pool_idx


def create_imbalanced_seed(data, num_classes, seed_size, prop, label):
    """Artificially make an imbalanced seed set. 
    Chooses examples from the pool at random and adds them to the list of seed examples until the desired proportion is reached.
    Params:
    - data (dataset): train pool dataset (Huggingface)
    - num_classes (int): number of classes
    - seed_size (int): number of examples to include in seed
    - prop (float): proportion of examples of imbalanced label to include (weight other classes as 1, weight label as prop)
    - label (int): imbalanced label
    Returns:
    - unlabelled_pool_idx (list(int)): list of unlabelled datapoint indices in train
    - labelled_pool_idx (list(int)): list of labelled datapoint indices in train
    """
    labelled_pool_idx = []
    unlabelled_pool_idx = [i for i in range(len(data))]
    label_weights = [1 if x != label else prop for x in range(num_classes)]
    total_weight = sum(label_weights)
    # these are the number of labelled examples of each class we would like to include.
    # these are floats as we can exceed some classes by 1 to get desired seed size
    desired_seed_label_count = [x*seed_size/total_weight for x in label_weights]
    # TODO change counts to defaultdicts to avoid key errors
    current_seed_label_count = [0 for _ in range(num_classes)]

    while len(labelled_pool_idx) < seed_size:
        sample_idx = random.choice(unlabelled_pool_idx)
        example = data[sample_idx]
        if current_seed_label_count[example['label']] < desired_seed_label_count[example['label']]:
            # add to labelled pool
            labelled_pool_idx.append(sample_idx)
            current_seed_label_count[example['label']] += 1
        # remove from unlabelled pool. TODO more efficient?
        unlabelled_pool_idx = [i for i in range(len(data)) if i not in labelled_pool_idx]

    return unlabelled_pool_idx, labelled_pool_idx


def create_imbalanced_dataset(example: dict, prop: float, label: int):
    """Filtering function to randomly remove some examples 
    of a particular class from the dataset.
    Params:
    - example (dict): A document in the train pool
    - prop (float): proportion of the examples of label cls to keep
    - label (int): class to subsample
    Returns:
    - keep (bool): whether or not to keep this example 
    """
    if example["label"] == label:
        return True if random.random() < prop else False
    else:
        return True


def load_yrf(dataset_config: dict):
    train_and_dev = load_dataset('yelp_review_full', split='train')
    train_and_dev = train_and_dev.train_test_split(test_size=dataset_config['val_prop'], seed=dataset_config["seed"])
    train = train_and_dev['train']
    dev = train_and_dev['test']
    test = load_dataset('yelp_review_full', split='test')
    # change to same columns and preprocess
    train = train.map(lambda examples: {'text': preprocess_text(examples['text'])})
    dev = dev.map(lambda examples: {'text': preprocess_text(examples['text'])})
    test = test.map(lambda examples: {'text': preprocess_text(examples['text'])})
    num_classes = test.features['label'].num_classes
    return train, dev, test, num_classes  


def load_ag_news(dataset_config: dict):
    train_and_dev = load_dataset('ag_news', split='train')
    train_and_dev = train_and_dev.train_test_split(test_size=dataset_config['val_prop'], seed=dataset_config["seed"])
    train = train_and_dev['train']
    dev = train_and_dev['test']
    test = load_dataset('ag_news', split='test')
    # change to same columns and preprocess
    train = train.map(lambda examples: {'text': preprocess_text(examples['text'])})
    dev = dev.map(lambda examples: {'text': preprocess_text(examples['text'])})
    test = test.map(lambda examples: {'text': preprocess_text(examples['text'])})
    num_classes = test.features['label'].num_classes
    return train, dev, test, num_classes


def load_dbpedia(dataset_config: dict):
    train_and_dev = load_dataset('dbpedia_14', split='train')
    train_and_dev = train_and_dev.train_test_split(test_size=dataset_config['val_prop'], seed=dataset_config["seed"])
    train = train_and_dev['train']
    dev = train_and_dev['test']
    test = load_dataset('dbpedia_14', split='test')
    # change to same columns and preprocess
    train = train.map(lambda examples: {'text': preprocess_text(examples['content'])}, remove_columns=['content'])
    dev = dev.map(lambda examples: {'text': preprocess_text(examples['content'])}, remove_columns=['content'])
    test = test.map(lambda examples: {'text': preprocess_text(examples['content'])}, remove_columns=['content'])
    num_classes = test.features['label'].num_classes
    return train, dev, test, num_classes


def load_cola(dataset_config: dict):
    # TODO all the test data labels are -1 for some reason?? (should be 0 or 1)
    train_and_dev = load_dataset('glue', 'cola', split='train')
    train_and_dev = train_and_dev.train_test_split(test_size=dataset_config['val_prop'], seed=dataset_config["seed"])
    train = train_and_dev['train']
    dev = train_and_dev['test']
    test = load_dataset('glue', 'cola', split='test')
    # change to same columns and preprocess
    train = train.map(lambda examples: {'text': preprocess_text(examples['sentence'])}, remove_columns=['sentence'])
    dev = dev.map(lambda examples: {'text': preprocess_text(examples['sentence'])}, remove_columns=['sentence'])
    test = test.map(lambda examples: {'text': preprocess_text(examples['sentence'])}, remove_columns=['sentence'])
    num_classes = test.features['label'].num_classes
    return train, dev, test, num_classes


def preprocess_text(text: str):
    """Preprocessing function for strings. 
    Call in dataset mapping.
    """
    # remove trailing/leading whitespace
    text = text.strip()

    # .lower() depends on model so doing this in collate function

    # TODO other preprocessing - punctuation/ascii etc.
    text = text.replace("\\n", " ")
    # text = text.replace("\\'", "\'")
    # text = text.replace('\\"', "\'")
    text = text.encode('ascii', 'ignore').decode()

    return text


class Collate:
    """Collate function class for dataloaders. Tokenizes data appropriately for the model.
    TODO might be a better place to do this?
    """
    def __init__(self, model_config: dict):
        self.model_type = model_config["model_type"]
        if self.model_type == "BERT":
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["model_hypers"]["architecture"]["pretrained_model"])
        elif self.model_type in ["RNN", "RNN-hid", "logistic", "MLP"]:
            self.max_length = model_config["model_hypers"]["architecture"]["max_length"]
            word_model = load_embedding_model(model_config["model_hypers"]["architecture"]["pretrained_emb"])
            self.dictionary = {k: v+1 for (k, v) in word_model.key_to_index.items()}
            self.dictionary["<PAD>"] = 0    # add pad token
            self.oov_id = len(self.dictionary)
            self.dictionary["<OOV>"] = self.oov_id          # add OOV token
            # Create a Tokenizer with spacy
            nlp = en_core_web_sm.load()
            self.tokenizer = nlp.tokenizer
        else:
            # tokenize in other ways for other models
            raise NameError(f"model type: {self.model_type} not allowed")

    def __call__(self, batch):
        text = [x['text'] for x in batch]
        labels = torch.tensor([x['label'] for x in batch])
        # tokenize
        if self.model_type == "BERT":
            inputs = self.tokenizer(
                text, return_token_type_ids=False,
                return_tensors="pt", padding=True, truncation=True
            )
        elif self.model_type in ["RNN", "RNN-hid", "logistic", "MLP"]:
            # find max length sequence in batch
            lengths = [len(doc) for doc in self.tokenizer.pipe(text, batch_size=len(batch))]
            max_length = max(lengths)
            # truncate if too long
            max_length = min([max_length, self.max_length])
            # TODO get rid of excess padding after stopword removal
            encoded_text = torch.zeros((len(batch), max_length), dtype=torch.int64)

            # tokenise and encode each document. TODO can we batch this better?
            for i, tokenized_doc in enumerate(self.tokenizer.pipe(text, batch_size=len(batch))):
                # remove stop words and punctuation
                if self.model_type != "RNN":
                    doc = [word.text.lower() for word in tokenized_doc if (word.text.lower() not in STOP_WORDS) and (word.text.isalpha())]                   
                else:
                    # keep them for RNN
                    doc = [word.text.lower() for word in tokenized_doc]

                length = len(doc)
                # pad
                if length < max_length:
                    padded = doc + ["<PAD>" for _ in range(max_length - length)]
                else:
                    padded = doc[:max_length]

                # TODO could do this in one step (how much time does this save?)
                int_encoded = [self.dictionary[word]
                               if word in self.dictionary
                               else self.oov_id
                               for word in padded]
                encoded_text[i, :] = torch.tensor(int_encoded)

            # return a dict for inputs to match BERT style {"input_ids": [101, 2, ...]}
            inputs = {"input_ids": encoded_text}
        else:
            raise NameError(f"model {self.model_type} not defined")

        return inputs, labels


if __name__ == "__main__":
    # check data loading correctly
    # train, dev, test, num_classes = load_cola({"seed": 123, "val_prop": 0.2})
    # train, dev, test, num_classes = load_ag_news({"seed": 123, "val_prop": 0.2})
    train, dev, test, num_classes = load_yrf({"seed": 123, "val_prop": 0.2})
    # train, dev, test, num_classes = load_dbpedia({"seed": 123, "val_prop": 0.2})
    print("len train/dev/test/classes", len(train), len(dev), len(test), num_classes)
    indices = list(range(len(train)))
    unlabelled_pool, labelled_pool = split(indices, random_state=123, test_size=50)
    print(labelled_pool[:5], len(labelled_pool))
    print(train[labelled_pool[:5]])

    # get stats for this batch and full pool
    import pandas as pd
    size_of_batch = 50
    # this batch
    df_full_labelled = pd.DataFrame(train[labelled_pool])
    df_batch = pd.DataFrame(train[labelled_pool[-size_of_batch:]])
    print(df_batch)
    train_class_support = df_full_labelled["label"].value_counts().sort_index().to_list()
    print(train_class_support)
    batch_class_support = df_batch["label"].value_counts().sort_index().to_list()
    print(batch_class_support)

    # print("Test labels", test[:100]["label"])
    # test dataloaders
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # BERT
    model_config = {"model_type":"BERT", "model_hypers": {"architecture": {"pretrained_model":"bert-base-uncased"} }}
    collate = Collate(model_config)
    pool = Subset(train, unlabelled_pool)
    pool_loader = DataLoader(
            pool,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=collate)
    train_loader = DataLoader(
                    train,
                    batch_size=4,  
                    sampler=SubsetRandomSampler(labelled_pool),
                    num_workers=2,
                    collate_fn=collate
                )
    print(next(iter(train_loader)))
    print(next(iter(pool_loader)))
    # RNN
    model_config = {"model_type":"RNN", "model_hypers": {"architecture": {"pretrained_emb": "glove", "max_length": 300}}}
    # model_config = {"model_type":"logistic", "model_hypers": {"architecture": {"pretrained_emb": "glove", "max_length": 300}}}
    # model_config = {"model_type":"MLP", "model_hypers": {"architecture": {"pretrained_emb": "glove", "max_length": 300}}}

    collate = Collate(model_config)
    pool = Subset(train, unlabelled_pool)
    pool_loader = DataLoader(
            pool,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=collate)
    train_loader = DataLoader(
                    train,
                    batch_size=4,  
                    sampler=SubsetRandomSampler(labelled_pool),
                    num_workers=2,
                    collate_fn=collate
                )
    test_loader = DataLoader(
                    test,
                    batch_size=4,
                    shuffle=False,
                    collate_fn=collate
                )
    for batch in pool_loader:
        text = batch[0]["input_ids"]
        for doc in text:
            if (doc == 0).all():
                print(doc)

    print(next(iter(train_loader)))
    print(next(iter(pool_loader)))
    print(next(iter(test_loader)))
