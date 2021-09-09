"""Available models zoo.
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

import gensim.downloader as api

from transformers import AutoModel


def get_model(model_config: dict, experiment_type: str, num_classes: int):
    """Return the appropriate model given the experiment and model config.
    """
    if model_config["model_type"] == "BERT" and experiment_type == "classification":
        return BertCls(model_config["model_hypers"], num_classes)
    elif model_config["model_type"] == "BERT" and experiment_type == "tagging":
        return BertTag(model_config["model_hypers"], num_classes)
    elif model_config["model_type"] == "RNN" and experiment_type == "classification":
        return RnnCls(model_config["model_hypers"], num_classes)
    elif model_config["model_type"] == "MLP" and experiment_type == "classification":
        return MLP(model_config["model_hypers"], num_classes)
    elif model_config["model_type"] == "logistic" and experiment_type == "classification":
        return LogisticReg(model_config["model_hypers"], num_classes)
    elif model_config["model_type"] == "RNN-hid" and experiment_type == "classification":
        return RnnClsHid(model_config["model_hypers"], num_classes)
    else:
        raise NotImplementedError(f'{model_config["model_type"]}, {experiment_type} not implemented')


class BertCls(nn.Module):
    """Transformer based model using Huggingface + Pytorch for classification
    """
    def __init__(self, config, num_classes):
        super(BertCls, self).__init__()
        self.num_classes = num_classes
        self.transformer = AutoModel.from_pretrained(config["architecture"]["pretrained_model"])
        self.dropout = nn.Dropout(p=config["architecture"]["dropout"])
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        self.fine_tune = config["architecture"]["fine_tune"]

        if not self.fine_tune:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, x, feats=False):
        """Params:
        - x (dict): dict of input tensors
        - feats (bool): whether to return text embedding
        """
        input_ids = x["input_ids"]
        attn_mask = x["attention_mask"]
        transformer_out = self.transformer(input_ids, attention_mask=attn_mask)
        pooled_output = transformer_out[1]              # [CLS] token (B, hid_dim)

        # classify
        logits = self.classifier(self.dropout(pooled_output))
        if feats:
            return logits, pooled_output
        else:
            return logits


class BertTag(nn.Module):
    """Transformer based model using Huggingface + Pytorch for sequence tagging (e.g. NER)
    """
    def __init__(self, config, num_classes):
        super(BertTag, self).__init__()
        self.num_classes = num_classes
        self.transformer = AutoModel.from_pretrained(config["architecture"]["pretrained_model"])
        # TODO

    def forward(self, x, feats=False):
        """Params:
        - x (dict): dict of input tensors
        - feats (bool): whether to return text embedding
        """
        # TODO
        input_ids, attn_mask = x
        outputs = self.cls(input_ids, attention_mask=attn_mask)
        return outputs


class RnnClsHid(nn.Module):
    """RNN model in Pytorch for classification
    Using hidden state as features (not outputs)
    """
    def __init__(self, config, num_classes):
        """Params:
        - config: model config dict
        - num_classes:
        """
        super(RnnClsHid, self).__init__()
        self.num_classes = num_classes
        self.pretrained_emb = config["architecture"]["pretrained_emb"]
        self.rnn_hid_dim = config["architecture"]["hid_dim"] // 2
        self.fine_tune = config["architecture"]["fine_tune"]

        self.embed = WordEmbedding(self.pretrained_emb, self.fine_tune)

        # RNN 
        self.rnn = nn.LSTM(
            input_size=self.embed.text_emb_size,
            hidden_size=self.rnn_hid_dim,
            num_layers=config["architecture"]["num_layers"],
            bidirectional=True,
            batch_first=True)

        # head
        self.dropout = nn.Dropout(p=config["architecture"]["dropout"])
        self.classifier = nn.Linear(self.rnn_hid_dim*2, self.num_classes)

    def forward(self, x, feats=False):
        """Params:
        - x (dict): dict of input tensors
        - feats (bool): whether to return text embedding
        """
        text = x["input_ids"]
        B, _ = text.shape
        # padding masks (padding token = 0)
        padding_mask = torch.where(text != 0, 1, 0)
        seq_lens = torch.sum(padding_mask, dim=-1).cpu()

        # embed
        text_embedding = self.embed(text)      # (B x max_seq_len x emb_dim)

        # feed through RNN
        text_embedding_packed = pack_padded_sequence(text_embedding, seq_lens, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()   # to prevent baal error
        _, (ht, _) = self.rnn(text_embedding_packed)

        # concat forward and backward results (takes hidden states)
        seq_embed = torch.cat((ht[0], ht[1]), dim=-1)
        # classify        
        logits = self.classifier(self.dropout(seq_embed))

        if feats:
            return logits, seq_embed
        else:
            return logits


class RnnCls(nn.Module):
    """RNN model in Pytorch for classification
    """
    def __init__(self, config, num_classes):
        """Params:
        - config: model config dict
        - num_classes:
        """
        super(RnnCls, self).__init__()
        self.num_classes = num_classes
        self.pretrained_emb = config["architecture"]["pretrained_emb"]
        self.rnn_hid_dim = config["architecture"]["hid_dim"] // 2      # assuming bidirectional
        # self.head_dim = config["architecture"]["head_dim"]
        self.fine_tune = config["architecture"]["fine_tune"]

        self.embed = WordEmbedding(self.pretrained_emb, self.fine_tune)

        # RNN 
        self.rnn = nn.LSTM(
            input_size=self.embed.text_emb_size,
            hidden_size=self.rnn_hid_dim,
            num_layers=config["architecture"]["num_layers"],
            bidirectional=True,
            batch_first=True)

        # head
        self.dropout = nn.Dropout(p=config["architecture"]["dropout"])
        # self.fc = nn.Linear(self.rnn_hid_dim*2, self.head_dim)
        # self.classifier = nn.Linear(self.head_dim, self.num_classes)
        self.classifier = nn.Linear(self.rnn_hid_dim*2, self.num_classes)

    def forward(self, x, feats=False):
        """Params:
        - x (dict): dict of input tensors
        - feats (bool): whether to return text embedding
        """
        text = x["input_ids"]
        B, _ = text.shape
        # padding masks (padding token = 0)
        padding_mask = torch.where(text != 0, 1, 0)
        seq_lens = torch.sum(padding_mask, dim=-1).cpu()

        # embed
        text_embedding = self.embed(text)      # (B x max_seq_len x emb_dim)

        # feed through RNN
        text_embedding_packed = pack_padded_sequence(text_embedding, seq_lens, batch_first=True, enforce_sorted=False)
        self.rnn.flatten_parameters()   # to prevent baal error
        rnn_out_packed, _ = self.rnn(text_embedding_packed)
        rnn_out, _ = pad_packed_sequence(rnn_out_packed, batch_first=True)     # (B, max_seq_len, rnn_hid_dim*2)

        # concat forward and backward results (takes output states)
        seq_len_indices = [length-1 for length in seq_lens]
        batch_indices = list(range(B))  # all
        rnn_out_forward = rnn_out[batch_indices, seq_len_indices, :self.rnn_hid_dim]   # last state of forward (not padded)
        rnn_out_backward = rnn_out[:, 0, self.rnn_hid_dim:]     # last state of backward (= first timestep)
        seq_embed = torch.cat((rnn_out_forward, rnn_out_backward), -1)        # (B, rnn_hid_dim*2)

        # classify        
        logits = self.classifier(self.dropout(seq_embed))

        if feats:
            return logits, seq_embed
        else:
            return logits


class RnnTag(nn.Module):
    """RNN model in Pytorch for sequence tagging
    """
    def __init__(self, config, num_classes):
        """Params:
        - config: model config dict
        - num_classes:
        """
        super(RnnTag, self).__init__()

    def forward(self, x, feats=False):
        raise NotImplementedError()


class LogisticReg(nn.Module):
    """Linear model for classification
    """
    def __init__(self, config, num_classes):
        """Params:
        - config: model config dict
        - num_classes:
        """
        super(LogisticReg, self).__init__()

        self.num_classes = num_classes
        self.pretrained_emb = config["architecture"]["pretrained_emb"]
        self.fine_tune = config["architecture"]["fine_tune"]
        self.pooling_strat = config["architecture"]["pooling_strat"]

        self.dropout = nn.Dropout(p=config["architecture"]["dropout"])
        self.embed = WordEmbedding(self.pretrained_emb, self.fine_tune)
        self.classifier = nn.Linear(self.embed.text_emb_size, self.num_classes)

    def forward(self, x, feats=False):
        """Params:
        - x (dict): dict of input tensors
        - feats (bool): whether to return text embedding
        """
        text = x["input_ids"]      
        # embed
        text_embedding = self.embed(text)      # (B x max_seq_len x emb_dim)

        # pool the embeddings
        if self.pooling_strat == "mean":
            padding_mask = torch.where(text != 0, 1, 0)                     # (B, max_seq_len)
            seq_lens = torch.sum(padding_mask, dim=-1).unsqueeze(-1)        # (B, 1)
            seq_embed = torch.sum(text_embedding, dim=1)                    # (B, emb_dim)
            # safe divide
            seq_embed = torch.where(seq_lens != 0, torch.div(seq_embed, seq_lens), seq_embed)      # (B, emb_dim)
        elif self.pooling_strat == "max":
            seq_embed = torch.max(text_embedding, dim=1)[0]
        else:
            raise NameError(f"{self.pooling_strat} pooling strat not defined")

        # classify
        logits = self.classifier(self.dropout(seq_embed))

        if feats:
            return logits, seq_embed
        else:
            return logits


class MLP(nn.Module):
    """Linear model for classification
    """
    def __init__(self, config, num_classes):
        """Params:
        - config: model config dict
        - num_classes:
        """
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.pretrained_emb = config["architecture"]["pretrained_emb"]
        self.fine_tune = config["architecture"]["fine_tune"]
        self.pooling_strat = config["architecture"]["pooling_strat"]
        self.hid_dims = config["architecture"]["hid_dims"]
        self.non_linearity = config["architecture"]["non_linearity"]
        
        self.embed = WordEmbedding(self.pretrained_emb, self.fine_tune)
        self.classifier = FullyConnected(input_dim=self.embed.text_emb_size,
                                         hid_dims=self.hid_dims,
                                         output_dim=self.num_classes,
                                         non_linearity=self.non_linearity,
                                         dropout=config["architecture"]["dropout"])

    def forward(self, x, feats=False):
        """Params:
        - x (dict): dict of input tensors
        - feats (bool): whether to return text embedding
        """
        text = x["input_ids"]      
        # embed
        text_embedding = self.embed(text)      # (B x max_seq_len x emb_dim)

        # pool the embeddings
        if self.pooling_strat == "mean":
            padding_mask = torch.where(text != 0, 1, 0)                     # (B, max_seq_len)
            seq_lens = torch.sum(padding_mask, dim=-1).unsqueeze(-1)        # (B, 1)
            seq_embed = torch.sum(text_embedding, dim=1)                    # (B, emb_dim)
            # safe divide
            seq_embed = torch.where(seq_lens != 0, torch.div(seq_embed, seq_lens), seq_embed)      # (B, emb_dim)
        elif self.pooling_strat == "max":
            seq_embed = torch.max(text_embedding, dim=1)[0]
        else:
            raise NameError(f"{self.pooling_strat} pooling strat not defined")

        # classify
        logits = self.classifier(seq_embed)

        if feats:
            return logits, seq_embed
        else:
            return logits


class FullyConnected(nn.Module):
    """Generic fully connected MLP with adjustable depth.
    """
    def __init__(self, input_dim, hid_dims, output_dim, non_linearity="ReLU", dropout=0.0):
        """Params:
        - input_dim (int): input dimension
        - hid_dims (List[int]): list of hidden layer dimensions
        - output_dim (int): output dimension
        - non_linearity (str): type of non-linearity in hidden layers
        - dropout (float): dropout rate (applied each layer)
        """
        super(FullyConnected, self).__init__()
        dims = [input_dim] + hid_dims
        self.dropout = nn.Dropout(p=dropout)
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        if non_linearity == "ReLU":
            self.act = nn.ReLU()
        elif non_linearity == "tanh":
            self.act = nn.Tanh()
        elif non_linearity == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NameError(f"activation {non_linearity} not defined")
        self.acts = nn.ModuleList([self.act for _ in range(len(dims)-1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, act in zip(self.fcs, self.acts):
            x = act(fc(self.dropout(x)))
        # non activated final layer
        return self.fc_out(x)


class WordEmbedding(nn.Module):
    """General class to load up word embeddings
    """
    def __init__(self, text_encoder_type, fine_tune):
        """Embeds a tokenised sequence into word embeddings
        """
        super(WordEmbedding, self).__init__()
        self.text_encoder_type = text_encoder_type
        self.fine_tune = fine_tune

        embedding_weights, dictionary = self.get_embedding_weights()
        # save dictionary for future use
        self.dictionary = dictionary
        self.text_emb_size = embedding_weights.shape[-1]
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights))

        # whether to fine tune embeddings
        if not self.fine_tune:
            for param in self.embed.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Params:
        - x (B x max_seq_len)
        Returns:
        - embeddings (B x max_seq_len x embedding_dim)
        """
        return self.embed(x)

    def get_embedding_weights(self):
        """Loads gensim word embedding weights into a matrix
        TODO OPTIMISE. this is slow so takes too long
        Returns:
        - embedding_matrix: matrix of embedding weights
        """
        # if self.pretrained_emb  == "rand":
        #     self.embed = nn.Embedding(len(self.token2id), self.rnn_hid_dim)
        #     self.text_emb_size = self.rnn_hid_dim

        word_model = load_embedding_model(self.text_encoder_type)
        embedding_dim = word_model.vector_size
        dictionary = {k: v+1 for (k, v) in word_model.key_to_index.items()}
        dictionary["<PAD>"] = 0    # add pad token
        oov_id = len(dictionary)
        dictionary["<OOV>"] = oov_id          # add OOV token

        weights = np.zeros((len(dictionary), embedding_dim))
        # randomly initialise OOV token between -1 and 1
        weights[oov_id, :] = 2*np.random.rand(embedding_dim) - 1
        for word, token in dictionary.items():
            if word not in ["<PAD>", "<OOV>"]:
                weights[token, :] = word_model[word]
        print(f"done. Vocab size: {weights.shape[0]}. Embedding dim: {weights.shape[1]}")

        return weights, dictionary


def load_embedding_model(text_encoder_type):
    """Loads a given word embedding model from Gensim
    Params:
    - text_encoder_type: type of word embedding
    Returns:
    - word_model: KeyedVectors gensim model
    """
    print("loading pretrained word vectors...")
    if text_encoder_type == "glove":
        word_model = api.load("glove-wiki-gigaword-300")
    elif text_encoder_type == "word2vec":
        word_model = api.load("word2vec-google-news-300")
    return word_model


if __name__ == "__main__":

    pass