import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor


class IMGDataset:
    """ Image Dataset For OpenAI CLIP Pipeline """
    def __init__(self, cfg, df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.df = df
        self.input_processor = AutoProcessor.from_pretrained(self.cfg.model)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        image = self.input_processor(image)
        target = self.labels[item]
        return image, target


class FBPDataset(Dataset):
    """ For Supervised Learning Pipeline """
    def __init__(self, cfg, tokenizer, df):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer  # get from train.py
        self.df = df

    def tokenizing(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.cfg.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None,
            add_special_tokens=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v)
        return inputs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = self.tokenizing(self.df.iloc[idx, 1])
        label = torch.tensor(self.df.iloc[idx, 2:8], dtype=torch.float)
        return inputs, label


class MPLDataset(Dataset):
    """ For Semi-Supervised Learning, Meta Pseudo Labels Pipeline """
    def __init__(self, cfg, tokenizer, df):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer  # get from train.py
        self.df = df

    def tokenizing(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.cfg.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None,
            add_special_tokens=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v)
        return inputs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = self.tokenizing(self.df.iloc[idx, 1])
        return inputs


class TestDataset(Dataset):
    """ For Inference Dataset Class """
    def __init__(self, cfg, tokenizer, df):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer  # get from train.py
        self.df = df

    def tokenizing(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.cfg.max_len,
            padding='max_length',
            truncation=True,
            return_tensors=None,
            add_special_tokens=True,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v)
        return inputs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = self.tokenizing(self.df.iloc[idx, 1])
        return inputs
