import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch import Tensor
from transformers import AutoProcessor
import torchvision.transforms as T


class SD2Dataset:
    """ Image, Prompt Dataset For OpenAI CLIP Pipeline """
    def __init__(self, cfg, df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.df = df
        self.input_processor = AutoProcessor.from_pretrained(self.cfg.model)

    @staticmethod
    def img_transform(img) -> Tensor:
        """ Preprocess Image For Style-Extractor """
        transform = T.Compose([
            T.Resize(512),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(img)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item) -> tuple[Tensor, Tensor, Tensor]:
        """
        No need to tokenize text, CLIP has its own tokenizer stage in model class (encode text)
        return:
            image: image for style-extractor
            clip_image: image for CLIP
            target: prompt for CLIP
        """
        image = Image.open(self.df[item].image_name)
        target = self.df[item].prompt

        image = self.img_transform(image)  # resize & normalize for style-extractor
        clip_image = self.input_processor(image=image)  # resize & crop for pretrained CLIP
        target = self.input_processor(text=target)  # tokenize & normalize for pretrained CLIP
        return image, clip_image, target


class TestDataset(Dataset):
    """ For Inference Dataset Class """
    def __init__(self, cfg, df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.df = df
        self.input_processor = AutoProcessor.from_pretrained(self.cfg.model)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item) -> tuple:
        """ No need to text(label), because our goal of competition is to inference(generate) text """
        image = Image.open(self.df[item].image_name)
        image = self.input_processor(image=image)
        return image
