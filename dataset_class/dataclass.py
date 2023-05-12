import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch import Tensor
from transformers import AutoProcessor, CLIPImageProcessor
import torchvision.transforms as T


class SD2Dataset:
    """ Image, Prompt Dataset For OpenAI CLIP Pipeline """
    def __init__(self, cfg, df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.df = df.applymap(str)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.cfg.model)
        self.tokenizer = cfg.tokenizer

    @staticmethod
    def img_transform(img) -> Tensor:
        """ Preprocess Image For Style-Extractor """
        transform = T.Compose([
            T.Resize(384),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(img)

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
        image = Image.open(self.df.iloc[item, 0])
        target = self.df.iloc[item, 1]

        clip_image = torch.tensor(self.image_processor(image)['pixel_values'])  # resize & crop for pretrained CLIP
        target = self.tokenizing(target)  # tokenize & normalize for pretrained CLIP
        image = self.img_transform(image)  # resize & normalize for style-extractor
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
