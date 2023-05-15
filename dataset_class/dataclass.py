import gc
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from torch.utils.data import Dataset
from torch import Tensor

import configuration
from .data_preprocessing import img_transform, tokenizing, clip_img_process, load_style_embedding


class SD2Dataset(Dataset):
    """ Image, Prompt Dataset For OpenAI CLIP Pipeline """
    def __init__(self, cfg: configuration.CFG, df: pd.DataFrame) -> None:
        self.cfg = cfg
        self.df = df
        self.tokenizer = tokenizing
        self.image_processor = clip_img_process
        self.img_transform = img_transform
        # self.style_table = load_style_embedding()

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
        image = rasterio.open(self.df.iloc[item, 0])
        # style_feature = self.embedding_table[self.df.iloc[item, 3]]
        tensor_image = image.read(
            out_shape=(3, int(512), int(512)),
            resampling=Resampling.bilinear
        ).transpose(1, 2, 0)
        pd_target = self.df.iloc[item, 1]
        clip_image = self.image_processor(self.cfg, image=tensor_image)
        target = self.tokenizer(self.cfg, pd_target)
        # style_image = self.style_table[self.df.iloc[item, 3]]
        style_image = self.img_transform(tensor_image)
        del image
        gc.collect()
        return style_image, clip_image, target
