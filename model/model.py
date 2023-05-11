import torch
import torch.nn as nn
import timm
import model.pooling as pooling
from torch import Tensor
from transformers import AutoConfig, AutoModel
from model.model_utils import freeze, reinit_topk


class SD2Model(nn.Module):
    """
    Model class for Open AI CLIP
    In OpenAI CLIP, Image and Text are encoded in same space
    So, extract embeddings from image and then use them for inference text prompt
    And add additional pooling layer to ViT last hidden state embedding
    Because in paper & code, they use CLS token pooling before fully-connected layer
    in common sense, Mean Pooling more good performance than CLS token pooling,
    So apply GEMPooling instead of CLS token pooling, which is more good performance in detect object
    and then, background called style feature extract from other CNN based Model such as Efficientnet, ResNet

    [Reference]
    https://github.com/openai/CLIP/blob/main/clip/model.py
    https://www.kaggle.com/code/tanreinama/style-extract-from-vgg-clip-object-extract
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        ).vision_model
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 384)  # maybe need to append
        self.pooling = getattr(pooling, cfg.pooling)(self.auto_cfg)
        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=False
            )  # load student model's weight: it already has fc layer, so need to init fc layer later

        if cfg.reinit:
            self._init_weights(self.fc)
            reinit_topk(self.model, cfg.num_reinit)

        if cfg.freeze:
            freeze(self.model.embeddings)
            freeze(self.model.encoder.layer[:cfg.num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def _init_weights(self, module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.auto_cfg.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def feature(self, inputs: dict):
        outputs = self.model(**inputs)
        return outputs

    def forward(self, inputs: dict) -> list[Tensor]:
        outputs = self.feature(inputs)
        feature = outputs.last_hidden_state
        if self.cfg.pooling == 'WeightedLayerPooling':
            feature = outputs.hidden_states
        embedding = self.pooling(feature, inputs['attention_mask'])  # maybe need to append
        logit = self.fc(embedding)
        return logit


class StyleExtractModel(nn.Module):
    """
    Model class for Style-Extractor Model (EfficientNet, Convnext, ResNet, ...etc)
    Style-Extractor Model is used for extract style feature(background) from image
    And then, Feature will be concatenated with CLIP's Image embedding
    This Model is used ONLY extracting embedding, just Only forward pass

    In CLIP Model's Code in Huggingface, AutoProcessor do center crop to image in resizing 224x224
    But in many prompt sentences, they have a lot of word for background called feature.
    So we need to style-extractor for more good performance in generate prompt text
    option:
        style_model: efficientnet_b7, convnext_base

    [Reference]
    https://www.kaggle.com/code/tanreinama/style-extract-from-vgg-clip-object-extract
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.style_model = timm.create_model(
            self.cfg.style_model,
            pretrained=True
        )
        self.p1 = self.style_model.features[:11]
        self.p1[4] = nn.AvgPool2d(kernel_size=2)
        self.p1[9] = nn.AvgPool2d(kernel_size=2)
        self.p2 = self.style_model.features[11:20]
        self.p2[7] = nn.AvgPool2d(kernel_size=2)
        self.p3 = self.style_model.features[20:29]
        self.p3[7] = nn.AvgPool2d(kernel_size=2)
        self.avg = nn.AdaptiveAvgPool1d(1)

    @staticmethod
    def gram_matrix(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        f = x.view(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2)) / (h * w)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.p1(x)
        x2 = self.p2(x1)
        x3 = self.p3(x2)
        g1 = self.gram_matrix(x1)
        g2 = self.gram_matrix(x2)
        g3 = self.gram_matrix(x3)
        g = [self.avg(g1).squeeze(2), self.avg(g2).squeeze(2), self.avg(g3).squeeze(2)]
        return torch.cat(g, dim=1)


