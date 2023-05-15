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
        self.drop = 0.0
        self.auto_cfg = AutoConfig.from_pretrained(
            cfg.model,
            output_hidden_states=True
        )
        self.vision_config = self.auto_cfg.vision_config
        self.text_config = self.auto_cfg.text_config

        self.model = AutoModel.from_pretrained(
            cfg.model,
            config=self.auto_cfg
        )
        self.vision_model = self.model.vision_model
        self.text_model = self.model.text_model

        self.vision_fc = nn.Sequential(
            nn.Linear(2944, 4096),  # will be added with style feature => 1024(ViT) + 1920(Style Model) = 2944
            nn.SiLU(),
            nn.Dropout(self.drop),
            nn.Linear(4096, 4096),
            nn.SiLU(),
            nn.Dropout(self.drop),
            nn.Linear(4096, 4096),
            nn.SiLU(),
            nn.Dropout(self.drop),
            nn.Linear(4096, 4096),
            nn.SiLU(),
            nn.Linear(4096, 384)
        )
        self.vision_pooling = getattr(pooling, cfg.image_pooling)(self.auto_cfg)  # for text pooling

        self.text_fc = nn.Linear(self.text_config.hidden_size, 384)  # for text embedding
        self.text_pooling = getattr(pooling, cfg.text_pooling)(self.auto_cfg)  # for text pooling

        if self.cfg.load_pretrained:
            self.model.load_state_dict(
                torch.load(cfg.checkpoint_dir + cfg.state_dict),
                strict=False
            )

        if cfg.reinit:
            self._init_weights(self.vision_fc)
            self._init_weights(self.text_fc)
            self.reinit_topk(self.vision_model, cfg.vision_num_reinit)
            self.reinit_topk(self.text_model, cfg.text_num_reinit)

        if cfg.freeze:
            freeze(self.vision_model.embeddings)
            freeze(self.vision_model.encoder.layer[:cfg.vision_num_freeze])

            freeze(self.text_model.embeddings)
            freeze(self.text_model.encoder.layer[:cfg.text_num_freeze])

        if cfg.gradient_checkpoint:
            self.model.gradient_checkpointing_enable()

    def reinit_topk(self, model, num_layers):
        """
        Re-initialize the last-k transformer Encoder layers.
        Encoder Layer: Embedding, Attention Head, LayerNorm, Feed Forward
        Args:
            model: The target transformer model.
            num_layers: The number of layers to be re-initialized.
        """
        if num_layers > 0:
            model.encoder.layers[-num_layers:].apply(self._init_weights)

    @staticmethod
    def _init_weights(module) -> None:
        """ over-ride initializes weights of the given module function (+initializes LayerNorm) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            """ reference from torch.nn.Layernorm with elementwise_affine=True """
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def forward(self, inputs: dict, mode: str, style_features: Tensor = None) -> list[Tensor]:
        """ forward pass function with mode (vision or text) """
        if mode == 'vision':
            outputs = self.vision_model(inputs)
            feature = outputs.last_hidden_state
            embedding = self.vision_pooling(feature)  # [batch_size, hidden_size(1024)]

            # clip_feature = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize
            # style_feature = style_features / style_features.norm(dim=-1, keepdim=True)  # normalize
            # logit = self.vision_fc(torch.cat([clip_feature, style_feature], dim=-1))
            logit = self.vision_fc(torch.cat([embedding, style_features], dim=-1))

        else:  # mode == 'text'
            outputs = self.text_model(**inputs)
            feature = outputs.last_hidden_state
            embedding = self.text_pooling(feature, inputs['attention_mask'])
            # text_features = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize
            logit = self.text_fc(embedding)

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
        style_model: efficientnet family, convnext_base, resent family
        efficientnet: pass keyword 'blocks' to forward function
        convnext_base: pass keyword 'stage' to forward function
        resnet: pass keyword 'layer1 ~ layer4' to forward function

    [Reference]
    https://www.kaggle.com/code/tanreinama/style-extract-from-vgg-clip-object-extract
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.style_model = timm.create_model(
            self.cfg.style_model,
            pretrained=True,
            features_only=False,  # will be drop classifier or regression head
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        if 'efficientnet' in self.cfg.style_model:
            layer_name = 'blocks'
        elif 'convnext' in self.cfg.style_model:
            layer_name = 'stages'
        elif 'resnet' in self.cfg.style_model:
            layer_name = ['layer1', 'layer2', 'layer3', 'layer4']
        self.feature1 = self.style_model.stem + self.style_model.stages[0:1]
        self.feature2 = self.style_model.stages[1:2]
        self.feature3 = self.style_model.stages[2:3]
        self.feature4 = self.style_model.stages[3:4]

    @staticmethod
    def gram_matrix(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        f = x.view(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2)) / (h * w)
        return g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding1 = self.feature1(x)
        embedding2 = self.feature2(embedding1)
        embedding3 = self.feature3(embedding2)
        embedding4 = self.feature4(embedding3)

        g1 = self.gram_matrix(embedding1)
        g2 = self.gram_matrix(embedding2)
        g3 = self.gram_matrix(embedding3)
        g4 = self.gram_matrix(embedding4)
        g = [self.avg(g1).squeeze(2), self.avg(g2).squeeze(2), self.avg(g3).squeeze(2), self.avg(g4).squeeze(2)]
        return torch.cat(g, dim=1)
