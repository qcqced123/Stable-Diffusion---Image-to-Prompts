import torch
import torch.nn as nn
import model.pooling as pooling
from torch import Tensor
from transformers import AutoConfig, AutoModel
from model.model_utils import freeze, reinit_topk


class SD2Model(nn.Module):
    """
    Model class for Open AI CLIP
    In OpenAI CLIP, Image and Text are encoded in same space
    So, extract embeddings from image and then use them for inference text prompt
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
        self.fc = nn.Linear(self.auto_cfg.hidden_size, 384)
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
        embedding = self.pooling(feature, inputs['attention_mask'])
        logit = self.fc(embedding)
        return logit
