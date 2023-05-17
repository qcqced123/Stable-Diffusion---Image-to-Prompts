import gc
import math
import dataset_class.dataclass as dataset_class
import model.metric as model_metric
import model.metric_learning as metric_learning
import model.model as model_arch
from torch.utils.data import DataLoader
from dataset_class.data_preprocessing import *
from utils.helper import *
from trainer.trainer_utils import *
from model.metric import *
from tqdm.auto import tqdm


class SD2Trainer:
    """ For OpenAI CLIP Fine-Tuned Pipeline with Multiple Negative Ranking Loss, Style-Extractor """
    def __init__(self, cfg, generator) -> None:
        self.cfg = cfg
        self.model_name = self.cfg.model.split('/')[1]
        self.generator = generator
        self.df = load_data('./dataset_class/final_final_prompt.csv')
        if self.cfg.gradient_checkpoint:
            self.save_parameter = f'(best_score){str(self.model_name)}_state_dict.pth'

    def make_batch(self, fold: int):
        """ Make Batch Dataset for main train loop """
        train = self.df[self.df['fold'] != fold].reset_index(drop=True)
        valid = self.df[self.df['fold'] == fold].reset_index(drop=True)

        # Custom Datasets
        train_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, train)
        valid_dataset = getattr(dataset_class, self.cfg.dataset)(self.cfg, valid)

        # DataLoader
        loader_train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        loader_valid = DataLoader(
            valid_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=self.generator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader_train, loader_valid, train

    def model_setting(self, len_train: int):
        """ set train & validation options for main train loop """
        model = getattr(model_arch, self.cfg.model_arch)(self.cfg)
        style_model = getattr(model_arch, self.cfg.style_model_arch)(self.cfg)
        if self.cfg.resume:
            model.load_state_dict(torch.load(self.cfg.checkpoint_dir + self.cfg.state_dict))

        model.to(self.cfg.device)
        style_model.to(self.cfg.device)

        criterion = getattr(metric_learning, self.cfg.loss_fn)(self.cfg.reduction)
        val_metrics = getattr(model_metric, self.cfg.metrics)()
        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=model.parameters(),
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )
        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

        return model, style_model, criterion, val_metrics, optimizer, lr_scheduler

    # Train Function
    def train_fn(self, loader_train, model, style_model, criterion, optimizer, lr_scheduler):
        """ Training Function """
        torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        losses = AverageMeter()
        model.train(), style_model.eval()
        for step, (style_images, clip_images, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            for k, v in labels.items():
                labels[k] = v.to(self.cfg.device)  # prompt to GPU

            clip_images = clip_images.squeeze().to(self.cfg.device)  # clip image to GPU
            batch_size = self.cfg.batch_size
            with torch.no_grad():
                style_images = style_images.to(self.cfg.device)  # style image to GPU
                style_features = style_model(style_images)  # style image to style feature

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                image_features = model(clip_images, 'vision', style_features=style_features)
                text_features = model(labels, 'text')

                image_features = image_features / image_features.norm(dim=-1, keepdim=True) * math.sqrt(384)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True) * math.sqrt(384)

                loss = (criterion(image_features, text_features) + criterion(text_features, image_features)) / 2

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()
            losses.update(loss.detach(), batch_size)

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()

            gc.collect()

        train_loss = losses.avg.detach().cpu().numpy()
        return train_loss

    # Validation Function
    def valid_fn(self, loader_valid, model, style_model, val_metrics) -> float:
        """ Validation Functions """
        metrics = AverageMeter()
        model.eval(), style_model.eval()
        # style_model.eval()
        with torch.no_grad():
            for step, (style_images, clip_images, labels) in enumerate(tqdm(loader_valid)):
                for k, v in labels.items():
                    labels[k] = v.to(self.cfg.device)  # prompt to GPU

                clip_images = clip_images.squeeze().to(self.cfg.device)  # clip image to GPU
                val_batch_size = clip_images.shape[0]

                style_images = style_images.to(self.cfg.device)  # style image to GPU
                style_features = style_model(style_images)  # style image to style feature

                image_features = model(clip_images, 'vision', style_features=style_features)
                text_features = model(labels, 'text')

                image_features = image_features / image_features.norm(dim=-1, keepdim=True) * math.sqrt(384)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True) * math.sqrt(384)

                for i in range(val_batch_size):
                    val_metric = val_metrics(image_features[i], text_features[i]).mean()
                    metrics.update(val_metric.detach(), 1)

        metric = metrics.avg.detach().cpu().numpy()
        gc.collect()
        return metric
