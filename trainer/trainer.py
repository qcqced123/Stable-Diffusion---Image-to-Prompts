import dataset_class.dataclass as dataset_class
import model.metric as model_metric
import model.loss as model_loss
import model.model as model_arch
from torch.optim.swa_utils import AveragedModel
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
        self.df = kfold(
            load_data('./dataset_class/data_folder/all_prompt.csv'),
            self.cfg
        )
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
            batch_size=self.cfg.batch_size,
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

        criterion = getattr(model_loss, self.cfg.loss_fn)(self.cfg.reduction)
        val_metrics = getattr(model_metric, self.cfg.metrics)()

        grouped_optimizer_params = get_optimizer_grouped_parameters(
            model,
            self.cfg.layerwise_lr,
            self.cfg.layerwise_weight_decay,
            self.cfg.layerwise_lr_decay
        )
        optimizer = getattr(transformers, self.cfg.optimizer)(
            params=grouped_optimizer_params,
            lr=self.cfg.layerwise_lr,
            eps=self.cfg.layerwise_adam_epsilon,
            correct_bias=not self.cfg.layerwise_use_bertadam
        )

        lr_scheduler = get_scheduler(self.cfg, optimizer, len_train)

        return model, style_model, criterion, val_metrics, optimizer, lr_scheduler

    # Train Function
    def train_fn(self, loader_train, model, style_model, criterion, optimizer, scheduler):
        """ Training Function """
        torch.autograd.set_detect_anomaly(True)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.amp_scaler)
        global_step, score_list = 0, []
        losses = AverageMeter()
        model.train(), style_model.eval()

        for step, (images, clip_images, labels) in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            for k, v in labels.items():
                labels[k] = v.to(self.cfg.device)  # prompt to GPU

            images = images.to(self.cfg.device)  # style image to GPU
            clip_images = clip_images.to(self.cfg.device)  # clip image to GPU
            batch_size = labels.size(0)

            style_features = style_model(images)  # style image to style feature

            with torch.cuda.amp.autocast(enabled=self.cfg.amp_scaler):
                clip_features = model(clip_images, 'vision', style_inputs=style_features)
                text_features = model(labels, 'text')
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # normalize

                image_features = torch.cat([clip_features, style_features], dim=1)  # same as pandas concat

                loss = (criterion(image_features, text_features) + criterion(text_features, image_features)) / 2
                losses.update(loss, batch_size)

            if self.cfg.n_gradient_accumulation_steps > 1:
                loss = loss / self.cfg.n_gradient_accumulation_steps

            scaler.scale(loss).backward()

            if self.cfg.clipping_grad and (step + 1) % self.cfg.n_gradient_accumulation_steps == 0 or self.cfg.n_gradient_accumulation_steps == 1:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(),
                    self.cfg.max_grad_norm * self.cfg.n_gradient_accumulation_steps
                )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                scheduler.step()

        train_loss = losses.avg.detach().cpu().numpy()
        grad_norm = grad_norm.detach().cpu().numpy()
        return train_loss, grad_norm, scheduler.get_lr()[0]

    # Validation Function
    def valid_fn(self, loader_valid, model, style_model, val_metrics) -> float:
        """ Validation Functions """
        valid_metrics = AverageMeter()
        model.eval(), style_model.eval()
        with torch.no_grad():
            for step, (images, clip_images, labels) in enumerate(tqdm(loader_valid)):
                for k, v in labels.items():
                    labels[k] = v.to(self.cfg.device)  # prompt to GPU

                images = images.to(self.cfg.device)  # style image to GPU
                clip_images = clip_images.to(self.cfg.device)  # clip image to GPU
                batch_size = labels.size(0)

                style_features = style_model(images)
                style_features = style_features / style_features.norm(dim=-1, keepdim=True)  # normalize

                clip_features = model.encode_image(clip_images)
                text_features = model.encode_text(labels)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)  # normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # normalize

                image_features = torch.cat([clip_features, style_features], dim=1)

                val_metrics = val_metrics(image_features, text_features)
                valid_metrics.update(val_metrics, batch_size)
        metric = valid_metrics.avg.detach().cpu().numpy()
        return metric
