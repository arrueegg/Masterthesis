import sys
from pathlib import Path
import os
import timeit

import torch
from torch import optim
from torch.utils import data as torch_data

from tabulate import tabulate
import wandb
import numpy as np

from utils import networks, datasets, loss_functions, evaluation, experiment_manager, parsers

def run_training(cfg):

    def linear_lr(step, total_steps, warmup_epochs):
        warmup_steps = int(warmup_epochs * len(dataloader))
        half = int(0.5 * total_steps)
        const = False
        if const == False:
            if step < warmup_steps:
                return float(step) / warmup_steps
            else:
                return max(0.0, float(total_steps - step) / float(total_steps - warmup_steps))
        if const == True:
            if step < warmup_steps:
                return float(step) / warmup_steps
            elif step < half:
                return 1
            else:
                return max(0.0, float(total_steps - step) / float(total_steps - half))
    
    def log_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            wandb.log({"learning_rate": lr})

    run_config = {
        'CONFIG_NAME': cfg.NAME,
        'device': device,
        'epochs': cfg.TRAINER.EPOCHS,
        'learning rate': cfg.TRAINER.LR,
        'batch size': cfg.TRAINER.BATCH_SIZE,
    }
    table = {'run config name': run_config.keys(),
             ' ': run_config.values(),
             }
    print(tabulate(table, headers='keys', tablefmt="fancy_grid", ))

    net = networks.create_network(cfg)
    net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    sar_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)
    optical_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)
    fusion_criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)
    discriminator_criterion = loss_functions.get_criterion(cfg.CONSISTENCY_TRAINER.DISCRIMINATOR_LOSS_TYPE)

    # reset the generators
    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset='training')
    print(dataset)

    dataloader_kwargs = {
        'batch_size': cfg.TRAINER.BATCH_SIZE,
        'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
        'shuffle': cfg.DATALOADER.SHUFFLE,
        'drop_last': True,
        'pin_memory': True,
    }

    custom_sampler = datasets.LabeledUnlabeledSampler(
        labeled_indices=dataset.ind_labeled,
        unlabeled_indices=dataset.ind_unlabeled,
        batch_size=cfg.TRAINER.BATCH_SIZE
    )

    dataloader = torch_data.DataLoader(dataset,sampler = custom_sampler, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.TRAINER.EPOCHS
    save_checkpoints = cfg.SAVE_CHECKPOINTS
    steps_per_epoch = len(dataloader)
    len_dataloader = len(dataloader)
    total_steps = len_dataloader * epochs
    warmup_epochs = 0.5

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: linear_lr(step, total_steps, warmup_epochs))

    # tracking variables
    global_step = epoch_float = 0

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        sar_loss_set, optical_loss_set, fusion_loss_set = [], [], []
        supervised_loss_set, discriminator_loss_set, loss_set = [], [], []
        n_labeled, n_notlabeled = 0, 0

        for i, batch in enumerate(dataloader):
            p = float(i + epoch * steps_per_epoch) / (epochs+1) / steps_per_epoch
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            #alpha = 1

            net.train()
            optimizer.zero_grad()

            x_fusion = batch['x'].to(device)
            y_gts = batch['y'].to(device)
            is_labeled = batch['is_labeled']
            y_gts = y_gts[is_labeled]

            sar_logits, optical_logits, fusion_logits, disc_logits_sar, disc_logits_optical = net(x_fusion, alpha)

            supervised_loss, discriminator_loss = None, None

            # supervised loss
            if is_labeled.any():
                sar_loss = sar_criterion(sar_logits[is_labeled], y_gts)
                sar_loss_set.append(sar_loss.item())

                optical_loss = optical_criterion(optical_logits[is_labeled], y_gts)
                optical_loss_set.append(optical_loss.item())

                fusion_loss = fusion_criterion(fusion_logits[is_labeled], y_gts)
                fusion_loss_set.append(fusion_loss.item())
                n_labeled += torch.sum(is_labeled).item()

                supervised_loss = sar_loss + optical_loss + fusion_loss
                supervised_loss_set.append(supervised_loss.item())


            # consistency loss for semi-supervised training
            if not is_labeled.all():
                not_labeled = torch.logical_not(is_labeled)
                n_notlabeled += torch.sum(not_labeled).item()

                # create one hot GT for domain classification, SAR == 1, optical == 0
                GT_shape = [disc_logits_sar.shape[0], 1, disc_logits_sar.shape[2], disc_logits_sar.shape[3]]
                dis_GT_sar = torch.ones((GT_shape)).cuda()
                dis_GT_sar = torch.nn.functional.one_hot(dis_GT_sar.to(torch.int64)).view(disc_logits_sar.shape)

                dis_GT_optical = torch.zeros((GT_shape)).cuda()
                dis_GT_optical = torch.nn.functional.one_hot(dis_GT_optical.to(torch.int64),2).view(disc_logits_optical.shape)

                # calculate discriminator loss for domain classifier
                discriminator_loss_sar = discriminator_criterion(disc_logits_sar.float(), dis_GT_sar.float())
                discriminator_loss_optical = discriminator_criterion(disc_logits_optical.float(), dis_GT_optical.float())
                discriminator_loss = discriminator_loss_sar + discriminator_loss_optical
                discriminator_loss = cfg.CONSISTENCY_TRAINER.LOSS_FACTOR * discriminator_loss
                discriminator_loss_set.append(discriminator_loss.item())
            
            loss =  discriminator_loss + supervised_loss
            loss_set.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()


            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.LOG_FREQ == 0 and not cfg.DEBUG:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')

                # evaluation on sample of training and validation set
                evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step, max_samples=1_000)
                evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step, max_samples=1_000)
                evaluation.disc_evaluation(net, cfg, device, 'validation', epoch_float, global_step, max_samples=1_000)

                # logging
                time = timeit.default_timer() - start
                labeled_percentage = n_labeled / (n_labeled + n_notlabeled) * 100
                wandb.log({
                    'sar_loss': np.mean(sar_loss_set),
                    'optical_loss': np.mean(optical_loss_set),
                    'fusion_loss': np.mean(fusion_loss_set),
                    'supervised_loss': np.mean(supervised_loss_set),
                    'consistency_loss': np.mean(discriminator_loss_set) if discriminator_loss_set else 0,
                    'loss_set': np.mean(loss_set),
                    'labeled_percentage': labeled_percentage,
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                log_learning_rate(optimizer)
                start = timeit.default_timer()
                sar_loss_set, optical_loss_set, fusion_loss_set = [], [], []
                supervised_loss_set, discriminator_loss_set, loss_set = [], [], []
                n_labeled, n_notlabeled = 0, 0

            if cfg.DEBUG:
                break
            # end of batch

        if not cfg.DEBUG:
            assert (epoch == epoch_float)
        evaluation.model_testing(net, cfg, device, global_step, epoch_float)

        if epoch in save_checkpoints and not cfg.DEBUG:
            print(f'saving network', flush=True)
            networks.save_checkpoint(net, optimizer, epoch, global_step, cfg)

            # logs to load network
            evaluation.model_evaluation(net, cfg, device, 'training', epoch_float, global_step)
            evaluation.model_evaluation(net, cfg, device, 'validation', epoch_float, global_step)


if __name__ == '__main__':

    args = parsers.training_argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)

    # make training deterministic
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('=== Runnning on device: ', device)

    wandb.init(
        name=cfg.NAME,
        config=cfg,
        entity='sensorfusion',
        project="Master_Thesis",
        tags=['run', 'urban', 'extraction', 'segmentation', ],
        mode='online' if not cfg.DEBUG else 'disabled',
    )

    try:
        run_training(cfg)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
