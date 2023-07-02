import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from tqdm import tqdm
from utils import datasets, metrics

def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int, max_samples: int = None):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer = metrics.MultiThresholdMetric(thresholds)

    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset=run_type, no_augmentations=True, include_unlabeled=False)

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=True, drop_last=True)

    stop_step = len(dataloader) if max_samples is None else max_samples
    
    #lists of metric values
    boundary_IoU, hausdorff, closed_IoU, opened_IoU, gradient_IoU, ssim = [],[],[],[],[],[]
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step == stop_step:
                break

            imgs = batch['x'].to(device)
            y_true = batch['y'].to(device)

            y_pred = net(imgs)
            y_pred = torch.sigmoid(y_pred)

            y_true = y_true.detach()
            y_pred = y_pred.detach()
            measurer.add_sample(y_true, y_pred)

            boundary_IoU.append(metrics.boundary_IoU(y_true, y_pred))
            closed_IoU.append(metrics.closed_IoU(y_true, y_pred))
            opened_IoU.append(metrics.opened_IoU(y_true, y_pred))
            gradient_IoU.append(metrics.gradient_IoU(y_true, y_pred))
            ssim.append(metrics.ssim(y_true, y_pred))


            if cfg.DEBUG:
                break
        

    print(f'Computing {run_type} F1 score ', end=' ', flush=True)

    f1s = measurer.compute_f1()
    precisions, recalls, IoU = measurer.precision, measurer.recall, measurer.IoU

    # best f1 score for passed thresholds
    f1 = f1s.max()
    argmax_f1 = f1s.argmax()

    best_thresh = thresholds[argmax_f1]
    precision = precisions[argmax_f1]
    recall = recalls[argmax_f1]
    IoU = IoU[argmax_f1]

    boundary_IoU = torch.mean(torch.stack(boundary_IoU))
    #hausdorff_fusion = torch.mean(torch.stack(hausdorff_fusion))
    closed_IoU = torch.mean(torch.stack(closed_IoU))
    opened_IoU = torch.mean(torch.stack(opened_IoU))
    gradient_IoU = torch.mean(torch.stack(gradient_IoU))
    ssim = torch.mean(torch.stack(ssim))

    print(f'{f1.item():.3f}', flush=True)

    wandb.log({f'{run_type} F1 {cfg.NAME}': f1,
               f'{run_type} threshold {cfg.NAME}': best_thresh,
               f'{run_type} precision {cfg.NAME}': precision,
               f'{run_type} recall {cfg.NAME}': recall,
               f'{run_type} IoU {cfg.NAME}': IoU,
               f'{run_type} boundary IoU {cfg.NAME}': boundary_IoU,
               f'{run_type} closed IoU {cfg.NAME}': closed_IoU,
               f'{run_type} opened IoU {cfg.NAME}': opened_IoU,
               f'{run_type} gradient IoU {cfg.NAME}': gradient_IoU,
               f'{run_type} SSIM {cfg.NAME}': ssim,
               'step': step, 'epoch': epoch,
               })
    

def get_rgb(x):

    quantile = 95

    rgb = np.flip(x[2:5].permute(1,2,0).cpu().numpy(),axis=2)

    maxi = np.percentile(rgb[:,:,0].flatten(),quantile)
    mini = np.percentile(rgb[:,:,0].flatten(),100-quantile)
    rgb[:,:,0] = np.where(rgb[:,:,0] > maxi, maxi, rgb[:,:,0])
    rgb[:,:,0] = np.where(rgb[:,:,0] < mini, mini, rgb[:,:,0])
    rgb[:,:,0] = (rgb[:,:,0]-mini)/(maxi-mini)

    maxi = np.percentile(rgb[:,:,1].flatten(),quantile)
    mini = np.percentile(rgb[:,:,1].flatten(),100-quantile)
    rgb[:,:,1] = np.where(rgb[:,:,1] > maxi, maxi, rgb[:,:,1])
    rgb[:,:,1] = np.where(rgb[:,:,1] < mini, mini, rgb[:,:,1])
    rgb[:,:,1] = (rgb[:,:,1]-mini)/(maxi-mini)

    maxi = np.percentile(rgb[:,:,2].flatten(),quantile)
    mini = np.percentile(rgb[:,:,2].flatten(),100-quantile)
    rgb[:,:,2] = np.where(rgb[:,:,2] > maxi, maxi, rgb[:,:,2])
    rgb[:,:,2] = np.where(rgb[:,:,2] < mini, mini, rgb[:,:,2])
    rgb[:,:,2] = (rgb[:,:,2]-mini)/(maxi-mini)

    return rgb


def model_testing(net, cfg, device, step, epoch):
    net.to(device)
    net.eval()

    dataset = datasets.SpaceNet7Dataset(cfg)

    y_true_dict = {'test': []}
    y_pred_dict = {'test': []}
    boundary_IoU, hausdorff, closed_IoU, opened_IoU, gradient_IoU, ssim = [],[],[],[],[],[]


    for index in range(len(dataset)):
        sample = dataset.__getitem__(index)

        with torch.no_grad():
            x = sample['x'].to(device)
            y_true = sample['y'].to(device)
            y_true = y_true[None, :]

            logits = net(x.unsqueeze(0))
            y_pred = torch.sigmoid(logits) #> 0.5

            boundary_IoU.append(metrics.boundary_IoU(y_true, y_pred).item())
            #hausdorff_fusion.append(metrics.hausdorff(y_true, y_pred_fusion))
            closed_IoU.append(metrics.closed_IoU(y_true, y_pred).item())
            opened_IoU.append(metrics.opened_IoU(y_true, y_pred).item())
            gradient_IoU.append(metrics.gradient_IoU(y_true, y_pred).item())
            ssim.append(metrics.ssim(y_true, y_pred).item())


            y_true = y_true.detach().cpu().flatten().numpy()
            y_pred = y_pred.detach().cpu().flatten().numpy()

            """region = sample['region']
            if region not in y_true_dict.keys():
                y_true_dict[region] = [y_true]
                y_pred_dict[region] = [y_pred]
            else:
                y_true_dict[region].append(y_true)
                y_pred_dict[region].append(y_pred)"""

            y_true_dict['test'].append(y_true)
            y_pred_dict['test'].append(y_pred)
    
    boundary_IoU = np.mean(boundary_IoU)
    #hausdorff_fusion = torch.mean(torch.stack(hausdorff_fusion))
    closed_IoU = np.mean(closed_IoU)
    opened_IoU = np.mean(opened_IoU)
    gradient_IoU = np.mean(gradient_IoU)
    ssim = np.mean(ssim)

    # add always the same image at testtime and add it to wandb
    mode = cfg.DATALOADER.MODE
    cfg.DATALOADER.MODE = "fusion"
    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset='training')
    sample = dataset.__getitem__(0, aug=False) # random index

    with torch.no_grad():
        x = sample['x'].to(device)
        rgb_train = get_rgb(x)
        if mode == "sar":
            x = x[:2, ]
        if mode == "optical":
            x = x[2:, ]

        train_shape = x.shape[1:]
        y_true = sample['y'].to(device)
        logits = net(x.unsqueeze(0))
        y_pred = torch.sigmoid(logits) > 0.5
        
        y_true_train = y_true.detach().cpu().flatten().numpy().reshape(train_shape)
        y_pred_train = y_pred.detach().cpu().flatten().numpy().reshape(train_shape)

    dataset = datasets.SpaceNet7Dataset(cfg)
    sample = dataset.__getitem__(0) # random index

    with torch.no_grad():
        x = sample['x'].to(device)[:,:391,:391]
        rgb_test = get_rgb(x)
        if mode == "sar":
            x = x[:2, ]
        if mode == "optical":
            x = x[2:, ]
        
        test_shape = x.shape[1:]
        y_true = sample['y'].to(device)[:,:391,:391]
        logits = net(x.unsqueeze(0))
        y_pred = torch.sigmoid(logits) > 0.5

        y_true_test = y_true.detach().cpu().flatten().numpy().reshape(test_shape)
        y_pred_test = y_pred.detach().cpu().flatten().numpy().reshape(test_shape)

    y_true_train = wandb.Image(y_true_train, caption= "GT")
    y_pred_train = wandb.Image(y_pred_train, caption= "Pred")
    Train_rgb = wandb.Image(rgb_train, caption="Train RGB", mode="RGB") #[:,:,::-1]

    y_true_test = wandb.Image(y_true_test, caption= "GT")
    y_pred_test = wandb.Image(y_pred_test, caption= "Pred")
    Test_rgb = wandb.Image(rgb_test, caption="Test RGB", mode="RGB") # [:,:,::-1]

    wandb.log({"Output Test": [Test_rgb, y_true_test, y_pred_test],
                "Output Train": [Train_rgb, y_true_train, y_pred_train]})

    def evaluate_region(region_name: str):
        y_true_region = torch.Tensor(np.concatenate(y_true_dict[region_name]))
        y_pred_region = torch.Tensor(np.concatenate(y_pred_dict[region_name]))
        prec = metrics.precision(y_true_region, y_pred_region, dim=0).item()
        rec = metrics.recall(y_true_region, y_pred_region, dim=0).item()
        f1 = metrics.f1_score(y_true_region, y_pred_region, dim=0).item()
        IoU = metrics.IoU(y_true_region, y_pred_region, dim=0).item()


        wandb.log({f'{region_name} F1 {cfg.NAME}': f1,
                   f'{region_name} precision {cfg.NAME}': prec,
                   f'{region_name} recall {cfg.NAME}': rec,
                   f'{region_name} IoU {cfg.NAME}': IoU,
                   f'{region_name} boundary IoU {cfg.NAME}': boundary_IoU,
                   f'{region_name} closed IoU {cfg.NAME}': closed_IoU,
                   f'{region_name} opened IoU {cfg.NAME}': opened_IoU,
                   f'{region_name} gradient IoU {cfg.NAME}': gradient_IoU,
                   f'{region_name} SSIM {cfg.NAME}': ssim,
                   'step': step, 'epoch': epoch,
                   })

    """for region in dataset.regions['regions'].values():
        evaluate_region(region)"""
    evaluate_region('test')
    cfg.DATALOADER.MODE = mode

   