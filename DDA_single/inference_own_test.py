import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import networks, datasets, loss_functions, parsers, experiment_manager, metrics, geofiles
import rasterio
import os

from torch.utils import data as torch_data
import torch

args = parsers.training_argument_parser().parse_known_args()[0]
cfg = experiment_manager.setup_cfg(args)
cfg.DATALOADER.INCLUDE_UNLABELED = False

# make training deterministic
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# loading config and network
cfg.NAME = "optical"
cfg.MODEL.IN_CHANNELS = 10
cfg.DATALOADER.MODE = 'optical'
net_optical, _, _ = networks.load_checkpoint(15, cfg, device)
net_optical.eval()

cfg.NAME = "SAR"
cfg.MODEL.IN_CHANNELS = 2
cfg.DATALOADER.MODE = 'sar'
net_sar, _, _ = networks.load_checkpoint(15, cfg, device)
net_sar.eval()

cfg.NAME = "fusion"
cfg.MODEL.IN_CHANNELS = 12
cfg.DATALOADER.MODE = 'fusion'
net_fusion, _, _ = networks.load_checkpoint(15, cfg, device)
net_fusion.eval()

print("\n networks loaded \n")

# Load the dataset you used for training
dataset = datasets.SpaceNet7Dataset(cfg=cfg)
print(dataset)

dataloader_kwargs = {
    'batch_size': 1,
    'num_workers': 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER,
    'shuffle': False,
    'drop_last': False,
    'pin_memory': True,
}

dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

# Define the performance metric you want to use (e.g., accuracy)
criterion = loss_functions.get_criterion(cfg.MODEL.LOSS_TYPE)

# Lists to store example information
examples = []
examples_losses = []
examples_metrics = []

# Iterate through the dataset
for i, batch in enumerate(tqdm(dataloader)):
    x_fusion = batch['x'].to(device)
    x_optical = x_fusion[:, 2:]
    x_sar = x_fusion[:, :2]
    y_gts = batch['y'].to(device)
    patch_id = batch["aoi_id"][0]
    #site = batch["country"][0]
    geotransform = batch["transform"]
    crs = batch["crs"][0]

    # Forward pass
    fusion_logits = net_fusion(x_fusion)
    sar_logits = net_sar(x_sar)
    optical_logits = net_optical(x_optical)
    y_pred_fusion = torch.sigmoid(fusion_logits)
    y_pred_SAR = torch.sigmoid(sar_logits)
    y_pred_OPT = torch.sigmoid(optical_logits)

    # Calculate the performance metric and losses
    fusion_loss = criterion(y_pred_fusion, y_gts).item()
    optical_loss = criterion(y_pred_OPT, y_gts).item()
    sar_loss = criterion(y_pred_SAR, y_gts).item()
    consistency_loss = criterion(y_pred_OPT, y_pred_SAR).item()
    total_loss = fusion_loss + optical_loss + sar_loss + consistency_loss

    prec_fusion = metrics.precision(y_gts.flatten(), y_pred_fusion.flatten(), dim=0).item()
    rec_fusion = metrics.recall(y_gts.flatten(), y_pred_fusion.flatten(), dim=0).item()
    f1_fusion = metrics.f1_score(y_gts.flatten(), y_pred_fusion.flatten(), dim=0).item()
    IoU_fusion = metrics.IoU(y_gts.flatten(), y_pred_fusion.flatten(), dim=0).item()
    ssim_fusion = metrics.ssim(y_gts, y_pred_fusion).item()

    prec_SAR = metrics.precision(y_gts.flatten(), y_pred_SAR.flatten(), dim=0).item()
    rec_SAR = metrics.recall(y_gts.flatten(), y_pred_SAR.flatten(), dim=0).item()
    f1_SAR = metrics.f1_score(y_gts.flatten(), y_pred_SAR.flatten(), dim=0).item()
    IoU_SAR = metrics.IoU(y_gts.flatten(), y_pred_SAR.flatten(), dim=0).item()
    ssim_SAR = metrics.ssim(y_gts, y_pred_SAR).item()

    prec_OPT = metrics.precision(y_gts.flatten(), y_pred_OPT.flatten(), dim=0).item()
    rec_OPT = metrics.recall(y_gts.flatten(), y_pred_OPT.flatten(), dim=0).item()
    f1_OPT = metrics.f1_score(y_gts.flatten(), y_pred_OPT.flatten(), dim=0).item()
    IoU_OPT = metrics.IoU(y_gts.flatten(), y_pred_OPT.flatten(), dim=0).item()
    ssim_OPT = metrics.ssim(y_gts, y_pred_OPT).item()


    y_pred_fusion = y_pred_fusion.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    y_pred_SAR = y_pred_SAR.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))
    y_pred_OPT = y_pred_OPT.squeeze(0).detach().cpu().numpy().transpose((1, 2, 0))

    #save predictions
    directory = f"../GM12_GUM/spacenet7/predictions_single/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    geofiles.write_tif(Path(f"../GM12_GUM/spacenet7/predictions_single/prediction_fusion_{patch_id}.tif"), y_pred_fusion.astype(np.float32), geotransform, crs)
    geofiles.write_tif(Path(f"../GM12_GUM/spacenet7/predictions_single/prediction_SAR_{patch_id}.tif"), y_pred_SAR.astype(np.float32), geotransform, crs)
    geofiles.write_tif(Path(f"../GM12_GUM/spacenet7/predictions_single/prediction_optical_{patch_id}.tif"), y_pred_OPT.astype(np.float32), geotransform, crs)
        
    # Determine if the example is good or bad based on the metric value
    examples_losses.append(("spacenet7", patch_id, fusion_loss, optical_loss,sar_loss,consistency_loss, total_loss))
    examples_metrics.append(("spacenet7", patch_id, prec_fusion, prec_OPT, prec_SAR, rec_fusion, rec_OPT, rec_SAR,
                              f1_fusion,f1_OPT,f1_SAR,IoU_fusion,IoU_OPT,IoU_SAR,ssim_fusion,ssim_OPT,ssim_SAR))

# Sort the examples based on the metric value in descending order
examples_losses.sort(key=lambda x: x[-1], reverse=True)
examples_metrics.sort(key=lambda x: x[-9], reverse=True)

if not os.path.exists("examples"):
    os.makedirs("examples")
torch.save(examples_losses, "examples/test_examples_losses.pth")
torch.save(examples_metrics, "examples/test_examples_metrics.pth")







