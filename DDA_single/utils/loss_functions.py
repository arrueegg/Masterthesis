import torch
import torch.nn as nn
from torch.nn import functional as F


def get_criterion(loss_type, negative_weight: float = 1, positive_weight: float = 1):

    if loss_type == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'CrossEntropyLoss':
        balance_weight = [negative_weight, positive_weight]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        balance_weight = torch.tensor(balance_weight).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=balance_weight)
    elif loss_type == 'SoftDiceLoss':
        criterion = soft_dice_loss
    elif loss_type == 'SoftDiceSquaredSumLoss':
        criterion = soft_dice_squared_sum_loss
    elif loss_type == 'SoftDiceBalancedLoss':
        criterion = soft_dice_loss_balanced
    elif loss_type == 'PowerJaccardLoss':
        criterion = power_jaccard_loss
    elif loss_type == 'MeanSquareErrorLoss':
        criterion = nn.MSELoss()
    elif loss_type == 'IoULoss':
        criterion = iou_loss
    elif loss_type == 'DiceLikeLoss':
        criterion = dice_like_loss
    elif loss_type == 'ContrastiveLoss':
        criterion = contrastiveLossOWN
    elif loss_type == 'MMD':
        criterion = mmd_loss
    elif loss_type == 'L1_loss':
        criterion = pixelwise_l1
    elif loss_type == 'L2_loss':
        criterion = pixelwise_l2
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion

def pixelwise_l1(input1, input2):
    # Flatten the inputs to compute the L2 distance on a pixelwise basis
    input1_flat = input1.view(input1.size(0), -1)
    input2_flat = input2.view(input2.size(0), -1)
    
    # Compute the L2 distance between the flattened inputs
    l1_distance = torch.abs(input1_flat - input2_flat)
    
    # Compute the mean L2 distance over all pixels in the input
    mean_l1_distance = torch.mean(l1_distance.view(l1_distance.size(0), -1), dim=1)
    
    return mean_l1_distance.mean()

def pixelwise_l2(input1, input2):
    # Flatten the inputs to compute the L2 distance on a pixelwise basis
    input1_flat = input1.view(input1.size(0), -1)
    input2_flat = input2.view(input2.size(0), -1)
    
    # Compute the L2 distance between the flattened inputs
    l2_distance = torch.nn.functional.mse_loss(input1_flat, input2_flat, reduction='none')
    
    # Compute the mean L2 distance over all pixels in the input
    mean_l2_distance = torch.mean(l2_distance.view(l2_distance.size(0), -1), dim=1)
    
    return mean_l2_distance.mean()

def rbf_kernel(X, Y, sigma=1):
    """
    Computes the RBF kernel between two matrices of flattened vectors
    """
    M = X.shape[0]
    N = Y.shape[0]
    X = X.view(M, -1)
    Y = Y.view(N, -1)
    KXX = torch.sum(X*X, dim=1, keepdim=True).expand(M, M)
    KYY = torch.sum(Y*Y, dim=1, keepdim=True).expand(N, N)
    KXY = torch.matmul(X, Y.t())
    dist = KXX + KYY.t() - 2*KXY
    kernel = torch.exp(-1 / (2*sigma**2) * dist)
    return kernel

def mmd_loss(X, Y, sigma=30):
    """
    Computes the maximum mean discrepancy (MMD) between two sets of feature maps
    """
    X = F.max_pool2d(X, kernel_size=3, stride=3)
    Y = F.max_pool2d(Y, kernel_size=3, stride=3)

    KXX = rbf_kernel(X, X, sigma=sigma)
    KYY = rbf_kernel(Y, Y, sigma=sigma)
    KXY = rbf_kernel(X, Y, sigma=sigma)
    loss = torch.mean(KXX) + torch.mean(KYY) - 2 * torch.mean(KXY)
    return loss

def MMD(x, y, sigma=1):
    """
    Compute the maximum mean discrepancy (MMD) between two sets of samples.
    Args:
        x (torch.Tensor): Tensor of size (batch_size, channels, height_1, width_1).
        y (torch.Tensor): Tensor of size (batch_size, channels, height_2, width_2).
        kernel (callable): Kernel function to use for computing the MMD.
        sigma (float): Standard deviation of the kernel function.
    Returns:
        torch.Tensor: MMD between the two sets of samples.
    """
    K_xx = rbf_kernel(x, x, sigma=sigma)
    K_xy = rbf_kernel(x, y, sigma=sigma)
    K_yy = rbf_kernel(y, y, sigma=sigma)

    mmd = torch.mean(K_xx) - 2 * torch.mean(K_xy) + torch.mean(K_yy)
    return mmd

def rbf_kernel_old(x, y, sigma=1.0):
    """
    Compute the RBF kernel matrix between two sets of samples.
    Args:
        x (torch.Tensor): Tensor of size (batch_size, channels, height_1, width_1).
        y (torch.Tensor): Tensor of size (batch_size, channels, height_2, width_2).
        sigma (float): Standard deviation of the RBF kernel.
    Returns:
        torch.Tensor: RBF kernel matrix of size (batch_size, height_1 * width_1, height_2 * width_2).
    """
    x = x.view(x.shape[0], x.shape[1], -1)
    y = y.view(y.shape[0], y.shape[1], -1)

    K = torch.zeros(x.shape[0], x.shape[2], y.shape[2]).to(x.device)
    for c in range(x.shape[1]):
        x_c = x[:, c, :]
        y_c = y[:, c, :]
        dist = torch.sum(x_c ** 2, dim=1, keepdim=True) + \
               torch.sum(y_c ** 2, dim=1, keepdim=True).t() - \
               2 * torch.matmul(x_c, y_c.t())
        K += torch.exp(-1.0 / (2 * sigma ** 2) * dist)
    return K

def contrastiveLossOWN(radar_batch, optical_batch, subsetting=10, temp=0.5):
    
    # sample only the center pixel
    optical_batch_subset = optical_batch[:,:,::subsetting,::subsetting].permute(0,2,3,1).reshape(-1,optical_batch.shape[1])
    radar_batch_subset = radar_batch[:,:,::subsetting,::subsetting].permute(0,2,3,1).reshape(-1,optical_batch.shape[1])

    optical_batch_subset = torch.nn.functional.normalize(optical_batch_subset, p=2, dim=1)
    radar_batch_subset = torch.nn.functional.normalize(radar_batch_subset, p=2, dim=1)

    # with concatenation
    """
    features = torch.cat([optical_batch_subset,radar_batch_subset],0)
    print(features.shape)
    sim_matrix = torch.exp( (features @ features.T) / temp )
    """

    # without concatenation
    expo = (optical_batch_subset @ radar_batch_subset.T) / temp
    #print(expo.min(), expo.max())
    sim_matrix = torch.exp( expo )    

    # calculate nominator and denominator with pos and neg pairs
    nom = torch.diag(sim_matrix).sum() # positive

    # takes negative elements from off diagonals (pairs of same pixel locations in different images for different locations)
    neg_diags = torch.cat([torch.diagonal(sim_matrix, offset=i*optical_batch_subset.shape[-1]) for i in range(1, 8)] + 
                        [torch.diagonal(sim_matrix, offset=-i*optical_batch_subset.shape[-1]) for i in range(1, 8)])
    
    denom = neg_diags.sum() + 1e-8 # negativ [::5]

    loss = -torch.log(nom/denom + 1e-8)
    #print(loss, "loss") 
    return loss

def soft_dice_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_prob = torch.sigmoid(y_logit)
    eps = 1e-6

    y_prob = y_prob.flatten()
    y_true = y_true.flatten()
    intersection = (y_prob * y_true).sum()

    return 1 - ((2. * intersection + eps) / (y_prob.sum() + y_true.sum() + eps))


# TODO: fix this one
def soft_dice_squared_sum_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_prob = torch.sigmoid(y_logit)
    eps = 1e-6

    y_prob = y_prob.flatten()
    y_true = y_true.flatten()
    intersection = (y_prob * y_true).sum()

    return 1 - ((2. * intersection + eps) / (y_prob.sum() + y_true.sum() + eps))


def soft_dice_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def soft_dice_loss_multi_class_debug(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y.sum(dim=sum_dims) + p.sum(dim=sum_dims)).clamp(eps)

    loss = 1 - (2. * intersection / denom).mean()
    loss_components = 1 - 2 * intersection/denom
    return loss, loss_components


def generalized_soft_dice_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-12

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width
    ysum = y.sum(dim=sum_dims)
    wc = 1 / (ysum ** 2 + eps)
    intersection = ((y * p).sum(dim=sum_dims) * wc).sum()
    denom =  ((ysum + p.sum(dim=sum_dims)) * wc).sum()

    loss = 1 - (2. * intersection / denom)
    return loss


def jaccard_like_loss_multi_class(input:torch.Tensor, y:torch.Tensor):
    p = torch.softmax(input, dim=1)
    eps = 1e-6

    # TODO [B, C, H, W] -> [C, B, H, W] because softdice includes all pixels

    sum_dims= (0, 2, 3) # Batch, height, width

    intersection = (y * p).sum(dim=sum_dims)
    denom =  (y ** 2 + p ** 2).sum(dim=sum_dims) + (y*p).sum(dim=sum_dims) + eps

    loss = 1 - (2. * intersection / denom).mean()
    return loss


def jaccard_like_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps

    return 1 - ((2. * intersection) / denom)


def dice_like_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() + eps

    return 1 - ((2. * intersection) / denom)


def power_jaccard_loss(input: torch.Tensor, target: torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps

    return 1 - (intersection / denom)


def iou_loss(y_logit: torch.Tensor, y_true: torch.Tensor):
    y_pred = torch.sigmoid(y_logit)
    eps = 1e-6

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    intersection = (y_pred * y_true).sum()
    union = (y_pred + y_true).sum() - intersection + eps

    return 1 - (intersection / union)


def jaccard_like_balanced_loss(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()
    denom = (iflat**2 + tflat**2).sum() - (iflat * tflat).sum() + eps
    piccard = (2. * intersection)/denom

    n_iflat = 1-iflat
    n_tflat = 1-tflat
    neg_intersection = (n_iflat * n_tflat).sum()
    neg_denom = (n_iflat**2 + n_tflat**2).sum() - (n_iflat * n_tflat).sum()
    n_piccard = (2. * neg_intersection)/neg_denom

    return 1 - piccard - n_piccard


def soft_dice_loss_balanced(input:torch.Tensor, target:torch.Tensor):
    input_sigmoid = torch.sigmoid(input)
    eps = 1e-6

    iflat = input_sigmoid.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    dice_pos = ((2. * intersection) /
                (iflat.sum() + tflat.sum() + eps))

    negatiev_intersection = ((1-iflat) * (1 - tflat)).sum()
    dice_neg =  (2 * negatiev_intersection) / ((1-iflat).sum() + (1-tflat).sum() + eps)

    return 1 - dice_pos - dice_neg