import numpy as np
import torch
from torch.utils import data as torch_data
from diffusers import AutoencoderKL
from utils import geofiles
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
from torchvision import transforms

def single_channel_to_features(img, vae, checks=True):
    if checks:
        assert torch.is_tensor(img), 'Expecting a torch tensor here'
        assert img.dtype == torch.float, f'Expecting a floating point data type, found {img.dtype}'
        assert img.shape == (1, 512, 512), f'We expect resolution 1x512x512, given {img.shape}'
        assert img.min() >= -1 and img.max() <= 1, f'The data range must be strictly within [-1, 1], found [{img.min()}, {img.max()}]'
        # assert img.min() < 0 and img.max() > 0, 'The data range seems not zero-mean'
    if torch.cuda.is_available():
        img = img.to('cuda')
    img = img.repeat((3, 1, 1))[None]
    with torch.no_grad():
        out = vae.encode(img).latent_dist.sample() * vae.config.scaling_factor
    out = out[0]
    return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "stabilityai/stable-diffusion-2-1-base"
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to("cuda")
#print(vae)

data_path = "../GM12_GUM"  # Replace with the path to your folder
city_names = []
for filename in os.listdir(data_path):
    if not filename.endswith(".gz") and not filename.endswith(".txt"):
        city_names.append(filename)

print(city_names[-16:])

convert_array_to_vae_input = [
    transforms.ToTensor(),
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((512, 512))
]
transform_vae = transforms.Compose(convert_array_to_vae_input)

for city in city_names:
    city_path_sen1 = os.path.join(data_path, city, "sentinel1")
    city_path_sen2 = os.path.join(data_path, city, "sentinel2")
    print(city)
    for sat in [city_path_sen1, city_path_sen2]:
        for image in tqdm(os.listdir(sat)):
            if image.endswith(".tif"):
                image_path = os.path.join(sat, image)
                if city == "spacenet7":
                    patch_coordinates = image_path[:-4].split('/')[-1].split('_')
                    sample_id = int(patch_coordinates[2]) + int(patch_coordinates[3]) + int(patch_coordinates[4]) # summed patch coordinates (x + y + z)
                    generator = torch.Generator().manual_seed(sample_id)
                else:
                    patch_coordinates = image_path[:-4].split('/')[-1].split('_')[-1].split('-')
                    sample_id = int(patch_coordinates[0]) + int(patch_coordinates[1]) # summed patch coordinates (x + y)
                    generator = torch.Generator().manual_seed(sample_id)
                img, transform, crs = geofiles.read_tif(Path(image_path))
                img = transform_vae(img)
                for i in range(img.shape[0]):
                    channel = img[i] * 2.0 - 1.0
                    channel = channel.to(device).unsqueeze(0)                    
                    output = single_channel_to_features(channel, vae)
                    output = output.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
                    image_path_new = "../GM12_GUM_new"+image_path[11:]
                    image_path_new = os.path.join(image_path_new[:-4]+"_"+str(i)+".tif")
                    geofiles.write_tif(Path(image_path_new), output.astype(np.float32), transform, crs)

