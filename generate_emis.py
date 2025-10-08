from captum.attr import IntegratedGradients, Saliency, Deconvolution, InputXGradient, GuidedBackprop, NoiseTunnel, IntegratedGradients, Occlusion, GradientShap, FeatureAblation, FeaturePermutation
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import timm
import sys


# 3. Generate Attribution Maps
attribution_methods = {
    'Saliency': Saliency,
    'NoiseTunnel_Saliency': lambda model: NoiseTunnel(Saliency(model)),
    'Deconvolution': Deconvolution,
    'InputXGradient': InputXGradient,
    'GuidedBackprop': GuidedBackprop,
    'GradientShap': GradientShap,
    'Occlusion': Occlusion,
    'IntegratedGradients': IntegratedGradients,
    'NoiseTunnel_Deconvolution': lambda model: NoiseTunnel(Deconvolution(model)),
    'NoiseTunnel_InputXGradient': lambda model: NoiseTunnel(InputXGradient(model)),
    'FeatureAblation': FeatureAblation,
    'FeaturePermutation': FeaturePermutation
}

# Functions for normalization and pixel selection
def normalize_channels(image):
    image = image.copy().transpose(1, 2, 0)
    image = np.abs(image)
    for channel in range(image.shape[2]):
        max_value = image[..., channel].max()
        if max_value > 0:
            image[..., channel] /= max_value
    return image

def select_pixels(maps, percentile=95):
    threshold = np.percentile(maps, percentile)
    mask = maps >= threshold
    mask = np.any(mask, axis=2)  # Reduce to 2D mask across channels
    return mask

def generate_perturbed_images(input_image, attribution_map, percentile, positive=True):
    normalized_attr_map = normalize_channels(attribution_map)
    mask = select_pixels(normalized_attr_map, percentile)
    grey_value = np.array([0.5, 0.5, 0.5]).reshape(1, 1, 3)

    perturbed_image = input_image.clone().permute(1, 2, 0).cpu().numpy()
    if positive:
        perturbed_image[~mask] = grey_value
    else:
        perturbed_image[mask] = grey_value

    return perturbed_image.transpose(2, 0, 1)

percentiles = [50, 60, 65, 70, 75, 80, 85, 90, 95, 97, 98, 99]

num_classes = 10
batch_size = 32
image_size = 224

device = torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root=f'coco200_perclass', transform=transform)
test_dataset.samples.sort(key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].replace('im', '')))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model_name = sys.argv[1]
print(model_name)
for method_name in attribution_methods.keys():
    print(method_name)
    for percentile in percentiles:
        save_dir = f'./perturbed_images/{model_name}/{method_name}/{percentile}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
            for idx, (inputs, labels) in enumerate(test_loader):
                for i in range(inputs.size(0)):
                    attr_map = np.load(f"./attribution_maps/{model_name}/{method_name}/image_{idx * test_loader.batch_size + i}.npy") 
                    pEMI = generate_perturbed_images(inputs[i], attr_map, percentile, positive=True)

                    np.save(f'{save_dir}/pEMI_{idx * test_loader.batch_size + i}.npy', pEMI)
