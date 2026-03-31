import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from captum.attr import IntegratedGradients, Saliency, Deconvolution, InputXGradient, GuidedBackprop, NoiseTunnel, IntegratedGradients, Occlusion, GradientShap, FeatureAblation, FeaturePermutation
import numpy as np
import os
import timm
import argparse

# Argument parser for model and method selection
parser = argparse.ArgumentParser(description='Run attribution methods in parallel.')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model to process')
parser.add_argument('--method_name', type=str, required=True, help='Attribution method to use')
parser.add_argument('--device', type=int, default=0, help='GPU device ID')
args = parser.parse_args()

# Setup device
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

# Dataset preparation
num_classes = 10
batch_size = 32
image_size = 224

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root='./coco200_perclass', transform=transform)
test_dataset.samples.sort(key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].replace('im', '')))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

models_dir = "./models"

# Load model
model_paths = {
    'resnet50': f'{models_dir}/resnet50_fine_tuned.pth',
    'resnet18': f'{models_dir}/resnet18_fine_tuned.pth',
    'alexnet': f'{models_dir}/alexnet_fine_tuned.pth',
    'convnext': f'{models_dir}/convnext_fine_tuned.pth',
    'vgg19': f'{models_dir}/vgg19_fine_tuned.pth',
    'vgg16': f'{models_dir}/vgg16_fine_tuned.pth',
    'vit': f'{models_dir}/vit_fine_tuned.pth',
    'vit_ssl': f'{models_dir}/vit_ssl_fine_tuned.pth',
    'resnet_ssl': f'{models_dir}/resnet_ssl_fine_tuned.pth',
    'efficientnet': f'{models_dir}/efficientnet_fine_tuned.pth',
    'swin': f'{models_dir}/swin_fine_tuned.pth'
}

models_dict = {
    'resnet50': models.resnet50(pretrained=True),
    'resnet18': models.resnet18(pretrained=True),
    'alexnet': models.alexnet(pretrained=True),
    'vit': models.vit_b_16(pretrained=True),
    'vgg19': models.vgg19(pretrained=True),
    'vgg16': models.vgg16(pretrained=True),
    'convnext': timm.create_model("convnext_base", pretrained=True),
    'vit_ssl': timm.create_model("vit_small_patch16_224_dino", pretrained=True),
    'resnet_ssl': torch.hub.load('facebookresearch/dino:main', 'dino_resnet50'),
    'efficientnet': timm.create_model("efficientnet_b3", pretrained=True),
    'swin': timm.create_model("swin_base_patch4_window7_224", pretrained=True)
}

# Adjust final layer and freeze parameters
for model_name, model in models_dict.items():
    print(model_name)
    if model_name in ['resnet50', 'resnet18']:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name == "resnet_ssl":
        for param in model.parameters():
            param.requires_grad = False
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        model.to(device)
        with torch.no_grad():
            features = model.forward(dummy_input)
        num_features = features.shape[1]
        model.fc = nn.Linear(num_features, num_classes)
        for param in model.fc.parameters():
            param.requires_grad = True
    elif model_name in ['alexnet', 'vgg19', 'vgg16']:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        for param in model.classifier[6].parameters():
            param.requires_grad = True
    elif model_name == 'efficientnet':
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif model_name in ['convnext', 'swin']:
        for param in model.parameters():
            param.requires_grad = False
        model.head.fc = nn.Linear(1024, num_classes)
        for param in model.head.fc.parameters():
            param.requires_grad = True
    elif model_name == 'vit':
        for param in model.parameters():
            param.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        for param in model.heads.head.parameters():
            param.requires_grad = True
    elif model_name == 'vit_ssl':
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Linear(384, num_classes)
        for param in model.head.parameters():
            param.requires_grad = True

def generate_attributions(method_instance, input_batch, target_batch):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    if isinstance(method_instance, GradientShap):
        baseline_dist = torch.cat([
            torch.zeros_like(input_batch),
            input_batch * 0.5,
            torch.randn_like(input_batch) * 0.001
        ], dim=0).to(device)
        attribution = method_instance.attribute(
            input_batch,
            baselines=baseline_dist,
            target=target_batch,
            n_samples=50
        )

    elif isinstance(method_instance, Occlusion):
        attribution = method_instance.attribute(
            input_batch,
            strides=(1, 8, 8),
            sliding_window_shapes=(3, 15, 15),
            target=target_batch
        )

    elif isinstance(method_instance, (FeatureAblation, FeaturePermutation)):
        attribution = method_instance.attribute(
            input_batch,
            target=target_batch,
            perturbations_per_eval=32
        )

    elif isinstance(method_instance, NoiseTunnel):
        attribution = method_instance.attribute(
            input_batch,
            nt_type='smoothgrad',
            target=target_batch
        )

    else:
        attribution = method_instance.attribute(
            input_batch,
            target=target_batch
        )

    return attribution.detach().cpu().numpy()


# Load fine-tuned weights
model = models_dict[args.model_name]
model.load_state_dict(torch.load(model_paths[args.model_name], map_location=device))
model.to(device)
model.eval()

# Attribution methods
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

method_instance = attribution_methods[args.method_name](model)

# Generate and save attributions
save_dir = f'./attribution_maps/{args.model_name}/{args.method_name}'
os.makedirs(save_dir, exist_ok=True)

# Process in batches
for idx, (inputs, labels) in enumerate(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    attributions_batch = generate_attributions(method_instance, inputs, labels)

    for i in range(inputs.size(0)):
        np.save(
            f'{save_dir}/image_{idx * test_loader.batch_size + i}.npy',
            attributions_batch[i].squeeze()
        )

print(f"Attributions for {args.model_name} using {args.method_name} saved successfully!")
