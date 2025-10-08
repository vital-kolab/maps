from captum.attr import IntegratedGradients, Saliency, Deconvolution, InputXGradient, GuidedBackprop, NoiseTunnel, IntegratedGradients, Occlusion, GradientShap, FeatureAblation, FeaturePermutation
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import timm
import sys
import json
import re
import sys
from utils import create_i1_test

num_classes = 10
batch_size = 32
image_size = 224

device = torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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


# Paths to saved models
model_paths = {
    'resnet50': 'models/resnet50_fine_tuned.pth',
    'resnet18': 'models/resnet18_fine_tuned.pth',
    'alexnet': 'models/alexnet_fine_tuned.pth',
    'convnext': 'models/convnext_fine_tuned.pth',
    'vgg19': 'models/vgg19_fine_tuned.pth',
    'vgg16': 'models/vgg16_fine_tuned.pth',
    'vit': 'models/vit_fine_tuned.pth',
    'vit_ssl': 'models/vit_ssl_fine_tuned.pth',
    'resnet_ssl': 'models/resnet_ssl_fine_tuned.pth',
    'efficientnet': 'models/efficientnet_fine_tuned.pth',
    'swin': 'models/swin_fine_tuned.pth'
}

# Initialize models with pretrained weights
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

# Adjust the final layer 
models_dict['resnet50'].fc = torch.nn.Linear(models_dict['resnet50'].fc.in_features, num_classes)
models_dict['resnet18'].fc = torch.nn.Linear(models_dict['resnet18'].fc.in_features, num_classes)
models_dict['alexnet'].classifier[6] = torch.nn.Linear(models_dict['alexnet'].classifier[6].in_features, num_classes)
models_dict['vit'].heads.head = torch.nn.Linear(models_dict['vit'].heads.head.in_features, num_classes)
models_dict['vgg16'].classifier[6] = torch.nn.Linear(models_dict['vgg16'].classifier[6].in_features, num_classes)
models_dict['vgg19'].classifier[6] = torch.nn.Linear(models_dict['vgg19'].classifier[6].in_features, num_classes)
models_dict['convnext'].head.fc = torch.nn.Linear(1024, num_classes)
models_dict['vit_ssl'].head = nn.Linear(384, num_classes)
models_dict['resnet_ssl'].fc = torch.nn.Linear( (models.resnet50(pretrained=True)).fc.in_features, num_classes)
models_dict['efficientnet'].classifier = nn.Linear(models_dict['efficientnet'].classifier.in_features, num_classes)
models_dict['swin'].head.fc = torch.nn.Linear(1024, num_classes)

# Load the fine-tuned weights
for model_name, model in models_dict.items():
    model.load_state_dict(torch.load(model_paths[model_name], map_location=torch.device('cpu')))  # Adjust device if needed
    model.eval()  # Set the model to evaluation mode

print("Models successfully loaded and ready for inference!")

# Custom sorting to ensure numerical order if filenames don't have leading zeros
test_dataset = datasets.ImageFolder(root=f'coco200_perclass', transform=transform)
test_dataset.samples.sort(key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].replace('im', '')))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

pattern_emi = re.compile(r'pEMI_(\d+)\.npy')

os.makedirs('./behavioral_responses', exist_ok=True)

ref_model_name = sys.argv[1]
percentile = sys.argv[2]
method_name = sys.argv[3]
print("Ref", ref_model_name)
    
for target_model_name, target_model in models_dict.items():
    print("Target", target_model_name)
    target_model.eval()
    target_model.to(device)
    
    emis_dir = f'./perturbed_images/{ref_model_name}/{method_name}/{percentile}'
    print(os.listdir(emis_dir))
    emis_files = sorted(
        [f for f in os.listdir(emis_dir) if pattern_emi.match(f)],
        key=lambda x: int(pattern_emi.match(x).group(1))
    )
    
    all_outputs_ref = []
    all_outputs_target = []

    # Save behavioral responses and i1 scores progressively
    response_dir = f'./behavioral_responses/{ref_model_name}/{method_name}/{percentile}'
    if not os.path.exists(f"{response_dir}/{target_model_name}_i1.json"):
        os.makedirs(response_dir, exist_ok=True)
        for file in emis_files:
            EMI_tensor = torch.tensor(np.load(os.path.join(emis_dir, file))).unsqueeze(0).to(device)

            output_target = target_model(EMI_tensor).softmax(dim=1).detach().cpu().numpy().tolist()

            all_outputs_target.append(output_target)

        print(len(all_outputs_target))
        i1_scores = {
            'target_pEMI': create_i1_test(np.vstack(all_outputs_target), target_model_name).tolist()
        }
        
        with open(f'{response_dir}/{target_model_name}_behavioral.json', 'w') as f:
            json.dump({
                'target_pEMI': all_outputs_target,
            }, f)

        with open(f'{response_dir}/{target_model_name}_i1.json', 'w') as f:
            json.dump(i1_scores, f)