import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
import os
import timm

num_classes = 10
batch_size = 32
image_size = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=f'coco1400_perclass', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root=f'coco200_perclass', transform=transform)
test_dataset.samples.sort(key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0].replace('im', '')))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

models_dict = {
    'resnet50': models.resnet50(pretrained=True),
    'alexnet': models.alexnet(pretrained=True),
    'convnext': timm.create_model("convnext_base", pretrained=True),
    'vgg19': models.vgg19(pretrained=True),
    'vit': models.vit_b_16(pretrained=True),
    'resnet18': models.resnet18(pretrained=True),
    'vgg16': models.vgg16(pretrained=True),
    "vit_ssl": timm.create_model("vit_small_patch16_224_dino", pretrained=True),
    "resnet_ssl": torch.hub.load('facebookresearch/dino:main', 'dino_resnet50'),
    "efficientnet": timm.create_model("efficientnet_b3", pretrained=True),
    "swin": timm.create_model("swin_base_patch4_window7_224", pretrained=True)
}

for model_name, model in models_dict.items():
    print(model_name)
    if model_name == 'resnet50' or model_name == "resnet18":
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
    elif model_name == 'alexnet' or model_name == "vgg19" or model_name == "vgg16":
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
    elif model_name == 'convnext' or model_name == "swin":
        for param in model.parameters():
            param.requires_grad = False
        model.head.fc = torch.nn.Linear(1024, num_classes)
        for param in model.head.fc.parameters():
            param.requires_grad = True
    elif model_name == 'vit':
        for param in model.parameters():
            param.requires_grad = False
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        for param in model.heads.head.parameters():
            param.requires_grad = True
    elif model_name == 'vit_ssl':
        for param in model.parameters():
            param.requires_grad = False
        model.head = torch.nn.Linear(384, num_classes)
        for param in model.head.parameters():
            param.requires_grad = True
    

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    model.train()
    for epoch in range(5):  # Fine-tune for 5 epochs
        print(epoch)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    models_dict[model_name] = model

for model_name, model in models_dict.items():
    torch.save(model.state_dict(), f'models/{model_name}_fine_tuned.pth')