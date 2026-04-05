#!/usr/bin/env bash
set -euo pipefail

SCRIPT="./get_attributions_gpus.py"

MODELS=(vgg19 vgg16 vit swin vit_ssl resnet_ssl efficientnet resnet50 resnet18 alexnet convnext)
METHODS=(Saliency NoiseTunnel_Saliency Deconvolution InputXGradient GuidedBackprop GradientShap IntegratedGradients NoiseTunnel_Deconvolution NoiseTunnel_InputXGradient Occlusion FeatureAblation FeaturePermutation)

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo "model=${model} | method=${method} "
    python "$SCRIPT" \
      --model_name "$model" \
      --method_name "$method" 
  done
done

echo "All done."
