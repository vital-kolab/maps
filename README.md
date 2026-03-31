# 🧠 MAPS: Masked Attribution-based Probing of Strategies 
**Official implementation of**  
*MAPS: Masked Attribution-based Probing of Strategies – A computational framework to align human and model explanations*  

[![OSF Data](https://img.shields.io/badge/Data-OSF-blue.svg)](https://osf.io/)  
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()  
[![License](https://img.shields.io/badge/license-MIT-green)]()  

---

## 🧩 Overview  

**MAPS** is a computational framework for testing whether visual explanations from deep neural networks capture human-like and primate-like visual strategies.  
The repository provides all scripts and notebooks to:  
- 𝌭 Fine-tune models on an object-recognition dataset  
- 🌈 Generate attribution maps and **Explanation-Masked Inputs (EMIs)**  
- 📊 Compute behavioral metrics (**B.I1**) and correlations with humans, monkeys, and neurons  
- 🔍 Compare attribution methods and assess explanation similarity (L1, L2, LPIPS)  
- 📈 Reproduce all figures and analyses from the MAPS paper  

Precomputed data and psychophysics datasets are publicly hosted on the [Open Science Framework (OSF)](https://osf.io/t6kw8/).

---

## ⚙️ Installation  (< 5 mins)

```bash
git clone https://github.com/vital-kolab/maps.git
cd maps
python -m venv maps_env
source maps_env/bin/activate
pip install ipykernel numpy scipy scikit-learn matplotlib h5py torch torchvision timm pandas captum
```

GPU acceleration (CUDA) is recommended for attribution generation.

---

## 🚀 Quickstart Pipeline  

### **1️⃣ Fine-tune and tests models** (5 mins per model on a single GPU)

Train the selected architectures on your dataset:
```bash
python finetune_models.py
```

Then test the finetuned models in the same way as humans on the 2AFC task:
```bash
jupyter notebook test_models.ipynb
```

- By default, the scripts uses the datasets in `./coco1400_per_class/` (training) and `./coco200_per_class/` (test).  
- To use your own data, change `./coco1400_per_class/` and `./coco200_per_class/` to your dataset folder name.

---

### **2️⃣ Select the best model** (< 2 mins)

Determine which model best matches human behavior:
```bash
jupyter notebook get_best_model.ipynb
```

- To ensure proper cross-validation throughout the paper, we used an additional dataset to identify the best model (images not provided but model and human behavior in the OSF directory).
- To use your own data, place your files in:  
  ```
  behavioral_responses/humans/
  behavioral_responses/{model}/
  ```
  where `model` is the name of each model within the pipeline

---

### **3️⃣ Generate explanations** (200 images on a single GPU: 2 mins for ConvNext and NoiseTunnel Saliency, > 20 hours for Feature Ablation/Permutation)
Compute attribution maps for your model:

```bash
python get_attributions_gpu.py --model_name convnext --method_name NoiseTunnel_Saliency
```

You can replace `convnext` with the best model assessed above and `NoiseTunnel_Saliency` with any Captum method.

To visualize the resulting explanations:
```bash
jupyter notebook visualize_explanations.ipynb
```

Pre-generated attribution maps from `convnext` are provided in the OSF directory.

---

### **4️⃣ Generate EMIs (Explanation-Masked Images)** (5 mins)
Perturb images based on attribution maps:

```bash
python generate_emis.py convnext NoiseTunnel_Saliency
```

As mentioned above, you can replace `convnext` with the best model evaluated earlier and `NoiseTunnel_Saliency` with any Captum method.

---

### **5️⃣ Compute behavioral metrics (B.I1)** (10 mins)
Evaluate model behavior on EMIs:

```bash
python generate_i1s.py convnext NoiseTunnel_Saliency
```
This produces per-image behavioral vectors used to compare models, humans, and monkeys.

As above, you can replace `convnext` with the best model assessed above and `NoiseTunnel_Saliency` with any Captum method.

All behavioral metrics are provided in the OSF directory.

---

### **6️⃣ Compare models, humans, and monkeys** (< 2 mins)
Use the provided notebooks for main analyses:

| Notebook | Description |
|-----------|--------------|
| `compare_explanations.ipynb` | Evaluate accuracy and correlations between models and humans, and compare against baseline explanations (bubbles, object-only). |
| `compare_explanations_with_monkeys.ipynb` | Compare model and monkey behavior under equivalent conditions. |

---

### **(Optional) Identify the best attribution method** (< 2 mins)
Generate Explanations, EMIs and Behavioral metrics for all explanation methods and all models, then run:

```
get_best_method.ipynb
```

This notebook identifies the attribution method that leads to the highest reference-target alignment on average across models.

---

## 📚 Additional Analyses  

Reproduce all results and figures from the paper:

| Notebook | Purpose |
|-----------|----------|
| `compare_similarities.ipynb` | Compare L1, L2, and LPIPS similarity between explanation maps. |
| `get_ground_truth.ipynb` | Compute ground-truth similarity between pairs of models. |
| `compare_ground_truth_proxy.ipynb` | Compare behavioral proxy correlations with ground-truth explanation similarity. |
| `test_models.ipynb` | Compute B.I1 scores for all models on clean images. |
| `compare_neural_predictions.ipynb` | Compare IT neural predictions for the best and worst explanation methods. |

---

## 📦 Data Availability  

All **precomputed model outputs** and **psychophysical data** are available on the  
👉 [**OSF Repository – MAPS Data**](https://osf.io/t6kw8/)

---

## 🔬 Citation  

If you use this code or data, please cite:  

> Muzellec, S., Alghetaa, Y., Kornblith, S., & Kar, K. (2025). *MAPS: Masked Attribution-based Probing of Strategies – A computational framework to align human and model explanations*

---

## 📄 License  

This repository is released under the **MIT License**.  
See `LICENSE` for details.
