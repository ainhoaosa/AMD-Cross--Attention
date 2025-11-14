# AMD-Cross-Attention

This repository contains the implementation of a **cascaded multimodal system** for diagnosing **Age-Related Macular Degeneration (AMD)** using **Cross-Attention**.  
It combines **fundus images** encoded via a **Vision Transformer (ViT)** with **clinical metadata** processed through a multilayer neural network (MLP).  
Explainable AI methods such as **LIME** and **GradCAM** are included to highlight important image regions and clinical factors influencing predictions.

---

## Abstract

This study introduces a new cascaded method for diagnosing AMD that uses cross-attention.  
Through a cross-attention module enabling dynamic multimodal fusion, the architecture combines fundus images encoded via a Vision Transformer (ViT) with clinical metadata processed through an MLP.  
Unlike approaches relying solely on image analysis, this method merges visual and clinical information to improve discrimination across AMD stages: Normal, Intermediate, Geographic Atrophy, and Wet.  

The three-stage cascaded CAD system achieved F1 scores above 93% and AUC values close to 0.99 for individual classifiers.  
The final cascade evaluation reached **96.15% accuracy** and **96.18% F1-score**.  
---

## Model Architecture

- **Vision Transformer Encoder**: Pretrained, extracts high-dimensional visual embeddings.  
- **Metadata MLP**: Processes clinical metadata (age, sex, treatment, dosing regimen, clinical center, etc.).  
- **Cross-Attention Fusion**:  
  - Query = metadata  
  - Key/Value = image embedding  
  - Produces joint multimodal representation  

**3-Stage Cascaded CAD System:**
1. Normal vs AMD  
2. Intermediate vs Advanced AMD  
3. Geographic Atrophy vs Wet AMD  

---

## Performance

| Stage | Task | F1-Score | AUC |
|-------|------|----------|-----|
| 1 | Normal vs AMD | >93% | ~0.99 |
| 2 | Intermediate vs Advanced | >93% | ~0.99 |
| 3 | GA vs Wet | >93% | ~0.99 |
| **Final Cascade** | 4-class AMD | **96.18%** | — |

---

## Synthetic Metadata Generator

Automatically generates synthetic metadata for images organized in folders by class.

**Features:**
- Detects: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`  
- Generates:
  - Patient ID  
  - Age (50–90)  
  - Sex (M/F)  
  - Anti-VEGF treatment  
  - Dosing regimen  
  - Clinical center (1–43)  
- Output CSV: `synthetic_metadata_A_and_B.csv`

---
# License

MIT License — free to use, modify, and distribute.
---
For research questions, collaborations, or updates related to multimodal AMD diagnosis, feel free to reach out.
