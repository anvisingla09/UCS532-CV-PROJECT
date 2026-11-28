# Indian Heritage Image-Text Retrieval

**CS-UCS532 — Computer Vision Open Challenge**

A text-to-image retrieval system for Indian heritage images using CLIP,
TF-IDF, and Hybrid Reciprocal Rank Fusion.

## 1. Project Overview

This project retrieves Indian heritage images using natural-language queries.

Three approaches are compared:
- CLIP ViT-B/32 semantic retrieval
- TF-IDF textual retrieval
- Hybrid Reciprocal Rank Fusion (RRF)

An interactive demo displays the top five retrieved images.

## 2. Dataset

Source: [Bodhisetu Dataset](https://huggingface.co/datasets/51Hypers/Bodhisetu)

A subset of 200 images was selected using random seed 42.

| Domain | Images |
|---|---:|
| Art | 50 |
| Buildings | 50 |
| History | 50 |
| Religion | 50 |
| **Total** | **200** |

## 3. Methodology

**CLIP:** Generates image and text embeddings using ViT-B/32.
Cosine similarity ranks images against text queries.

**TF-IDF:** Uses image names, descriptions, and location metadata
to calculate textual similarity.

**Hybrid:** Combines both rankings using Reciprocal Rank Fusion:

RRF(d) = w/(60 + r_CLIP) + (1-w)/(60 + r_TFIDF)

The selected hybrid weight is 0.50.

## 4. Evaluation

The final evaluation uses 80 queries, with 20 queries per domain.

The expected domain is used as a category-level relevance proxy.

### Final Results

| Method | Hit@1 | Hit@5 | MRR |
|---|---:|---:|---:|
| CLIP | 0.4250 | 0.8875 | 0.6069 |
| Hybrid | 0.5125 | 0.8000 | 0.6225 |
| TF-IDF | 0.4750 | 0.7875 | 0.5823 |

### Weight Ablation

| CLIP Weight | Hit@1 | Hit@5 | MRR |
|---:|---:|---:|---:|
| 0.00 | 0.4750 | 0.7875 | 0.5823 |
| 0.25 | 0.4500 | 0.7750 | 0.5796 |
| 0.50 | 0.5125 | 0.8000 | 0.6225 |
| 0.75 | 0.4375 | 0.7625 | 0.5685 |
| 1.00 | 0.4250 | 0.8875 | 0.6069 |

Hybrid achieves the best Hit@1 and MRR.
CLIP achieves the best Hit@5.

## 5. Failure Analysis

A failure occurs when no image from the expected domain appears
in the top five results.

Buildings account for 32 of the 42 method-query failures,
making it the most challenging category.

## 6. Results and Visualizations

The results/figures directory contains:

- dataset_distribution.png
- method_comparison.png
- weight_ablation.png
- failure_analysis.png

The results/tables directory contains evaluation and analysis CSV files.

## 7. Interactive Demo

The notebook includes an interactive search interface.
Users enter a natural-language query and view the top five retrieved images.

## 8. Repository Structure

- Image_Text_Retrieval.ipynb
- src/load_data.py
- results/figures/
- results/tables/
- configs/
- report/

Dataset images and model embeddings are excluded from GitHub.

## 9. Requirements

- Python 3
- PyTorch
- Transformers
- CLIP
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Pillow
- Hugging Face Datasets
- ipywidgets

The notebook contains setup and execution instructions.

## 10. Limitations

- Experiments use 200 images, not the complete dataset.
- Evaluation uses category-level relevance rather than manually
  annotated semantic relevance.
- Only four heritage domains are evaluated.
- Retrieval quality can vary with query wording and metadata quality.
- Generalization to unseen datasets has not been established.

## 11. Dataset Attribution

Bodhisetu Dataset:
https://huggingface.co/datasets/51Hypers/Bodhisetu

Dataset license: CC BY 4.0.
Please follow the original dataset attribution requirements.

## 12. Course

CS-UCS532 — Computer Vision Open Challenge
