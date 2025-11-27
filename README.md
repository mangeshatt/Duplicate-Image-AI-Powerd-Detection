🤖 AI-Duplicate-Detector: Multi-Stage Image Similarity Engine



## ✨ Project Overview

**AI-Duplicate-Detector** is a sophisticated, multi-stage computer vision library designed to identify duplicate or highly similar images, even after aggressive editing, resizing, or lossy compression—a common challenge with AI-generated content.

Unlike simple hash checkers, this tool employs a **Triage Pipeline** to balance speed and accuracy:

1.  **Stage 1 (Fast):** Perceptual Hashing (pHash) for near-exact matches.
2.  **Stage 2 (Smart):** Deep Learning Embeddings (ResNet/ViT) for conceptual and semantic similarity.
3.  **Stage 3 (High-Fidelity):** SIFT/ORB Feature Matching for pixel-level structural verification on borderline cases.

## 🚀 Installation

### Prerequisites
* Python 3.8+
* NVIDIA GPU (Recommended for faster embedding generation)

```bash
# Clone the repository
git clone [https://github.com/yourusername/ai-image-detector.git](https://github.com/yourusername/ai-image-detector.git)
cd ai-image-detector

# Install required packages
pip install -r requirements.txt
