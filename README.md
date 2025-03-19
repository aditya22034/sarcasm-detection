🎭 Sarcasm Detection using Multimodal Speech Processing
📌 Overview
This project aims to detect sarcasm in speech using deep learning and multimodal feature extraction. It leverages three powerful pre-trained models:

Facebook Wav2Vec2 XLS-R-1B – Extracts deep speech representations.
Microsoft UniSpeech-SAT-Base-100h-Libri-Fit – Enhances speech embeddings with speaker-aware tuning.
ImageBind – Enables multimodal integration.
We process audio embeddings from context and utterance separately, apply noise reduction techniques (PCA), feature selection (SelectKBest), and deep learning with fully connected networks.

🚀 Features
✔️ Uses pretrained speech models to extract robust audio embeddings.
✔️ Applies PCA to remove noise from embeddings.
✔️ Implements SelectKBest for feature selection.
✔️ Adds delta features to capture sarcasm variations.
✔️ Uses a deep neural network with dropout and batch normalization for better generalization.
✔️ Optimized with AdamW optimizer instead of Adam for better convergence.
✔️ Model checkpointing ensures the best epoch is used for final testing.

📂 Dataset
Input: Audio speech samples labeled as sarcastic or non-sarcastic.
Features extracted using Wav2Vec2 XLS-R-1B, UniSpeech-SAT, and ImageBind.
Processed features stored as .csv files.

🏗 Model Architecture
The model consists of two branches:

Context Branch – Processes previous context speech features.
Utterance Branch – Processes the current utterance features.
Fusion Layer – Combines both branches and classifies sarcasm.
