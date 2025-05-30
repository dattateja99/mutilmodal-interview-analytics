# 🎤 Behavioral and Stress Evaluation of Interview Performance

![multimodal](https://img.shields.io/badge/Machine_Learning-Multimodal-informational) ![Status](https://img.shields.io/badge/status-Completed-brightgreen)

This repository contains the final project for USC's DSCI course, where we developed a **multimodal AI system** to automatically evaluate behavioral traits and stress levels during interviews by analyzing **audio**, **video**, and **textual** data.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Datasets](#-datasets)
- [Methodology](#-methodology)
- [Modeling & Results](#-modeling--results)
- [Setup & Installation](#-setup--installation)
- [Team](#-team)
- [References](#-references)

---

## 📖 Overview

Traditional interview assessments often rely on subjective judgments, leading to bias and inconsistency. Our project proposes an **objective, data-driven approach** by analyzing:

- Facial expressions and head movements (Video)
- Speech prosody and pauses (Audio)
- Verbal content and emotion scores (Text)

We use **early fusion** of multimodal features to predict key behavioral metrics such as *Engagement*, *Eye Contact*, *Calmness*, and *Stress Score*.

---

## 📊 Datasets

### 🧑‍💼 MIT Interview Dataset
- 138 mock interview videos from 69 students
- Behavioral ground truth scores via Amazon Mechanical Turk
- Labels: Engagement, Eye Contact, Focus, etc.

### 📹 MAS (Multi-Affect-Stress) Dataset
- 353 annotated interview clips from YouTube
- Labels for vocal/facial stress, fidgeting, etc.
- Normalized stress scores (0–7 scale)

---

## 🧠 Methodology

We use an **early fusion pipeline** that combines:

- **Audio Features**: pitch, jitter, energy, pauses, speaking rate  
- **Video Features**: head pose (yaw, pitch, roll), facial distances (e.g., lip, eye)  
- **Text Features**: word count, filler words, emotion (via BERT), speech rate  

### Steps:

1. Extract features using `Librosa`, `Mediapipe`, `Google Speech-to-Text`, `NLTK`, and `BERT`
2. Normalize and select top features via F-regression and correlation filtering
3. Train models (Random Forest, Gradient Boosting, LightGBM, Neural Nets)
4. Evaluate predictions using **Mean Squared Error (MSE)**

---

## 📈 Modeling & Results

### MIT Interview Dataset

| Label            | Random Forest | Gradient Boosting | Ridge     |
|------------------|----------------|-------------------|-----------|
| Eye Contact      | 0.4874         | **0.3685**         | 1.1790    |
| Speaking Rate    | 0.1785         | **0.1544**         | 0.3302    |
| Engaged          | **0.2695**     | 0.3179             | 0.6309    |
| Calm             | 0.2773         | **0.2727**         | 0.3298    |
| Not Stressed     | **0.2351**     | 0.2381             | 0.5421    |
| Average (All)    | 0.2925         | **0.2549**         | 0.5139    |

### MAS Dataset (Stress Score)

| Model                  | MSE     |
|------------------------|---------|
| LightGBM               | 0.4681  |
| Neural Network         | 0.6698  |
| Gradient Boosting (3f) | **0.0161**  |

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/interview-eval-ai.git
cd interview-eval-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
