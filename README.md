# Numerical Rating Prediction from Text: Comparative Performance of DL and ML Algorithms on Game Reviews

This repository contains the source code, data preprocessing pipeline, and project report for the sentiment analysis project on the Metacritic Game Reviews dataset. The goal of this project is to predict numerical game review scores (ranging from 0 to 100) from user reviews using various Machine Learning and Deep Learning architectures.

## Authors
- **Mahmut Özkızılcık** (2220765019)
- **Mukhamediyar Amanzhol** (2210765052)
- *Hacettepe University, Artificial Intelligence Engineering*

## Project Overview
Sentiment analysis is widely studied as a binary classification problem (positive vs. negative). In this project, we treat it as a **regression task**, predicting numerical ranges reflecting sentiment depth from highly emotive text such as game reviews. 

The dataset consists of **1.6 million reviews**. The reviews were filtered for the English language and thoroughly cleaned. We evaluated the following architectures:
1. **Ridge Regression** (Baseline with TF-IDF features)
2. **XGBoost** (SVD-reduced features)
3. **Bi-LSTM with Attention** (Recurrent Neural Network on PyTorch)
4. **DistilBERT** (Transformer model fine-tuned on A100 GPU)

## Key Challenges Solved
- **Target Scaling**: Models like BERT initially failed to converge when trained on the 0-100 scale. Normalizing the targets to a **0 to 1 range** before training stabilized learning and significantly improved regression performance.
- **Hardware Efficiency**: Full BERT was too computationally expensive to train effectively. We transitioned to **DistilBERT**, achieving comparable performance with feasible fine-tuning speeds, and leveraged a Google Colab A100 instance instead of local RTX GPUs.

## Results

We evaluated each model based on the Mean Squared Error (MSE, calculated on scaled values) and the $R^2$ Score.

| Model | MSE (Scaled) | $R^2$ Score |
| :--- | :--- | :--- |
| **Ridge Regression (Baseline)** | 0.0355 | 0.59 |
| **XGBoost** | 0.0312 | 0.64 |
| **Bi-LSTM + Attention** | 0.0245 | 0.72 |
| **DistilBERT** | **0.0182** | **0.78** |

DistilBERT significantly outperformed traditional models, successfully capturing nuanced user sentiment. Using an attention mechanism in the Bi-LSTM also proved essential for interpretable and accurate sequence modeling.

## Repository Contents
- `sentiment_analysis.ipynb`: The primary Jupyter Notebook containing data preprocessing, model definitions (Bi-LSTM, Ridge, XGBoost), training loops, and evaluation metrics.
- `sentiment_analyses_regression_report.pdf`: The detailed academic report explaining methodologies, literature review, and attention visualizations.

## Requirements
You can install the dependencies via `pip` based on the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```
Key libraries include `torch`, `transformers`, `xgboost`, `scikit-learn`, `pandas`, `shap`, and `langdetect`.

## Future Work
- DistilBERT training on the entire 1.6M dataset for 20+ epochs.
- Hyperparameter tuning using automated tools like Optuna.
- Developing a real-time web application to predict rating scores dynamically from custom review texts.
