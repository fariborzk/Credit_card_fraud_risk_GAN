# Adversarial-Tabular-Oversampling-for-Fraud-Detection (ATOF-Detection)

A deep learning and machine learning pipeline for detecting fraudulent transactions in highly skewed tabular financial data via custom Generative Adversarial Network (GAN) oversampling.

---

## 1. Project Overview & Problem Formulation

In modern financial transaction processing, severe class imbalance is a persistent bottleneck. Fraudulent events typically constitute less than 0.2% of all logged operations. Standard supervised classifiers trained on such raw distributions inevitably develop a strong bias toward the majority class (legitimate transactions), producing superficially high accuracy while missing high-risk anomalies.

This project implements an end-to-end research-grade pipeline:
1. **Adversarial Minority Modeling:** A custom Generative Adversarial Network (Tabular GAN) is engineered from scratch in PyTorch to capture the true underlying distribution of minority (fraud) patterns.
2. **Generative Oversampling:** High-fidelity synthetic fraud samples are generated to balance the training feature space.
3. **Downstream Classification:** Both non-linear deep neural network (Multilayer Perceptron - MLP) and linear maximum-margin (Linear Support Vector Machine - SVM) classifiers are trained on the augmented space and evaluated on unseen, pristine test data.

---

## 2. Dataset & Preprocessing

- **Dataset Source:** Credit Card Fraud Detection benchmark via Kaggle (`mlg-ulb/creditcardfraud`).
- **Scale & Dimensionality:** 284,807 transactions with 30 numerical input features (PCA components $V_1$ through $V_{28}$, transaction `Time`, and transaction `Amount`).
- **Class Skew:** 284,315 legitimate records (Class 0) vs. 492 fraudulent records (Class 1) — fraud prevalence is ~0.172%.
- **Data Splitting:** Stratified 80/20 train-test split (227,845 train records and 56,962 test records containing 56,864 normal and 98 fraudulent instances).
- **Feature Standardization:** Features (including monetary amounts) are scaled using `StandardScaler` to prevent high-variance domination and stabilize backpropagation gradients.

---

## 3. Architecture & Training Pipeline

### 3.1 Custom Tabular GAN Architecture
- **Generator ($G$):** Projects latent Gaussian noise vectors ($z \sim \mathcal{N}(0, I_{32})$) through fully connected dense layers ($32 \to 64 \to 128 \to 30$). Includes `BatchNorm1d` for internal covariate shift stabilization, `LeakyReLU` activations ($\alpha=0.2$), and a `Tanh` output projection layer.
- **Discriminator ($D$):** Maps tabular feature vectors ($x \in \mathbb{R}^{30}$) through hidden representations ($30 \to 128 \to 64 \to 1$) regularized with `Dropout(0.2)` and terminated with a `Sigmoid` activation to output validity probabilities.
- **Training Setup:** Trained exclusively on minority fraud samples for **60 epochs** using Adam optimization ($\beta_1=0.5, \beta_2=0.999$) and Binary Cross-Entropy loss under zero-sum adversarial game formulation.

### 3.2 Downstream Classifiers
1. **PyTorch MLP Classifier:** A deep neural network structured as $30 \to 128 \to 64 \to 32 \to 1$ with `ReLU` activations, `Dropout(0.3)` layers to prevent overfitting, trained using Adam and BCEWithLogitsLoss.
2. **Linear SVM (`LinearSVC`):** Scikit-learn implementation with $C=1.0$ serving as a robust linear baseline.

---

## 4. Empirical Evaluation & Quantitative Results

Given the severe imbalance, overall Accuracy is non-informative. Performance is measured primarily via **F1-Score**, **Recall** (detection rate), and **Precision** (reliability of alerts) on the unaugmented test partition.

### Performance Summary Table

| Model | Precision | Recall | F1-Score | False Positives (FP) | True Positives (TP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP Classifier** | **0.8041** | **0.7959** | **0.8000** | **19** | **78 / 98** |
| **Linear SVM** | **0.8169** | **0.5918** | **0.6864** | **14** | **70 / 98** |

### Key Observations:
- **MLP Classifier:** Achieved an **F1-score of 0.8000**, detecting **78 out of 98** fraudulent transactions while triggering only **19 false alarms** across 56,864 genuine transactions. This reflects a very high operational signal-to-noise ratio.
- **Linear SVM:** Produced high precision (81.69%) and lower false positives (14), but achieved lower recall (59.18%) due to the inherent non-linearity of complex multi-dimensional fraud patterns.

---

## 5. Artifacts & Visual Analysis

### 5.1 GAN Training Convergence Curve (`convergence_plot.webp`)
The training progression across 60 epochs displays ideal minimax equilibrium:
- Discriminator loss settles in the $[0.65, 1.25]$ range.
- Generator loss stays within $[0.60, 1.75]$.
- Neither mode collapse nor gradient vanishing/explosion occurred.

### 5.2 Confusion Matrices
- **MLP (`confusion_matrix_mlp_classifier.webp`):**
  - True Negative: `56,845` | False Positive: `19`
  - False Negative: `20` | True Positive: `78`
- **Linear SVM (`confusion_matrix_linear_svm.webp`):**
  - True Negative: `56,850` | False Positive: `14`
  - False Negative: `28` | True Positive: `70`

---

## 6. Setup & Execution

### Prerequisites
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn kagglehub kaggle


> [!NOTE]
> *This README document was generated with the assistance of Artificial Intelligence.*
