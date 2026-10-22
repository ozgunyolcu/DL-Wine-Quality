# DL-Wine-Quality: Deep Learning Wine Quality Score Prediction

**Course:** RAI2117307 Deep Learning Applications  
**Institution:** Istanbul Medipol University, Vocational School – Robotics and Artificial Intelligence (Fall 2026–2027)  
**Instructor:** Batuhan Hangün  
**Student:** Özgün Yolcu  
**Repository:** [https://github.com/ozgunyolcu/DL-Wine-Quality](https://github.com/ozgunyolcu/DL-Wine-Quality)  

---

## 📌 Project Overview
This project predicts the sensory quality score (0–10) of Portuguese *\"Vinho Verde\"* wines based on 11 physicochemical laboratory measurements (acidity, sugar, alcohol, sulfur dioxide, pH, etc.) using deep neural networks implemented in **PyTorch**.

* **Task:** Tabular Regression
* **Dataset:** [UCI Machine Learning Repository - Wine Quality (ID: 186)](https://archive.ics.uci.edu/dataset/186/wine+quality)
* **Sample Size:** 6,497 samples (Red and White wine variants)
* **Features:** 11 physicochemical inputs
* **Target:** Quality score (scale 0–10)
* **Evaluation Metrics:** Mean Absolute Error (MAE) and Coefficient of Determination ($R^2$)

---

## 🛠️ Project Structure & Pipeline (Week 6 Template)
1. **Data Ingestion:** Fetching through `ucimlrepo` API.
2. **Exploratory Data Analysis (EDA):** Target distribution analysis and correlation heatmap.
3. **Data Preprocessing & Splitting:**
   - Missing value verification.
   - 70% Train, 15% Validation, 15% Test split.
   - `StandardScaler` feature scaling strictly fit on the training partition (zero data leakage).
   - PyTorch `TensorDataset` and `DataLoader` pipeline.
4. **Baseline Model:** Simple 1-hidden-layer feedforward neural network (`BaselineWineNet`).
5. **Improved Model:** Deeper multi-layer architecture with `Dropout`, `BatchNorm1d`, and `LeakyReLU` regularizations (`ImprovedWineNet`).
6. **Training Telemetry:** Comparative loss curves (Train vs. Validation Loss over epochs).
7. **Final Evaluation:** MAE, $R^2$, and Predicted vs. True scatter plots on the held-out test set.
8. **Model Checkpointing:** Serializing weights with `torch.save`.
9. **Findings & Next Steps:** Engineering evaluation of physicochemical predictability on sensory ratings.

---

## 🚀 How to Run
Open the main notebook [`wine_quality_regression.ipynb`](./wine_quality_regression.ipynb) in **VSCode**, **PyCharm** or upload to **Google Colab**.
```bash
pip install -q ucimlrepo torch scikit-learn pandas numpy matplotlib seaborn
```
Execute all cells sequentially from top to bottom.
