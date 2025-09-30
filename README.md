# Safe Drinking Water Prediction Project

## 1. Project Overview

This is an end-to-end Machine Learning project aimed at classifying the potability of water samples based on nine physicochemical parameters. The goal was to develop a robust classification model and deploy it as an accessible web application.

The live application is deployed on Streamlit Cloud and is accessible via the link below.

| Component | Status | Link |
|-----------|--------|------|
| **Live Web App** | Deployed | [https://safedrinkingwater.streamlit.app/](https://safedrinkingwater.streamlit.app/) |
| **GitHub Repo** | Complete | [https://github.com/krnike96/safe_drinking_water.git](https://github.com/krnike96/safe_drinking_water.git) |

## 2. Dataset and Features

The model was trained on the Water Potability dataset, which includes the following features (input parameters):

| Feature | Unit/Context | Expected Range (Raw Input) |
|---------|--------------|----------------------------|
| **ph** | Acidity/Basicity | 0-14 (typically 6.5-8.5) |
| **Hardness** | mg/L (Calcium/Magnesium) | 0-500+ mg/L |
| **Solids** | ppm (Total Dissolved Solids) | 0-50,000 ppm |
| **Chloramines** | ppm (Disinfectant) | 0-10 ppm |
| **Sulfate** | mg/L | 0-500 mg/L |
| **Conductivity** | μS/cm | 0-1000 μS/cm |
| **Organic_carbon** | ppm | 0-20 ppm |
| **Trihalomethanes** | μg/L | 0-120 μg/L |
| **Turbidity** | NTU (Nephelometric Turbidity Units) | 0-10 NTU |
| **Potability (Target)** | Binary (1: Potable, 0: Not Potable) | - |

## 3. Methodology and Model Performance

The project followed a rigorous ML pipeline to handle data quality issues and optimize the final classifier.

### Pipeline Steps:

1. **Data Cleaning:** Missing values in ph, Sulfate, and Trihalomethanes were imputed using the column mean.

2. **Feature Scaling:** All features were scaled using StandardScaler to normalize the data distribution.

3. **Handling Imbalance:** Due to a significant class imbalance in the target variable, **SMOTE** (Synthetic Minority Over-sampling Technique) was applied to the training set to prevent the model from ignoring the minority ("Potable") class.

4. **Model Selection & Tuning:** A **Random Forest Classifier** was selected for its robustness. GridSearchCV was used to optimize hyperparameters (n_estimators=200, max_depth=20) on the resampled training data.

### Final Model Metrics

The model was evaluated on a held-out test set (unseen data). The metrics below highlight the model's ability to balance identification of both safe (Potable) and unsafe (Not Potable) water.

| Metric | Class 0 (Not Potable) | Class 1 (Potable) | Overall |
|--------|----------------------|-------------------|---------|
| **Precision** | 0.73 | 0.44 | - |
| **Recall** | 0.81 | 0.35 | - |
| **F1-Score** | 0.77 | 0.39 | - |
| **Accuracy** | - | - | **0.67** |

**Interpretation:** The model is strong at identifying unsafe water (high Recall of 0.81 for Class 0), but its precision on predicting *safe* water is low (0.44), indicating a tendency to be cautious and flag potentially safe water as unsafe (high False Negative rate). For a safety-critical application, high recall on the "Not Potable" class is arguably the most important factor.

## 4. Repository Structure

```
├── .gitignore
├── Model/
│   ├── best_rf_model.pkl      # The final, tuned Random Forest model
│   └── scaler.pkl              # The saved StandardScaler for deployment preprocessing
├── water_potability.csv        # Original dataset file
├── README.md                   # This document
├── app.py                      # Streamlit web application code (Frontend)
├── main.py                     # Full ML pipeline script (Data cleaning, SMOTE, Training)
└── requirements.txt            # Minimal list of dependency versions for deployment
```

## 5. Local Setup and Installation

To reproduce the model training and run the application locally, follow these steps:

### 1. Clone the repository:
```bash
git clone https://github.com/krnike96/safe_drinking_water.git
cd safe_drinking_water
```

### 2. Create and activate a virtual environment:
```bash
# Create environment (Windows/Linux/Mac)
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### 3. Install dependencies:
Use the pinned versions from the requirements.txt file to ensure compatibility:
```bash
pip install -r requirements.txt
```

### 4. Run the ML pipeline:
Execute main.py to regenerate the model artifacts (best_rf_model.pkl and scaler.pkl):
```bash
python main.py
```

### 5. Run the Streamlit app locally:
```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## Technologies Used

- **Python 3.x**
- **scikit-learn** - Machine Learning & Model Training
- **imbalanced-learn** - SMOTE Implementation
- **pandas** - Data Manipulation
- **Streamlit** - Web Application Framework
- **pickle** - Model Serialization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available for educational purposes.