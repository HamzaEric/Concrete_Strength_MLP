# Concrete Strength Predictor: From Math Fundamentals to MLP Deployment

A comprehensive machine learning and deep learning project that builds predictive modeling from the ground up. This repository documents the transition from raw mathematical foundations, linear regression, and gradient descent, through non-linear activation functions, culminating in a custom PyTorch Multi-Layer Perceptron (MLP) deployed as an interactive Streamlit web application.

---

## Project Learning Path

The project is structured as an incremental progression of deep learning concepts:

1. **Deep Learning Math Fundamentals (`Notebooks/Deep_Learning_Math_Fundamentals.ipynb`)**
   * Explores matrix operations, dot products, vector transformations, and the foundational math ($XW + b$) that powers neural networks.
2. **Linear Regression & Gradient Descent (`Notebooks/Linear_Regression_&_Gradient_Descent.ipynb`)**
   * Implements optimization from scratch, understanding how models minimize error iteratively using gradient descent.
3. **Non-linearity & Activation Functions (`Notebooks/Non_linearity_&_Activation_Functions.ipynb` & Data Analysis)**
   * Explores why linear models fail on complex engineering problems and introduces activation functions (like ReLU) to model non-linear relationships in concrete data.
4. **Multi-Layer Perceptron (MLP) & Deployment (`app.py`)**
   * Scales up to a custom deep neural network (`DeeperConcreteMLP`) trained on concrete ingredient mixes and packaged into a user-facing web app.

---

## Application Features

* **Custom PyTorch Architecture:** Powered by an 8-32-16-1 feedforward neural network trained to predict concrete compressive strength in Megapascals (MPa).
* **Industry Preset Library:** Includes baseline industry mixes (e.g., Standard Residential, High Strength Infrastructure, Eco-Friendly Fly Ash) accessible via a quick-select dropdown.
* **Interactive Fine-Tuning:** Allows engineers and students to modify individual mix parameters (`kg/m³`) and age to test custom formulations.
* **Mathematical Visualization:** Features an interactive breakdown of the linear layer foundation (`$XW + b$) rendered via LaTeX directly in the UI.

---

## Project Structure

```text
Concrete_Strength_MLP/
│
├── DataFiles/
│   └── Concrete_Data.csv                   # Raw concrete compressive strength dataset
│
├── Models/
│   ├── concrete_mlp_model.pth              # Trained PyTorch weights and biases
│   └── concrete_scaler.pkl                 # Fitted scikit-learn StandardScaler
│
├── Notebooks/
│   ├── Concrete_Data_Analysis.ipynb        # Exploratory data analysis on concrete ingredients
│   ├── Deep_Learning_Math_Fundamentals.ipynb   # Core matrix math and linear algebra concepts
│   ├── Linear_Regression_&_Gradient_Descent.ipynb # Optimization and gradient descent logic
│   └── Non_linearity_&_Activation_Functions.ipynb # Introduction to ReLU and multi-layer behavior
│
├── app.py                                  # Streamlit web application script
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation