import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

st.set_page_config(layout="wide", page_title="Concrete MLP")


# --- 1. DEFINE THE EXACT ARCHITECTURE ---
class DeeperConcreteMLP(nn.Module):
    def __init__(self):
        super(DeeperConcreteMLP, self).__init__()
        self.fc1 = nn.Linear(8, 32)  # First hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)  # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 1)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# --- 2. LOAD PYTORCH MODEL & SCALER ---
@st.cache_resource
def load_components():
    # Instantiate your custom class
    model = DeeperConcreteMLP()

    # Load the trained weights from the Models folder
    model.load_state_dict(torch.load('Models/concrete_mlp_model.pth', weights_only=True))
    model.eval()

    # Load the standardization scaler
    scaler = joblib.load('Models/concrete_scaler.pkl')

    return model, scaler


model, scaler = load_components()

# --- 3. DEFINE PRESET RECIPES ---
PRESET_MIXES = {
    "Standard Residential (approx. 20-25 MPa)":
        [250.0, 0.0, 0.0, 180.0, 0.0, 1050.0, 800.0, 28],
    "High Strength Infrastructure (approx. 45-50 MPa)":
        [400.0, 20.0, 0.0, 150.0, 10.0, 1000.0, 750.0, 28],
    "Eco-Friendly Fly Ash Mix (approx. 30 MPa)":
        [200.0, 0.0, 150.0, 160.0, 5.0, 1020.0, 780.0, 28]
}

# --- 4. BUILD THE UI ---
st.title("Concrete Strength Multi-Layer Perceptron")

st.subheader("Mathematical Flow of a Linear Layer (The Foundation)")
st.markdown("---")

# Create 4 equal-width columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("1. Input Matrix ($X$)")
    st.markdown("**Shape: (1000, 8)**")
    st.markdown("Every row is a specific concrete mixture, and every column is one of the 8 features.")
    st.latex(r"""
    X = \begin{bmatrix} 
    x_{1,1} & \dots & x_{1,8} \\ 
    x_{2,1} & \dots & x_{2,8} \\ 
    \vdots & \ddots & \vdots \\ 
    x_{1000,1} & \dots & x_{1000,8} 
    \end{bmatrix}
    """)

with col2:
    st.subheader("2. Weights Vector ($W$)")
    st.markdown("**Shape: (8, 1)**")
    st.markdown("The multipliers the model learns. One weight for each of the 8 columns.")
    st.latex(r"""
    W = \begin{bmatrix} 
    \omega_1 \\ 
    \omega_2 \\ 
    \vdots \\ 
    \omega_8 
    \end{bmatrix}
    """)

with col3:
    st.subheader("3. Bias ($b$)")
    st.markdown("**Shape: (1000, 1)**")
    st.markdown("A scalar number automatically broadcasted into a column to add to every row.")
    st.latex(r"""
    b = \begin{bmatrix} 
    \beta \\ 
    \beta \\ 
    \vdots \\ 
    \beta 
    \end{bmatrix}
    """)

with col4:
    st.subheader("4. Output Vector ($\hat{y}$)")
    st.markdown("**Shape: (1000, 1)**")
    st.markdown("The result of $XW + b$. A column vector representing the predicted strength.")
    st.latex(r"""
    \hat{y} = \begin{bmatrix} 
    (\omega_1 x_{1,1} + \dots) + \beta \\ 
    (\omega_1 x_{2,1} + \dots) + \beta \\ 
    \vdots \\ 
    (\omega_1 x_{1000,1} + \dots) + \beta 
    \end{bmatrix} 
    """)

st.markdown("---")

# Dropdown that drives the default values of the input boxes
selected_profile = st.selectbox("Start from a Baseline Mixture:", options=list(PRESET_MIXES.keys()))
defaults = PRESET_MIXES[selected_profile]

st.markdown("### Fine-Tune Your Recipe (kg/m³)")
col1, col2 = st.columns(2)

with col1:
    cement = st.number_input("Cement", min_value=0.0, value=float(defaults[0]))
    slag = st.number_input("Blast Furnace Slag", min_value=0.0, value=float(defaults[1]))
    fly_ash = st.number_input("Fly Ash", min_value=0.0, value=float(defaults[2]))
    water = st.number_input("Water", min_value=0.0, value=float(defaults[3]))

with col2:
    superplasticizer = st.number_input("Superplasticizer", min_value=0.0, value=float(defaults[4]))
    coarse_agg = st.number_input("Coarse Aggregate", min_value=0.0, value=float(defaults[5]))
    fine_agg = st.number_input("Fine Aggregate", min_value=0.0, value=float(defaults[6]))
    age = st.number_input("Age (Days)", min_value=1, value=int(defaults[7]))

# --- 5. PYTORCH PREDICTION LOGIC ---
if st.button("Predict Strength", type="primary"):
    # 1. Gather inputs into a raw numpy array
    raw_features = np.array([[
        cement, slag, fly_ash, water,
        superplasticizer, coarse_agg, fine_agg, age
    ]])

    # 2. Scale features
    scaled_features = scaler.transform(raw_features)

    # 3. Convert to PyTorch Tensor
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    # 4. Forward Pass through PyTorch Model
    with torch.no_grad():
        prediction = model(input_tensor)

    # 5. Extract and clamp final value
    final_strength = max(0.0, prediction.item())

    st.success(f"### Predicted Compressive Strength: {final_strength:.2f} MPa")