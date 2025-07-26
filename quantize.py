import joblib
import numpy as np
import torch

# Load sklearn model params
model = joblib.load('model.joblib')
coef = model.coef_
intercept = model.intercept_

# Save unquantized parameters
unquant_params = {'coef': coef, 'intercept': intercept}
joblib.dump(unquant_params, 'unquant_params.joblib')

# Quantization (to uint8)
def quantize(x):
    min_w, max_w = x.min(), x.max()
    scale = (max_w - min_w) / 255 if max_w != min_w else 1
    zero_point = np.round(-min_w / scale)
    q = np.clip(np.round(x / scale + zero_point), 0, 255).astype(np.uint8)
    return q, scale, zero_point

# Dequantization
def dequantize(q, scale, zero_point):
    return scale * (q.astype(np.float32) - zero_point)

q_coef, coef_scale, coef_zero = quantize(coef)
q_intercept, int_scale, int_zero = quantize(np.array([intercept]))

quant_params = {
    'q_coef': q_coef, 'coef_scale': coef_scale, 'coef_zero': coef_zero,
    'q_intercept': q_intercept, 'int_scale': int_scale, 'int_zero': int_zero
}
joblib.dump(quant_params, 'quant_params.joblib')

# Create and save PyTorch model with quantized params
import torch.nn as nn

class QuantizedLinear(nn.Module):
    def __init__(self, input_dim, q_coef, coef_scale, coef_zero, q_intercept, int_scale, int_zero):
        super().__init__()
        self.q_coef = torch.tensor(q_coef, dtype=torch.uint8)
        self.coef_scale = coef_scale
        self.coef_zero = coef_zero
        self.q_intercept = torch.tensor(q_intercept, dtype=torch.uint8)
        self.int_scale = int_scale
        self.int_zero = int_zero

    def forward(self, x):
        weights = self.coef_scale * (self.q_coef.float() - self.coef_zero)
        bias = self.int_scale * (self.q_intercept.float() - self.int_zero)
        return x @ weights + bias

model_pt = QuantizedLinear(
    input_dim=len(coef),
    q_coef=q_coef, coef_scale=coef_scale, coef_zero=coef_zero,
    q_intercept=q_intercept, int_scale=int_scale, int_zero=int_zero
)
torch.save(model_pt.state_dict(), 'quantized_model.pth')

# Testing model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

model_pt.eval()
with torch.no_grad():
    y_pred = model_pt(X_test_tensor).numpy()

original_model = joblib.load('model.joblib')
orig_pred = original_model.predict(X_test)

print("Original Sklearn model R²:", r2_score(y_test, orig_pred))
print("Quantized PyTorch model R²:", r2_score(y_test, y_pred))

