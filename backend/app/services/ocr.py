import torch
import torch.nn as nn
import numpy as np
import cv2
import os

# Define the same architecture as in train_cnn.py
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Path to the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.pth")

# Global variable to hold the loaded model
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = DigitCNN().to(_device)
            _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
            _model.eval()
        else:
            print(f"Warning: Model not found at {MODEL_PATH}. OCR will fail.")
            print("Please run 'python backend/app/services/train_cnn.py' to generate the model.")
    return _model

def recognize_digit(cell_img):
    """
    Recognizes a single digit using a PyTorch CNN model.
    """
    model = get_model()
    if model is None:
        return 0

    try:
        # 1. Resize to 28x28
        if cell_img.shape != (28, 28):
            cell_img = cv2.resize(cell_img, (28, 28))

        # 2. Normalize and convert to tensor
        img_array = cell_img.astype("float32") / 255.0
        # MNIST normalization: (mean=0.1307, std=0.3081)
        img_array = (img_array - 0.1307) / 0.3081
        
        input_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0).to(_device)

        # 3. Predict
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        digit = predicted.item()
        
        # Sudoku only has 1-9. If CNN predicts 0, it's likely noise or an empty cell.
        if digit == 0 or confidence.item() < 0.95:
            return 0
            
        return int(digit)
    except Exception as e:
        print(f"PyTorch OCR Error: {e}")
        return 0

def recognize_digits_parallel(cells):
    """
    Recognizes digits from multiple cells using batch inference.
    """
    if not cells:
        return [[0]*9 for _ in range(9)]
    
    results = [0] * 81
    model = get_model()
    
    if model is None:
        return [[0]*9 for _ in range(9)]

    images = []
    cell_indices = []
    
    for r, c, img in cells:
        # Resize to 28x28
        img_28 = cv2.resize(img, (28, 28))
        # Normalize
        img_norm = (img_28.astype("float32") / 255.0 - 0.1307) / 0.3081
        images.append(img_norm)
        cell_indices.append((r, c))
    
    if images:
        batch_tensor = torch.FloatTensor(np.array(images)).unsqueeze(1).to(_device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicteds = torch.max(probabilities, 1)
        
        for i, (r, c) in enumerate(cell_indices):
            digit = predicteds[i].item()
            conf = confidences[i].item()
            if digit != 0 and conf > 0.95:
                print(f"CNN: Cell {r},{c} predicted {digit} with confidence {conf:.4f}")
                results[r * 9 + c] = int(digit)
            elif digit != 0:
                # print(f"CNN: Cell {r},{c} predicted {digit} but REJECTED (conf {conf:.4f})")
                pass

    # Reshape back to 9x9 board
    board = []
    for i in range(0, 81, 9):
        board.append(results[i:i+9])
        
    return board
