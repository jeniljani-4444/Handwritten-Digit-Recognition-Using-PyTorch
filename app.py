import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time

title = "Handwritten Digit Recognition App ✍️"
description = "A handwriting recognition system by PyTorch"

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def predict(image):
    start_time = time.time()
    image = image.convert('L')  
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        class_names = ['0 - zero','1 - one','2 - two','3 - three','4 - four','5 - five','6 - six','7 - seven','8- eight','9 - nine']
        model = torch.load("main_model")
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
        end_time = time.time()
        prediction_time = end_time - start_time
        pred_probs = torch.softmax(model(image_tensor), dim=1)
        pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }
        return  pred_labels_and_probs,prediction,prediction_time 


demo = gr.Interface(fn=predict, 
                    inputs=gr.Image(type="pil"), 
                    outputs=[gr.Label(num_top_classes=10, label="Predictions"), 
                             gr.Number(label="Prediction time (s)")], 
                    title=title,
                    description=description,
                    theme=gr.themes.Soft(),
                   
                  )

demo.launch()
