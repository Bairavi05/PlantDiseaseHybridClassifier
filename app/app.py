from flask import Flask, request, render_template, jsonify
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import VGG19_Weights, ResNet18_Weights
from PIL import Image
import numpy as np
import warnings


warnings.filterwarnings("ignore")


app = Flask(__name__)

class HybridVGGResNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridVGGResNet, self).__init__()

        # Load pretrained VGG19 and ResNet
        self.vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove the final classification layers
        self.vgg19.classifier = nn.Sequential(*list(self.vgg19.classifier.children())[:-1])
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Freeze pretrained layers (optional)
        for param in self.vgg19.parameters():
            param.requires_grad = False
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Combine VGG19 and ResNet feature dimensions
        vgg19_feature_dim = 4096  # From VGG19's classifier output
        resnet_feature_dim = 512  # From ResNet18's avgpool output

        # Final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(vgg19_feature_dim + resnet_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features from both VGG19 and ResNet
        vgg_features = self.vgg19(x)
        resnet_features = torch.flatten(self.resnet(x), 1)

        # Concatenate features from both models
        combined_features = torch.cat((vgg_features, resnet_features), dim=1)

        # Pass through the final classifier
        out = self.fc(combined_features)
        return out
    

model = HybridVGGResNet(num_classes=3)  # Adjust num_classes if needed
model.load_state_dict(torch.load('model/hybrid_vgg_resnet_gpu.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def home():
    return render_template('index.html')  # Create a simple HTML form for image upload

@app.route('/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded!', 400

    file = request.files['file']
    if file.filename == '':
        return 'No file selected!', 400

    try:
        # Process the image
        img = Image.open(file.stream)
        img = transform(img).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            outputs = model(img)
            probabilities = F.softmax(outputs, dim=1)  # Get probabilities
            top_prob, top_class = torch.max(probabilities, 1)  # Get the top prediction
            top_probs, top_classes = torch.topk(probabilities, 3)  # Get top-3 predictions

        # Map index back to class name
        classes = ['Disease Free leaves', 'Leaf Rust', 'Leaf Spot']

        # Create a detailed response with all predictions and their probabilities
        all_predictions = []
        for i in range(top_classes.size(1)):
            class_index = top_classes[0][i].item()
            class_name = classes[class_index]
            confidence = top_probs[0][i].item() * 100  # Convert to percentage
            all_predictions.append({
                'class_name': class_name,
                'confidence': f'{confidence:.2f}%'
            })

        # Get the top prediction for the frontend
        predicted_class = classes[top_class.item()]
        confidence = top_prob.item() * 100  # Convert to percentage


        print('the output is:')
        print(all_predictions)
        # Return JSON response with all predictions and the top prediction
        return jsonify({
            'predictions': all_predictions,
            'top_prediction': predicted_class,
            'top_confidence': f'{confidence:.2f}%'
        })

    except Exception as e:
        return str(e), 500



if __name__ == '__main__':
    app.run(debug=True)