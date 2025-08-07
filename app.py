import numpy as np
import re
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import base64
from transformers import RobertaTokenizer, RobertaModel

# ---------------------- Roberta  ----------------------
class TextClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(TextClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)

# ---------------------- CNN-LSTM CLASSIFIER ----------------------
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNLSTM, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        features = self.features(x)
        features = features.permute(0, 2, 3, 1).reshape(batch_size, -1, 512)
        lstm_out, _ = self.lstm(features)
        x = self.dropout(lstm_out[:, -1, :])
        return self.fc(x)

# ---------------------- DETECTOR CLASS ----------------------
class SocialMediaMentalHealthDetector:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.class_names = ['Depression', 'Anxiety', 'Normal', 'Other']
        
        # Text
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.text_model = TextClassifier(num_classes=4).to(self.device).eval()
        
        # Image
        self.image_model = CNNLSTM(num_classes=4).to(self.device).eval()
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Fallback keywords
        self.depression_keywords = ['sad', 'depressed', 'hopeless', 'lonely', 'worthless']
        self.anxiety_keywords = ['anxious', 'panic', 'stress', 'worried', 'fear']
        self.normal_keywords = ['happy', 'joy', 'love', 'grateful', 'peaceful']
        
        self.image_patterns = {
            'Depression': ['dark', 'grey', 'black'],
            'Anxiety': ['chaotic', 'blur', 'red'],
            'Normal': ['bright', 'sun', 'smile'],
            'Other': ['neutral', 'object', 'abstract']
        }
    
    def analyze_text(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=512,
            padding="max_length", truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()[0]
    
    def analyze_image(self, image_data):
        try:
            if image_data.startswith("data:image"):
                # Remove base64 prefix and decode
                image_data = image_data.split(",")[1]
                image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
            else:
                image = Image.open(image_data).convert('RGB')
        except Exception as e:
            print(f"Image decoding failed: {e}")
            return self.analyze_image_with_patterns(image_data)
        
        img_tensor = self.image_transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.image_model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()[0]
    
    def analyze_image_with_patterns(self, image_path):
        image_path = str(image_path).lower()
        scores = [0.1, 0.1, 0.1, 0.1]
        for i, cat in enumerate(self.class_names):
            for kw in self.image_patterns[cat]:
                if kw in image_path:
                    scores[i] += 0.3
        total = sum(scores)
        return [s/total for s in scores]
    
    def combine_predictions(self, text_probs, image_probs, text_weight=0.6, image_weight=0.4):
        combined = text_weight * np.array(text_probs) + image_weight * np.array(image_probs)
        return combined / np.sum(combined)
    
    def classify(self, text, image_input):
        text_probs = self.analyze_text(text)
        image_probs = self.analyze_image(image_input)
        combined_probs = self.combine_predictions(text_probs, image_probs)
        predicted_label = self.class_names[np.argmax(combined_probs)]
        return predicted_label, combined_probs

# ---------------------- DEMO FUNCTION ----------------------
def get_user_input():
    print("\n===== Enter Social Media Content for Mental Health Analysis =====")
    text_input = input("Enter the text of the post: ")
    image_input = input("Enter the path to the image or paste base64 string: ")
    return text_input, image_input

def analyze_post(post_text, post_image):
    detector = SocialMediaMentalHealthDetector()
    predicted_class, probabilities = detector.classify(post_text, post_image)
    return predicted_class, probabilities

def run_demo():
    text_input, image_input = get_user_input()
    print("\n----- Analyzing Your Post -----")
    predicted_class, probabilities = analyze_post(text_input, image_input)
    print(f"Prediction: {predicted_class}")
    print("Probability breakdown:")
    for label, prob in zip(['Depression', 'Anxiety', 'Normal', 'Other'], probabilities):
        print(f"  - {label}: {prob:.2f}")
    print("\n")

# ---------------------- RUN MAIN ----------------------
if __name__ == "__main__":
    run_demo()
