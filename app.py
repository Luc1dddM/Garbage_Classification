import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

class_to_idx = {'battery': 0,
                'biological': 1,
                'cardboard': 2,
                'clothes': 3,
                'glass': 4,
                'metal': 5,
                'paper': 6,
                'plastic': 7,
                'shoes': 8,
                'trash': 9}

# Create reverse mapping from index to class name
idx_to_class = {v: k for k, v in class_to_idx.items()}

class GarbageClassifier(torch.nn.Module):
    def __init__(self, num_classes):
      super().__init__()
      self.model = models.resnet50(pretrained=True)
      in_features = self.model.fc.in_features
      self.model.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x, y=None):
      out = self.model(x)
      return out

@st.cache_resource
def load_model(model_path, num_classes=10):
    garbage_model = GarbageClassifier(num_classes)
    garbage_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    garbage_model.eval()
    return garbage_model
model = load_model('classifier_model.pt')

def inference(image, model):
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)
        wnew, hnew = image.size
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_new = transform(image)
    img_new = img_new.expand(1, 3, 224, 224)
    with torch.no_grad():
        predictions = model(img_new)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    class_name = idx_to_class[yhat.item()]
    return p_max.item()*100, class_name

def main():
    st.title('Garbage Classification')
    st.subheader('Model: GarbageClassifier. Dataset: Custom')
    option = st.selectbox('How would you like to give the input?', ('Upload Image File', 'Run Example Image'))
    if option == "Upload Image File":
        file = st.file_uploader("Please upload an image of garbage", type=["jpg", "png"])
        if file is not None:
            image = Image.open(file)
            p, label = inference(image, model)
            st.image(image)
            st.success(f"The uploaded image is of the class {label} with {p:.2f} % probability.")

    elif option == "Run Example Image":
        image = Image.open('plastic_1014.jpg')
        p, label = inference(image, model)
        st.image(image)
        st.success(f"The image is of the class {label} with {p:.2f} % probability.")

if __name__ == '__main__':
    main() 
