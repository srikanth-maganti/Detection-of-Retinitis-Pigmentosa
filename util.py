import torch
from torchvision import transforms,models
import torch.nn as nn
from PIL import Image

transformation=transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),         
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  
])

class Resnet_Model(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.network=models.resnet50(pretrained=False)
        self.network.fc=nn.Linear(self.network.fc.in_features,num_classes)
    def forward(self,xb):
        return self.network(xb)

def predict(image):
    # Load the model
    model = Resnet_Model(in_channels=3, num_classes=2)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    

    image = Image.open(image).convert("RGB")
    image = transformation(image).unsqueeze(0)  

    
    with torch.no_grad():
        output = model(image)
        print(output)
        _, predicted = torch.max(output, 1)

    return predicted.item()