'''model.py'''
import torch
from torchvision import transforms
from PIL import Image
from neural_networks.ConvolutionalClassifier import ConvolutionalClassifier
from neural_networks.FullyConnectedClassifier import FullyConnectedClassifier

def build_model(name:str):
    if name == "fc":  return FullyConnectedClassifier()
    if name == "cnn": return ConvolutionalClassifier()
    raise ValueError("unknown model")

_tf = transforms.Compose([transforms.Grayscale(), transforms.Resize((28,28)), transforms.ToTensor()])
def preprocess(img: Image.Image):
    return _tf(img).unsqueeze(0)

def predict(model_name:str, weights_path:str, img:Image.Image) -> int:
    net = build_model(model_name)
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()
    with torch.no_grad():
        y = net(preprocess(img))
    return int(torch.argmax(y, 1).item())
