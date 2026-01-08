import torch
from train import SimpleCNN

def predict(image_tensor, model_path="models/factory_cnn.pt", num_classes=50):
    model = SimpleCNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))
        _, pred = torch.max(outputs, 1)
    return pred.item()

# Beispiel:
# from PIL import Image
# import torchvision.transforms as transforms
# img = Image.open("data/part_a/sample1.jpg")
# transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
# tensor = transform(img)
# print("Prediction:", predict(tensor))
