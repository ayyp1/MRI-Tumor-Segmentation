from models import UNet
import torch 
import albumentations as A 
import cv2
import torchvision.transforms as T 
import numpy as np 

device = torch.device('cpu')

path = 'saved_model/best-model.pth'
model = UNet(3, 1).to(device)
model.load_state_dict(torch.load(path , map_location=torch.device('cpu')))
model.eval()

def evaluate(image): 
    transform = A.Compose([
    A.Resize(width=128, height=128, p=1.0),  
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image = image)['image']

    pred = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0).permute(0,3,1,2)
    pred = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(pred)
    with torch.no_grad():
        pred = model(pred.to(device))
        pred = pred.detach().cpu().numpy()[0,0,:,:]
    
    return pred 

if __name__ == "__main__":
    image = cv2.imread("sample.jpg")  
    output = evaluate(image)
    cv2.imwrite("output.jpg")
