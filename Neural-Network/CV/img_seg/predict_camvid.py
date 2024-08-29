"""分割代码预测脚本
"""
import time
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from models import build_model


timestamp = [0] * 4

# step1 load model
model_weights = r"ouputs\20230624-1001_camvid\best.pth"
checkpoint = torch.load(model_weights)

model = build_model('deeplabv3+', num_cls=12)
model.load_state_dict(checkpoint['model'])
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# step2 read image
img_path = r"e:\data\CamVid\test\Seq05VD_f04050.png"
img0 = Image.open(img_path)

# step3 preprocess
timestamp[0] = time.time()
transform_img = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
img = transform_img(img0).unsqueeze(0).to(device)
timestamp[1] = time.time()

# step4 inference
with torch.no_grad():
    outputs = model(img)  # (1, num_cls, 360, 480)
    
torch.cuda.synchronize()
timestamp[2] = time.time()

# step5 postprecess
outputs = torch.max(outputs, dim=1)[1]  # (1, 360, 480)
pred_label = outputs.squeeze(0).cpu().numpy()
timestamp[3] = time.time()

# step6 imshow
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img0)
ax1.set_title('src')
ax2.imshow(pred_label)
ax2.set_title('pred')
plt.show()