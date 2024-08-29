import time
import cv2
import numpy as np
import torch

from models import build_model


def init_model(model_weights, device):
    ckpt = torch.load(model_weights)
    model = build_model('bisenetv1', num_cls=1)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.to(device=device)
    
    model(torch.rand(1, 3, 512, 512, device=device))
    return model


def run(model, src, device):
    # preproc 
    img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (512, 512)).astype(np.float32)
    img /= 255.
    img -= 0.5
    img /= 0.5
    img = img.transpose((2, 0 ,1))
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # inference
    torch.cuda.synchronize()
    t_start = time.time()
    with torch.no_grad():
        outputs = model(img).squeeze(0)  # (1, 512, 512)

    torch.cuda.synchronize()
    t_stop = time.time()
    use_time = (t_stop-t_start) * 1000
    print(f"inference time use: {use_time:.3f} ms")

    # postproc
    pred = outputs.sigmoid().squeeze(0).cpu().numpy()  # (512, 512)
    h0, w0 = src.shape[:2]
    pred = cv2.resize(pred, (w0, h0))
    
    background = np.ones_like(src, dtype=np.uint8)  # (512, 512, 3)
    background[:] = 255
    pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)  # (512, 512, 3)
    result = np.uint8(pred * src + (1-pred) * background)  # (512, 512, 3)

    return result


def show_result(src, result, save_path=None):
    out_img = np.concatenate([src, result], axis=1)
    if save_path is None:
        cv2.imshow('out_img', out_img)
        cv2.waitKey()
    else:
        cv2.imwrite(save_path, out_img)


if __name__ == "__main__":
    model_weights = r"ouputs\20230627-0856_portrait\best.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_model(model_weights, device=device)
    
    img_files = [
        # r"e:\data\Portrait-dataset-2000\testing\00002.png",
        # r"e:\data\Portrait-dataset-2000\testing\00003.png",
        r"D:\User\桌面\20230712215737.png",
    ]
    for path in img_files:
        img_bgr = cv2.imread(path)
        result = run(model, img_bgr, device)

        show_result(img_bgr, result, save_path=None)