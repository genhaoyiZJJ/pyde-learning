"""
分割模型封装
"""
import os
import time
import cv2
import torch
import numpy as np
from models import build_model
import matplotlib.pyplot as plt
import ttach as tta


def clock(func):
    
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.time()
        
        output = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_stop = time.time()

        use_time = (t_stop-t_start) * 1000
        print(f"{func.__name__} time use: {use_time:.3f} ms")
        return output

    return wrapper


class Predictor(object):
    def __init__(self, model_weights, input_size, 
                 model_name="bisenetv1", tta=False):
        self.model_name = model_name
        self.model_weights = model_weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        self.tta = tta
        self.model = self._init_model()
        print("!!init done!!")

    def predict(self, img, **kwargs):
        if isinstance(img, str):
            img = cv2.imread(img)

        # preprocess
        input_tensor = self._preprocess(img)

        # inference
        outputs = self.forward(input_tensor)

        # postprocess
        result = self._postprocess(img, outputs, **kwargs)

        return result

    @clock
    def _init_model(self):
        model = build_model(self.model_name, num_cls=1, pretrained=False)

        ckpt = torch.load(self.model_weights, map_location=self.device)
        model.load_state_dict(ckpt['model'])
        model.eval()
        model.to(device=self.device)

        if self.tta:
            transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    # tta.Rotate90(angles=[0, 180]),
                    # tta.Scale(scales=[1, 2, 4]),
                    # tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
            model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')

        model(torch.rand(1, 3, 512, 512, device=self.device))

        return model

    def _preprocess(self, img_bgr):
        if isinstance(self.input_size, int):
            self.input_size = (self.input_size, self.input_size)
        input_h, input_w = self.input_size[0], self.input_size[1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, (input_w, input_h)).astype(np.float32)
        img /= 255.
        img -= 0.5
        img /= 0.5
        img = img.transpose((2, 0 ,1))
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return img

    @clock
    @torch.no_grad()
    def forward(self, input_tensor):
        outputs = self.model(input_tensor).squeeze(0)  # (1, 512, 512)
        return outputs

    def _postprocess(self, img, outputs, color="w", hide=False):
        pred = outputs.sigmoid().squeeze(0).cpu().numpy()  # (512, 512)

        # 背景颜色
        background = np.zeros_like(img, dtype=np.uint8)
        if color == "b":
            background[:, :, 0] = 255
        elif color == "w":
            background[:] = 255
        elif color == "r":
            background[:, :, 2] = 255

        # alpha
        alpha = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        alpha = cv2.resize(alpha, (w, h))

        # fusion
        if not hide:
            result = np.uint8(img * alpha + background * (1 - alpha))
        else:
            result = np.uint8(background * alpha + img * (1 - alpha))
        return result

    @staticmethod
    def show_result(src, result, save_path=None, delay=0):
        out_img = np.concatenate([src, result], axis=1)
        if save_path is None:
            cv2.imshow('out_img', out_img)
            cv2.waitKey(delay)
        else:
            base_dir = os.path.dirname(save_path)
            os.makedirs(base_dir, exist_ok=True)
            cv2.imwrite(save_path, out_img)

    @staticmethod
    def save_img(path_img, img_src):
        base_dir = os.path.dirname(path_img)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        cv2.imwrite(path_img, img_src)


if __name__ == '__main__':

    input_size = 512
    model_weights = r"ouputs\20230627-0856_portrait\best.pth"
    predictor = Predictor(model_weights, input_size=input_size, tta=True)

    set_name = "testing"
    root_dir = r"E:\data\Portrait-dataset-2000\{}".format(set_name)
    out_dir = os.path.join(os.path.dirname(model_weights), "{}_{}".format(set_name, input_size))

    ## image
    names_lst = os.listdir(root_dir)
    names_lst = [n for n in names_lst if not n.endswith("matte.png")]
    path_imgs = [os.path.join(root_dir, n) for n in names_lst]
    for idx, path in enumerate(path_imgs[:2]):
        # read image
        img_bgr = cv2.imread(path)
        
        # predict
        result = predictor.predict(img_bgr, color="w")

        # show and save
        save_path = os.path.join(out_dir, os.path.basename(path))
        predictor.show_result(img_bgr, result, save_path=save_path)