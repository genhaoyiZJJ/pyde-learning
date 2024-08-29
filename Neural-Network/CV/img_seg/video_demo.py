import cv2
from predictor import Predictor


if __name__ == '__main__':

    input_size = 512
    model_weights = r"ouputs\20230627-0856_portrait\best.pth"
    predictor = Predictor(model_weights, input_size=input_size, tta=False)

    video_path = r"xujingyu.mp4"
    # video_path = 0 + cv2.CAP_DSHOW  # 0表示打开视频，cv2.CAP_DSHOW去除黑边
    vid = cv2.VideoCapture(video_path)  

    while True:
        # read a frame
        ret, img_bgr = vid.read()
        if img_bgr is None or not ret:
            continue

        # 按q键结束提前退出, ord: 字符串转ASCII数值
        if cv2.waitKey(1) == ord('q'):
            break

        # predict
        result = predictor.predict(img_bgr, color="w")

        # show
        predictor.show_result(img_bgr, result, save_path=None, delay=1)

    vid.release()  # release()释放摄像头