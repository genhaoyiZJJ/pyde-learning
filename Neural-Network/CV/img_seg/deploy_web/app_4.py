"""
对图片进行分割推理
"""
from flask import Flask, render_template, request
import time
import os
import cv2
import sys
sys.path.append('.')
from predictor import Predictor


# 初始化
BASEDIR = os.path.dirname(__file__)
app = Flask(__name__)
upload_dir = os.path.join(BASEDIR, "static", "upload_img")
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)


input_size = 512
model_weights = r"ouputs\20230627-0856_portrait\best.pth"
predictor = Predictor(model_weights, input_size=input_size, tta=True)


def save_img(file, out_dir):
    time_stamp = str(time.time())
    file_name = time_stamp + file.filename
    path_to_img = os.path.join(out_dir, file_name)
    file.save(path_to_img)
    return path_to_img, file_name


def gen_html(matting, img_bgr, file_name, out_dir):
    matting_name = os.path.splitext(file_name)[0] + "_matting.jpg"
    # resize_name = os.path.splitext(file_name)[0] + "_resize.jpg"
    path_matting = os.path.join(out_dir, matting_name)
    path_resize = os.path.join(out_dir, file_name)
    cv2.imencode('.jpg', matting)[1].tofile(path_matting)
    # cv2.imencode('.jpg', img_bgr)[1].tofile(path_resize)

    show_info = {"path_resize": path_resize,
                 "path_matting": path_matting,
                 "width": img_bgr.shape[1],
                 "height": img_bgr.shape[0]}
    name_html = "matting_result_{}.html".format(file_name)
    path_html = os.path.join(BASEDIR, "templates", name_html)

    html_template(show_info, path_html)
    return name_html


def html_template(show_info, path_html):
    html_string_start = """
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Matting result</title>
    </head>

    <style>
    #left-bar {
      position: fixed;
      display: table-cell;
      top: 100;
      bottom: 10;
      left: 10;
      width: 50%;
      overflow-y: auto;
    }

    #right-bar {
      position: fixed;
      display: table-cell;
      top: 100;
      bottom: 10;
      right: 10;
      width: 35%;
      overflow-y: auto;
    }
    </style>
    <body>
    """

    html_string_end = """

    </body>
    </html>

    """

    path_resize = "../static" + show_info["path_resize"].split("static")[-1]
    img_resize_html = """<div id= "left-bar" > 
    <picture> <img src="{}" height="{}" width="{}"> </picture> <br>原始图片<br>""".format(
        path_resize, show_info["height"], show_info["width"])

    path_matting = "../static" + show_info["path_matting"].split("static")[-1]
    img_matting_html = """<div id= "right-bar" > 
    <picture> <img src="{}" height="{}" width="{}"> </picture><br>效果图片<br>""".format(
        path_matting, show_info["height"], show_info["width"])

    file_content = html_string_start + img_resize_html + img_matting_html + html_string_end
    with open(path_html, 'w', encoding="utf-8") as f:
        f.write(file_content)


# 定义该url接收get和post请求， 可到route的add_url_rule函数中看到默认是get请求
@app.route("/", methods=["GET", "POST"])
def func():
    # request 就是一个请求对象，用户提交的请求信息都在request中
    print('='*10, request.method, '='*10)
    if request.method == "POST":
        try:
            # step1: 接收传入的图片
            f = request.files['imgfile']
            path_img, file_name = save_img(f, upload_dir)
            # step2：推理
            img_bgr = cv2.imread(path_img)
            img_matting = predictor.predict(img_bgr)
            # step3: 生成用于展示的html
            name_html = gen_html(img_matting, img_bgr, file_name, upload_dir)
            return render_template(name_html)
        except Exception as e:
            return f"{e}, Please try it again!"
    else:
        return render_template("upload.html")


if __name__ == '__main__':
    app.run()

    # 允许外部访问，但无公网IP，仅局域网内其他主机可访问，如同wifi下的设备，本机IP可通过ipconfig命令查看， mac通过ipconfig /a
    # app.run(host="0.0.0.0", port=80)
