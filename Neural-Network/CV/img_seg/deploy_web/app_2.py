"""
采用HTML展示信息
"""
from flask import Flask, render_template


app = Flask(__name__)


@app.route("/")
def show_something():
    return render_template("first_html.html")   # 默认路径为同级目录下的templates文件夹下寻找***.html文件


if __name__ == '__main__':
    app.run()

