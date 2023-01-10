import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)#初始化APP

model = pickle.load(open("ufo-model.pkl", "rb"))#加载模型


@app.route("/")#装饰器
def home():
    return render_template("index.html")#先引入index.html，同时根据后面传入的参数，对html进行修改渲染。


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]#存储用户输入的参数
    final_features = [np.array(int_features)]#将用户输入的值转化为一个数组
    prediction = model.predict(final_features)#输入模型进行预测

    output = prediction[0]#将预测值传入output

    countries = ["Australia", "Canada", "Germany", "UK", "US"]#根据预测值判断国家

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])#将预测值返回到Web界面，使我们看到
    )


if __name__ == "__main__":
    app.run(debug=True)#调试模式下运行文件，实时反应结果。仅限测试使用，生产模式下不要使用