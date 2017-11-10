from flask import Flask, request
import model
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/predict", methods=['POST'])
def predict():
    req = request.json
    # doc = model.load_file(
    #     '/Volumes/DATA/workspace/vnu/spc/git/doit-classification/DataSource/Data_raw/test/NhanVan/Luan an Le Tien Dung.txt')
    doc = req.get('content')
    print(doc)
    predict_set = model.load_text(doc)

    index_classification, target_name = model.predict(predict_set)
    print(index_classification)
    print(target_name)
    return target_name
    # test()


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)