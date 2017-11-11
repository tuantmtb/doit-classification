from flask import Flask, request
import model
from flask import jsonify
import requests
import json

app = Flask(__name__)

# api_update_label_doc = 'http://doit.uet.vnu.edu.vn/api/documents/updatelabel'
api_update_label_doc = 'http://localhost:8000/api/documents/'


@app.route("/")
def hello():
    return "Hello World!"


@app.route("/predict", methods=['POST'])
def predict():
    req = request.json
    doc = req.get('content')
    print(doc)
    predict_set = model.load_text(doc)
    names_predict= model.predict(predict_set)
    names_predict_output = json.dumps(names_predict)
    print('names_predict_output:', names_predict_output)

    # update field if sent document_id
    document_id = req.get('document_id')
    if (document_id is not None):
        print(document_id)
        url = api_update_label_doc + str(document_id) + '/updatelabel'
        try:
            r = requests.post(url, data=names_predict_output)
            print('updated label for doc_id = ' + str(document_id))
        except:
            print('can not sent request update')

    return jsonify(names_predict)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4000)
