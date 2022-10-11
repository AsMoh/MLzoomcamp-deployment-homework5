import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model2.bin'
dict_v='dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open(dict_v, 'rb') as f_in1:
    dict_vict = pickle.load(f_in1)


app = Flask('credit')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dict_vict.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    credit_card = y_pred >= 0.5

    result = {
        'credit_card_probability': float(y_pred),
        'credit_card': bool(credit_card)
    }
    print(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)