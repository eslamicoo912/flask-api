from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

portability_model = tf.keras.models.load_model("models/portability/model.h5")
level_model = tf.keras.models.load_model("models/level/model.h5")
tank1_model = tf.keras.models.load_model("models/Tank 1/model.h5")
tank2_model = tf.keras.models.load_model("models/Tank 2/model.h5")
leaks_acc_branched_model = tf.keras.models.load_model(
    "models/leaks/Acc/Branched/model/model.h5")
leaks_acc_looped_model = tf.keras.models.load_model(
    "models/leaks/Acc/Looped/model/model.h5")

leaks_hydro_branched_model = tf.keras.models.load_model(
    "models/leaks/Hydrophone/Branched/model/model.h5")
leaks_hydro_looped_model = tf.keras.models.load_model(
    "models/leaks/Hydrophone/Looped/model/model.h5")

leaks_press_branched_model = tf.keras.models.load_model(
    "models/leaks/Pressure/Branched/model/model.h5")
leaks_press_looped_model = tf.keras.models.load_model(
    "models/leaks/Pressure/Looped/model/model.h5")

treatment_model = tf.keras.models.load_model('models/treatment/best_model.h5')


@app.route("/")
def running():
    return 'Flask is running now'


@app.route('/portability', methods=['POST'])
def predict_portability():
    data = request.get_json(force=True)
    input_data = preprocess_portability_data(data)

    predictions = portability_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route('/level', methods=['POST'])
def predict_level():
    data = request.get_json(force=True)
    input_data = preprocess_level_data(data)

    predictions = level_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route("/tank1", methods=['POST'])
def predict_tank1():
    data = request.get_json(force=True)
    input_data = preprocess_tank1_data(data)

    predictions = tank1_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route("/tank2", methods=['POST'])
def predict_tank2():
    data = request.get_json(force=True)
    input_data = preprocess_tank2_data(data)

    predictions = tank2_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route('/leaks/acc_branched', methods=['POST'])
def predict_leaks_acc_branched():
    data = request.get_json(force=True)
    input_data = preprocess_leaks_acc(data)

    predictions = leaks_acc_branched_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route('/leaks/acc_looped', methods=['POST'])
def predict_leaks_acc_looped():
    data = request.get_json(force=True)
    input_data = preprocess_leaks_acc(data)

    predictions = leaks_acc_looped_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route('/leaks/hydro_branched', methods=['POST'])
def predict_leaks_hydro_branched():
    data = request.get_json(force=True)
    input_data = preprocess_leaks_hydro(data)

    predictions = leaks_hydro_branched_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route('/leaks/hydro_looped', methods=['POST'])
def predict_leaks_hydro_looped():
    data = request.get_json(force=True)
    input_data = preprocess_leaks_hydro(data)

    predictions = leaks_hydro_looped_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route("/leaks/press_branched", methods=['POST'])
def predict_leaks_press_branched():
    data = request.get_json(force=True)
    input_data = preprocess_leaks_press(data)

    predictions = leaks_press_branched_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route("/leaks/press_looped", methods=['POST'])
def predict_leaks_press_looped():
    data = request.get_json(force=True)
    input_data = preprocess_leaks_press(data)

    predictions = leaks_press_looped_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


@app.route("/treatment", methods=['POST'])
def predict_treatment():
    data = request.get_json(force=True)
    input_data = preprocess_treatment(data)

    predictions = treatment_model.predict(input_data)

    response = {'predictions': predictions.tolist()}

    return jsonify(response)


def preprocess_portability_data(data):
    features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    # Create a numpy array with shape (1, 9) to hold the input data
    input_data = np.zeros((1, len(features)), dtype=np.float32)

    for i, feat in enumerate(features):
        input_data[0, i] = data[feat]

    return input_data


def preprocess_level_data(data):
    features = ['Hour']
    input_data = np.zeros((1, len(features)), dtype=np.float32)

    for i, feat in enumerate(features):
        input_data[0, i] = data[feat]

    return input_data


def preprocess_tank1_data(data):
    features = ['entry_id', 'year', 'month', 'day', 'hour', 'minute']

    input_data = np.zeros((1, 1, len(features)), dtype=np.float32)

    for i, feat in enumerate(features):
        input_data[0, 0, i] = data[feat]

    return input_data


def preprocess_tank2_data(data):
    features = ['entry_id', 'year', 'month', 'day', 'hour', 'minute']

    input_data = np.zeros((1, 1, len(features)), dtype=np.float32)

    for i, feat in enumerate(features):
        input_data[0, 0, i] = data[feat]

    return input_data


def preprocess_leaks_acc(data):
    features = ['value1', 'value2', 'value3', 'value4',
                'value5', 'value6', 'value7', 'value8']

    input_data = np.zeros((1, len(features)), dtype=np.float32)

    for i, feat in enumerate(features):
        input_data[0, i] = data[feat]

    return input_data


def preprocess_leaks_hydro(data):
    features = ['value1', 'value2', 'value3', 'value4',
                'value5', 'value6', 'value7', 'value8', 'value9', 'value10', 'value11', 'value12']

    input_data = np.zeros((1, len(features)), dtype=np.float32)

    for i, feat in enumerate(features):
        input_data[0, i] = data[feat]

    return input_data


def preprocess_leaks_press(data):
    features = ['value1', 'value2', 'value3', 'value4',
                'value5', 'value6', 'value7', 'value8']

    input_data = np.zeros((1, len(features)), dtype=np.float32)

    for i, feat in enumerate(features):
        input_data[0, i] = data[feat]

    return input_data


def preprocess_treatment(data):
    features = ['Var', 'Variable', 'COU', 'Country', 'YEA',
                'Year', 'Unit Code',	'Unit', 'PowerCode Code',	'PowerCode']
    input_data = np.zeros((1, 1, len(features)), dtype=np.float32)

    for i, feat in enumerate(features):
        input_data[0, 0, i] = data[feat]

    return input_data


if __name__ == "__main__":
    app.run()
