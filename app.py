from predecir import predecir
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://ppvsm26.firebaseapp.com", "https://ppvsm26.web.app"])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        datos = request.get_json()
        precio = predecir(datos)
        return jsonify({'precio': round(precio, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)