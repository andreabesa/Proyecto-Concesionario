from flask import Flask, request, jsonify, send_from_directory
from predecir import predecir
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'predictor-precios.html')

@app.route('/predecir', methods=['POST'])
def predict():
    datos = request.get_json()
    precio = predecir(datos)
    return jsonify({'precio_estimado': round(precio, 2)})

if __name__ == '__main__':
    app.run(debug=True)