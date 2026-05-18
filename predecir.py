import os
import joblib
import numpy as np
import pandas as pd

BASE  = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, 'model')

model      = joblib.load(os.path.join(MODEL, 'modelo.pkl'))
columnas   = joblib.load(os.path.join(MODEL, 'columnas.pkl'))
freq_brand = joblib.load(os.path.join(MODEL, 'freq_brand.pkl'))
freq_model = joblib.load(os.path.join(MODEL, 'freq_model.pkl'))

def predecir(datos_usuario: dict) -> float:
    datos = datos_usuario.copy()
    marca  = datos.pop('brand', None)
    modelo = datos.pop('model', None)

    df = pd.DataFrame([datos])
    print("1. After DataFrame:", df.shape, df.columns.tolist())

    df['brand_freq'] = freq_brand.get(marca, 0)
    df['model_freq'] = freq_model.get(modelo, 0)
    df['antiguedad'] = 2026 - df['yearOfRegistration']
    print("2. After freq+antiguedad:", df.shape)

    df = pd.get_dummies(df, columns=['notRepairedDamage', 'vehicleType', 'fuelType'])
    print("3. After get_dummies:", df.shape, df.columns.tolist())

    df = df.astype({col: int for col in df.select_dtypes('bool').columns})
    print("4. columnas esperadas:", columnas)

    df = df.reindex(columns=columnas, fill_value=0)
    print("5. After reindex:", df.shape)

    pred_log = model.predict(df)
    return float(np.expm1(pred_log)[0])


if __name__ == '__main__':
    datos_prueba = {
        'brand':              'bmw',
        'model':              '3er',
        'yearOfRegistration': 1995,
        'monthOfRegistration': 10,
        'yearOfPurchase':     2016,
        'monthOfPurchase':    4,
        'kilometer':          150000,
        'powerPS':            102,
        'vehicleType':        'sedan',
        'fuelType':           'gasoline',
        'notRepairedDamage':  'yes',
    }

    precio = predecir(datos_prueba)
    print(f"Precio estimado: {precio:.2f} €")