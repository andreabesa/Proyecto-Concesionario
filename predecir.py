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

    df['brand_encoded'] = freq_brand.get(marca, 0)
    df['model_encoded'] = freq_model.get(modelo, 0)
    df['antiguedad'] = 2026 - df['yearOfRegistration']
    
    df = pd.get_dummies(df, columns=['notRepairedDamage', 'vehicleType', 'fuelType'])
    df = df.astype({col: int for col in df.select_dtypes('bool').columns})
    df = df.reindex(columns=columnas, fill_value=0)

    pred_log = model.predict(df)
    return float(np.expm1(pred_log)[0])


if __name__ == '__main__':
    datos_prueba = {
        'brand':              'toyota',
        'model':              'auris',
        'yearOfRegistration': 2016,
        'monthOfRegistration': 6,
        'yearOfPurchase':     2026,
        'monthOfPurchase':    3,
        'kilometer':          20000,
        'powerPS':            130,
        'vehicleType':        'sedan',
        'fuelType':           'gasoline',
        'notRepairedDamage':  'no',
    }

    precio = predecir(datos_prueba)
    print(f"Precio estimado: {precio:.2f} €")