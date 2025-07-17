import os
import cloudpickle
import pandas as pd
import numpy as np
from sklearn import set_config

set_config(transform_output="pandas")

def get_X():
    """
    Carga y preprocesa los datos para inferencia, aplicando
    las mismas transformaciones que en el entrenamiento.
    """
    # Carga del CSV de inferencia
    df = pd.read_csv(os.environ["INFERENCE_DATA_PATH"])

    # Si necesitamos datos adicionales de hoteles para la inferencia
    hotels_path = os.path.dirname(os.environ["INFERENCE_DATA_PATH"]) + "/hotels.csv"
    if os.path.exists(hotels_path):
        hotels = pd.read_csv(hotels_path)
        df = df.merge(hotels, on="hotel_id", how="left")

    # Aplicar las mismas transformaciones que en el entrenamiento
    # Convertir fechas
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")

    # Crear características derivadas si están disponibles las columnas necesarias
    if all(col in df.columns for col in ["country_x", "country_y"]):
        df["is_foreign"] = (df["country_x"] != df["country_y"]).astype(int)

    if all(col in df.columns for col in ["pool_and_spa", "restaurant", "parking"]):
        df["extras"] = df[["pool_and_spa", "restaurant", "parking"]].sum(axis=1)

    if all(col in df.columns for col in ["arrival_date", "booking_date"]):
        df["days_in_advance"] = (df["arrival_date"] - df["booking_date"]).dt.days

    # Manejo de valores nulos
    df = df.fillna(np.nan)

    # Eliminar columnas que no se usaron en entrenamiento
    drop_columns = [
        "reservation_status", "reservation_status_date", "hotel_id",
        "country_x", "country_y", "pool_and_spa", "restaurant",
        "parking", "arrival_date", "booking_date", "board", "market_segment"
    ]

    # Solo eliminar columnas que existen en el dataframe
    columns_to_drop = [col for col in drop_columns if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # Convertir columnas categóricas
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    return df

def get_pipeline():
    """
    Carga el modelo serializado desde la ruta especificada en las variables de entorno.
    """
    with open(os.environ["MODEL_PATH"], mode="rb") as f:
        pipe = cloudpickle.load(f)

    return pipe

def get_predictions(pipe, X=None):
    """
    Genera predicciones utilizando el modelo cargado y los datos preprocesados.
    Devuelve una Series con las predicciones en el mismo orden que el dataframe original.
    """
    if X is None:
        X = get_X()

    # Guardar índice original para mantener el orden
    original_index = X.index

    # Predecir con el pipeline
    predictions = pipe.predict(X)

    # Crear Series con las predicciones manteniendo el índice original
    preds = pd.Series(predictions, index=original_index, name="prediction")

    return preds

if __name__ == "__main__":
    # Cargar datos y modelo
    X = get_X()
    pipe = get_pipeline()

    # Realizar predicciones
    preds = get_predictions(pipe, X)

    # Guardar predicciones en formato solicitado
    preds.to_csv("output_predictions.csv", header=True, index=False)