import os
import cloudpickle
import numpy as np
import pandas as pd
from sklearn import set_config

from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

set_config(transform_output="pandas")


def get_pipeline():
    # DEFINICIÓN DE LA PIPELINE + HIPERPARÁMETROS SELECCIONADOS
    # Transformador para variables categóricas
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore'))
    ])

    # Transformador para variables numéricas
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Columnas por tipo de dato
    transformer = ColumnTransformer(transformers=[
        ('cat', cat_transformer, make_column_selector(dtype_include=["category", "object"])),
        ('num', num_transformer, make_column_selector(dtype_exclude=["category", "object"]))
    ], remainder='passthrough', verbose_feature_names_out=False)

    # Clasificador XGBoost con hiperparámetros optimizados
    xgb = XGBClassifier(
        eval_metric="auc",
        use_label_encoder=False,
        random_state=42,
        subsample=0.8,
        scale_pos_weight=3,
        reg_lambda=0.5,
        reg_alpha=0,
        n_estimators=150,
        min_child_weight=5,
        max_depth=4,
        learning_rate=0.01,
        gamma=2.0,
        colsample_bytree=0.8
    )

    # Pipeline completo con SMOTE para balanceo de clases
    return ImbPipeline(steps=[
        ('transformer', transformer),
        ('variance_threshold', VarianceThreshold()),
        ('resampler', SMOTE(random_state=42, sampling_strategy=0.5, k_neighbors=7)),
        ('select_features', SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000))),
        ('clf', xgb)
    ])


def get_X_y():
    # CODIGO PARA CALCULAR EL DATASET QUE ENTRA A ENTRENAMIENTO Y PARA CALCULAR EL TARGET
    bookings = pd.read_csv("bookings_train.csv")
    hotels = pd.read_csv("hotels.csv")
    df = bookings.merge(hotels, on="hotel_id", how="left")

    # Filtrado de datos
    df = df[df["reservation_status"] != "Booked"]
    df = df[df["total_guests"] != 0]

    # Tratamiento de outliers en la tarifa
    q05 = df["rate"].quantile(0.05)
    q95 = df["rate"].quantile(0.95)
    df = df[df["rate"].between(q05, q95).fillna(False)]

    # Conversión de fechas
    df["booking_date"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")

    # Creación de features
    df["is_foreign"] = (df["country_x"] != df["country_y"]).astype(int)
    df["extras"] = df[["pool_and_spa", "restaurant", "parking"]].sum(axis=1)
    df["days_in_advance"] = (df["arrival_date"] - df["booking_date"]).dt.days

    # Manejo básico de NAs y reset del índice
    df = df.fillna(np.nan)
    df = df.reset_index(drop=True)

    # Creación de la variable objetivo (target)
    df_target = df.copy()
    df_target["arrival_date"] = pd.to_datetime(df_target["arrival_date"], errors="coerce")
    df_target["reservation_status_date"] = pd.to_datetime(df_target["reservation_status_date"], errors="coerce")
    days_to_arrival = (df_target["arrival_date"] - df_target["reservation_status_date"]).dt.days
    days_to_arrival = days_to_arrival.fillna(-999)

    # Target: cancelaciones dentro de los 30 días anteriores a la llegada
    y = ((df_target["reservation_status"] == "Canceled") & (days_to_arrival <= 30)).astype(int)

    # Columnas a eliminar
    drop_columns = [
        "reservation_status", "reservation_status_date", "hotel_id",
        "country_x", "country_y", "pool_and_spa", "restaurant",
        "parking", "arrival_date", "booking_date", "board", "market_segment",
    ]
    X = df.drop(columns=drop_columns)

    # Convertir columnas categóricas al tipo adecuado
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        X[col] = X[col].astype('category')

    return X, y


def save_pipeline(pipe):
    # AQUI SE SERIALIZA EL MODELO CON CLOUDPICKLE
    with open(os.environ["MODEL_PATH"], mode="wb") as f:
        cloudpickle.dump(pipe, f)


if __name__ == "__main__":
    X, y = get_X_y()
    pipe = get_pipeline()
    pipe.fit(X, y)
    save_pipeline(pipe)