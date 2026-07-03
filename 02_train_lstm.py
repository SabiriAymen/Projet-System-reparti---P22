"""
Etape 2 : Modèles LSTM
- Modèle A (baseline)   : Close, Volume  -> prédiction du Close du lendemain
- Modèle B (sentiment)  : Close, Volume, sentiment_mean, tweet_volume -> idem
- Même architecture, même split, mêmes hyperparamètres -> comparaison équitable
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW = 10        # nb de jours passés utilisés pour prédire le lendemain
TEST_RATIO = 0.15
VAL_RATIO = 0.15

df = pd.read_csv("output/dataset_final.csv", parse_dates=["Date"])


def make_sequences(features: np.ndarray, target: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(features)):
        X.append(features[i - window:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def split_train_val_test(X, y, test_ratio, val_ratio):
    n = len(X)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
    return (X[:n_train], y[:n_train],
            X[n_train:n_train + n_val], y[n_train:n_train + n_val],
            X[n_train + n_val:], y[n_train + n_val:])


def build_model(n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(WINDOW, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def run_experiment(feature_cols, name):
    print(f"\n{'=' * 60}\nModèle : {name}  |  features = {feature_cols}\n{'=' * 60}")

    data = df[feature_cols].values.astype(float)
    target = df["Close"].shift(-1).values.astype(float)  # prédire le Close du lendemain
    # on enlève la dernière ligne (pas de "lendemain" pour elle)
    data, target = data[:-1], target[:-1]

    # Scaling (fit uniquement sur la portion train pour éviter la fuite de données)
    n_total = len(data)
    n_test = int(n_total * TEST_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_train = n_total - n_test - n_val

    feat_scaler = MinMaxScaler()
    feat_scaler.fit(data[:n_train])
    data_scaled = feat_scaler.transform(data)

    target_scaler = MinMaxScaler()
    target_scaler.fit(target[:n_train].reshape(-1, 1))
    target_scaled = target_scaler.transform(target.reshape(-1, 1)).flatten()

    X, y = make_sequences(data_scaled, target_scaled, WINDOW)
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X, y, TEST_RATIO, VAL_RATIO)

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    model = build_model(n_features=len(feature_cols))
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100, batch_size=16, verbose=0, callbacks=[es],
    )

    pred_scaled = model.predict(X_test, verbose=0).flatten()
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae = float(mean_absolute_error(actual, pred))
    r2 = float(r2_score(actual, pred))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)

    print(f"RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}  MAPE={mape:.2f}%  (epochs run={len(history.history['loss'])})")

    return {
        "name": name,
        "features": feature_cols,
        "rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
        "epochs_run": len(history.history["loss"]),
        "history_loss": history.history["loss"],
        "history_val_loss": history.history["val_loss"],
        "actual": actual.tolist(),
        "pred": pred.tolist(),
        "test_dates": df["Date"].iloc[-len(actual):].dt.strftime("%Y-%m-%d").tolist(),
    }


results = {}
results["baseline"] = run_experiment(["Close", "Volume"], "A - Sans sentiment (baseline)")
results["sentiment"] = run_experiment(["Close", "Volume", "sentiment_mean", "tweet_volume"], "B - Avec sentiment")

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 60}\nRÉCAPITULATIF COMPARATIF\n{'=' * 60}")
print(f"{'Métrique':<10}{'Sans sentiment':>18}{'Avec sentiment':>18}{'Amélioration':>15}")
for metric in ["rmse", "mae", "mape"]:
    a, b = results["baseline"][metric], results["sentiment"][metric]
    improv = (a - b) / a * 100
    print(f"{metric.upper():<10}{a:>18.3f}{b:>18.3f}{improv:>14.1f}%")
r2a, r2b = results["baseline"]["r2"], results["sentiment"]["r2"]
print(f"{'R2':<10}{r2a:>18.4f}{r2b:>18.4f}{(r2b - r2a):>14.4f}")
