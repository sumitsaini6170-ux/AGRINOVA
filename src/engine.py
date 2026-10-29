"""Forecasting + irrigation decision engine. Pure logic, no Streamlit."""

from pathlib import Path

import numpy as np

from src.config import (
    FEATURES,
    FLOW_LPM,
    FIELD_AREA_M2,
    ROOT_DEPTH_MM,
    STEPS,
    WINDOW,
)


# =========================================================
# NORMALIZATION FILES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MINIMUM_PATH = BASE_DIR / "models" / "minimum.npy"
MAXIMUM_PATH = BASE_DIR / "models" / "maximum.npy"


# =========================================================
# LOAD NORMALIZATION VALUES
# =========================================================

try:
    MINIMUM = np.load(MINIMUM_PATH).astype(np.float32)
    MAXIMUM = np.load(MAXIMUM_PATH).astype(np.float32)

    print("LSTM normalization values: loaded")

except Exception as e:
    MINIMUM = None
    MAXIMUM = None

    print(
        "LSTM normalization values: NOT loaded",
        f"({type(e).__name__}: {e})"
    )


# =========================================================
# MODEL LOADER
# =========================================================

def load_model(path):
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(path), None

    except Exception as e:

        return None, (
            f"{type(e).__name__}: {e}"
        )


# =========================================================
# CREATE LSTM WINDOW
# =========================================================

def make_window(df, r):
    """
    Create the last 24-hour raw input window.

    The window contains:

        temperature
        humidity
        rainfall
        soil_moisture

    Normalization is performed later in _predict().
    """

    base = (
        df[FEATURES]
        .tail(WINDOW - 1)
        .to_numpy("float32")
    )

    cur = np.array(
        [[
            r["temperature"],
            r["humidity"],
            r["rainfall"],
            r["moisture"],
        ]],
        dtype=np.float32,
    )

    return np.vstack(
        [base, cur]
    )


# =========================================================
# NORMALIZE INPUT
# =========================================================

def _normalize_input(x):
    """
    Apply the SAME Min-Max normalization used
    during model training.

    Training formula:

        (data - minimum)
        -----------------
        maximum - minimum
    """

    x = x.copy().astype(np.float32)

    if MINIMUM is None or MAXIMUM is None:
        raise RuntimeError(
            "LSTM normalization files are missing."
        )

    denominator = (
        MAXIMUM - MINIMUM
    )

    denominator = np.where(
        denominator == 0,
        1.0,
        denominator
    )

    x = (
        x - MINIMUM
    ) / denominator

    return x


# =========================================================
# INVERSE NORMALIZE MOISTURE
# =========================================================

def _inverse_moisture(y):
    """
    Convert normalized LSTM output back to
    actual soil moisture percentage.

    Training target:

        normalized[i, 3]

    Therefore column 3 must be inverse-transformed.
    """

    if MINIMUM is None or MAXIMUM is None:
        raise RuntimeError(
            "LSTM normalization files are missing."
        )

    moisture_min = float(
        MINIMUM[3]
    )

    moisture_max = float(
        MAXIMUM[3]
    )

    return (
        y * (
            moisture_max
            - moisture_min
        )
        + moisture_min
    )


# =========================================================
# MODEL PREDICTION
# =========================================================

def _predict(
    model,
    x,
    training=False,
):
    """
    Normalize input exactly as during training,
    run LSTM prediction, then convert the output
    back to real soil moisture percentage.
    """

    normalized_x = _normalize_input(
        x
    )

    prediction = np.asarray(
        model(
            normalized_x,
            training=training,
        )
    ).ravel()

    prediction = _inverse_moisture(
        prediction
    )

    return prediction


# =========================================================
# FALLBACK FORECAST
# =========================================================

def _fallback(w):
    """
    Simple fallback estimate when the LSTM model
    cannot be loaded.

    This operates on RAW values.
    """

    return (
        w[-1, 3]
        + (
            w[-1, 3]
            - w[-4, 3]
        ) / 3
        - 0.2 * w[-1, 0] / 30
        + 0.5 * w[-1, 2]
    )


# =========================================================
# FORECAST
# =========================================================

def forecast(
    model,
    win,
    steps=STEPS,
    mc=20,
):
    """
    Recursive moisture forecast.

    Returns:

        forecast list
        uncertainty standard deviation
    """

    w = win.copy()

    out = []

    std = 0.0

    for i in range(steps):

        # -------------------------------------------------
        # LSTM
        # -------------------------------------------------

        if model is None:

            p = _fallback(
                w
            )

        else:

            p = float(
                _predict(
                    model,
                    w[None],
                    training=False,
                )[0]
            )

            # ---------------------------------------------
            # MC Dropout uncertainty
            # ---------------------------------------------

            if i == 0 and mc > 1:

                samples = _predict(
                    model,
                    np.repeat(
                        w[None],
                        mc,
                        axis=0,
                    ),
                    training=True,
                )

                std = float(
                    samples.std()
                )

        # -------------------------------------------------
        # Keep moisture between 0 and 100%
        # -------------------------------------------------

        p = float(
            np.clip(
                p,
                0,
                100,
            )
        )

        out.append(
            p
        )

        # -------------------------------------------------
        # Recursive next input
        # -------------------------------------------------

        nxt = w[-1].copy()

        # Only moisture changes.
        # Temperature, humidity and rainfall
        # are held constant for this forecast.
        nxt[3] = p

        w = np.vstack(
            [
                w[1:],
                nxt,
            ]
        )

    return out, std


# =========================================================
# EVAPOTRANSPIRATION
# =========================================================

def et_mm_day(
    temp,
    hum,
):
    """
    Rough evapotranspiration heuristic.

    This is NOT a calibrated FAO-56 calculation.
    """

    return float(
        max(
            0.5,
            0.1
            * temp
            * (
                1
                - hum / 130
            ),
        )
    )


# =========================================================
# WATER REQUIREMENT
# =========================================================

def litres_needed(
    moisture,
    target,
):
    return max(
        target - moisture,
        0,
    ) / 100 * ROOT_DEPTH_MM * FIELD_AREA_M2


# =========================================================
# LITRES TO MOISTURE %
# =========================================================

def litres_to_pct(
    litres,
):
    return (
        litres
        / (
            ROOT_DEPTH_MM
            * FIELD_AREA_M2
        )
        * 100
    )


# =========================================================
# IRRIGATION DECISION
# =========================================================

def decide(
    r,
    crop,
    fc,
    raining,
    tank_ok,
):

    thr = crop["thr"]

    target = crop["target"]

    low = (
        r["moisture"]
        < thr
    )

    hrs = next(
        (
            i + 1
            for i, v in enumerate(fc)
            if v < thr
        ),
        None,
    )

    litres = litres_needed(
        r["moisture"],
        target,
    )

    plan = {
        "litres": litres,

        "minutes": (
            litres
            / FLOW_LPM
        ),

        "hours_to_thr": hrs,

        "et": et_mm_day(
            r["temperature"],
            r["humidity"],
        ),
    }

    if low and raining:

        res = (
            "warn",
            "LOW MOISTURE, RAIN DETECTED",
            "Irrigation held while it rains.",
        )

    elif low and not tank_ok:

        res = (
            "bad",
            "WATER REQUIRED, TANK LOW",
            "Refill the water tank. Pump is locked.",
        )

    elif low:

        res = (
            "bad",
            "WATER REQUIRED",
            (
                f"Moisture "
                f"{r['moisture']:.1f}% "
                f"is below "
                f"{thr}%. "
                f"Irrigate about "
                f"{plan['minutes']:.0f} min."
            ),
        )

    elif hrs:

        res = (
            "warn",
            "MOISTURE FALLING",
            (
                f"LSTM expects moisture "
                f"below {thr}% "
                f"in about {hrs} h. "
                f"Prepare irrigation."
            ),
        )

    else:

        res = (
            "ok",
            "SOIL HEALTHY",
            "No irrigation needed right now.",
        )

    return (
        *res,
        plan,
    )


# =========================================================
# ANOMALY DETECTION
# =========================================================

def anomalies(
    hist,
    live_source,
):
    """
    Simple rule-based moisture sensor
    fault detection.
    """

    m = hist["moisture"]

    out = []

    # Sudden change
    if (
        len(m) >= 2
        and abs(
            m[-1] - m[-2]
        ) > 20
    ):

        out.append(
            (
                "Sudden moisture change "
                f"{m[-1] - m[-2]:+.0f}% "
                "(possible sensor fault)"
            )
        )

    # Frozen sensor
    if (
        live_source
        and len(m) >= 10
        and (
            max(m[-10:])
            - min(m[-10:])
            < 0.01
        )
    ):

        out.append(
            "Moisture sensor value "
            "frozen for 10 readings"
        )

    return out