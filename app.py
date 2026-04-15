# ============================
#  HUMAN FACTORS MULTIMODAL
#  FULLY FIXED FLASK BACKEND
# ============================

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify
import joblib
import tensorflow as tf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import matplotlib
matplotlib.use('Agg')
# --------------------------------------
# CONFIGURATION
# --------------------------------------
MODEL_DIR = "model"
STATIC_DIR = "static"  # radar PNG folder

os.makedirs(STATIC_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "pilot_multimodal_model.h5")
G_SCALER_PATH = os.path.join(MODEL_DIR, "g_scaler.pkl")
C_SCALER_PATH = os.path.join(MODEL_DIR, "c_scaler.pkl")
PRED_HISTORY_CSV = os.path.join(MODEL_DIR, "predictions_history.csv")

# Init history CSV if not exists
if not os.path.exists(PRED_HISTORY_CSV):
    pd.DataFrame(columns=[
        "timestamp", "sample_id", "pred_class",
        "pred_label", "confidence", "hf_score"
    ]).to_csv(PRED_HISTORY_CSV, index=False)

# --------------------------------------
# LABELS & HUMAN FACTORS DEFINITIONS
# --------------------------------------
label_map = {
    0: "Low Awareness",
    1: "Medium Awareness",
    2: "High Awareness"
}

hf_explanation = {
    0: {
        "title": "Low Awareness",
        "description": "Pilot has reduced situational awareness.",
        "risk": "High risk — workload, fixation, and stress indicators present.",
        "factors": [
            "High cognitive load",
            "Attention tunneling",
            "Poor gaze distribution",
            "Erratic control inputs"
        ],
        "recommendations": [
            "Reduce task load",
            "Increase automation support",
            "Provide strong visual cues"
        ]
    },
    1: {
        "title": "Medium Awareness",
        "description": "Pilot has moderate situational awareness.",
        "risk": "Moderate risk — manageable workload.",
        "factors": [
            "Stable but effortful control",
            "Balanced attention"
        ],
        "recommendations": [
            "Provide soft warnings",
            "Maintain monitoring"
        ]
    },
    2: {
        "title": "High Awareness",
        "description": "Pilot has excellent situational awareness.",
        "risk": "Low risk — optimal performance.",
        "factors": [
            "Stable control",
            "Efficient gaze scan",
            "Low stress indicators"
        ],
        "recommendations": [
            "Continue monitoring"
        ]
    }
}

# --------------------------------------
# LOAD MODEL & SCALERS
# --------------------------------------
print("Loading model + scalers...")
model = tf.keras.models.load_model(MODEL_PATH)
g_scaler = joblib.load(G_SCALER_PATH)
c_scaler = joblib.load(C_SCALER_PATH)
print("Loaded successfully.")

app = Flask(__name__)

# --------------------------------------
# SUPPORT FUNCTIONS
# --------------------------------------

def compute_hf_score(class_id, conf):
    """Convert prediction + confidence into a human factor score."""
    if class_id == 2:
        return min(100, 70 + int(conf * 30))
    elif class_id == 1:
        return min(100, 40 + int(conf * 29))
    else:
        return max(0, int(39 * (1 - conf * 0.9)))

def risk_color(score):
    if score >= 70:
        return "green"
    elif score >= 40:
        return "gold"
    return "red"

def create_radar(hf_metrics, out_file):
    labels = list(hf_metrics.keys())
    values = list(hf_metrics.values())

    values += values[:1]  # close loop
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    ax.set_ylim(0, 100)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()

def write_history(row):
    df = pd.read_csv(PRED_HISTORY_CSV)
    df.loc[len(df)] = row
    df.to_csv(PRED_HISTORY_CSV, index=False)

# --------------------------------------
# PREDICTION LOGIC
# --------------------------------------
def predict_from_csvs(gaze_df, control_df):
    if gaze_df.shape != (20,4) or control_df.shape != (20,4):
        raise ValueError("CSV inputs must be exactly 20 rows × 4 columns.")

    # Scale inputs
    g = g_scaler.transform(gaze_df.values).reshape(1,20,4)
    c = c_scaler.transform(control_df.values).reshape(1,20,4)

    # Predict
    probs = model.predict([g, c])
    class_id = int(np.argmax(probs))
    conf = float(np.max(probs))

    score = compute_hf_score(class_id, conf)
    color = risk_color(score)
    hf = hf_explanation[class_id]

    # Radar metrics
    hf_metrics = {
        "Attention": score,
        "Workload": 100 - score,
        "Stability": score,
        "Scan": score,
        "Stress": 100 - score
    }

    # Save radar
    radar_file = f"radar_{datetime.utcnow().timestamp()}.png"
    radar_path = os.path.join(STATIC_DIR, radar_file)
    create_radar(hf_metrics, radar_path)

    # Log history
    write_history({
        "timestamp": datetime.utcnow().isoformat(),
        "sample_id": -1,
        "pred_class": class_id,
        "pred_label": label_map[class_id],
        "confidence": conf,
        "hf_score": score
    })

    return {
        "pred_label": label_map[class_id],
        "confidence": conf,
        "hf_score": score,
        "color": color,
        "hf": hf,
        "radar_file": radar_file,
        "hf_metrics": hf_metrics
    }

# --------------------------------------
# ROUTES
# --------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gaze_file = request.files.get("gaze_csv")
    control_file = request.files.get("control_csv")

    gaze_df = pd.read_csv(gaze_file)
    control_df = pd.read_csv(control_file)

    try:
        result = predict_from_csvs(gaze_df, control_df)
    except Exception as e:
        return f"Prediction error: {e}", 400

    return render_template("result_hf.html", pred=result)

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    """Generate and download the PDF report for the latest prediction."""
    df = pd.read_csv(PRED_HISTORY_CSV)
    last = df.iloc[-1]
    class_id = int(last["pred_class"])
    hf = hf_explanation[class_id]

    # Latest radar file
    radar_file = sorted(os.listdir(STATIC_DIR))[-1]
    radar_path = os.path.join(STATIC_DIR, radar_file)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 760, "Pilot Situation Awareness Report")

    y = 720
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Timestamp: {last['timestamp']}"); y -= 20
    c.drawString(50, y, f"Prediction: {last['pred_label']}"); y -= 20
    c.drawString(50, y, f"Confidence: {last['confidence']:.4f}"); y -= 20
    c.drawString(50, y, f"Human Factor Score: {last['hf_score']}"); y -= 30

    # Explanation
    c.drawString(50, y, hf["description"]); y -= 40

    # Radar
    c.drawImage(radar_path, 50, y-250, width=500, height=250)

    c.save()
    buf.seek(0)

    return send_file(buf, as_attachment=True,
                     download_name="prediction_report.pdf",
                     mimetype="application/pdf")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/dashboard/latest")
def dashboard_latest():
    df = pd.read_csv(PRED_HISTORY_CSV)
    return df.tail(50).to_json(orient="records")

# --------------------------------------
# RUN
# --------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
