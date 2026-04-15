# app.py
# -------------------------
# Human Factors Multimodal - Flask with Auth, Admin, Metrics
# -------------------------

import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, flash, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import tensorflow as tf
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


# -------------------------
# Configuration
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
DB_PATH = os.path.join(BASE_DIR, "app.db")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Flask
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = os.environ.get("FLASK_SECRET", "very-secret-key-change-me")
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + DB_PATH
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database + Login
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# Prediction history CSV (keeps original behavior)
PRED_HISTORY_CSV = os.path.join(MODEL_DIR, "predictions_history.csv")
if not os.path.exists(PRED_HISTORY_CSV):
    pd.DataFrame(columns=[
        "timestamp", "sample_id", "pred_class",
        "pred_label", "confidence", "hf_score"
    ]).to_csv(PRED_HISTORY_CSV, index=False)

# Model / scalers (attempt to load, but app still runs if absent)
MODEL_PATH = os.path.join(MODEL_DIR, "pilot_multimodal_model.h5")
G_SCALER_PATH = os.path.join(MODEL_DIR, "g_scaler.pkl")
C_SCALER_PATH = os.path.join(MODEL_DIR, "c_scaler.pkl")

model = None
g_scaler = None
c_scaler = None

def try_load_model_and_scalers():
    global model, g_scaler, c_scaler
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
        if os.path.exists(G_SCALER_PATH):
            g_scaler = joblib.load(G_SCALER_PATH)
        if os.path.exists(C_SCALER_PATH):
            c_scaler = joblib.load(C_SCALER_PATH)
        print("Model/scalers loaded (if present).")
    except Exception as e:
        print("Warning: failed to load model/scalers:", e)

try_load_model_and_scalers()

# -------------------------
# Label map & HF explanations
# -------------------------
label_map = {0: "Low Awareness", 1: "Medium Awareness", 2: "High Awareness"}

hf_explanation = {
    0: {
        "title": "Low Awareness",
        "description": "Pilot shows low situational awareness — signs of task saturation, attention tunneling, or erratic control inputs.",
        "risk": "High risk — immediate intervention recommended.",
        "factors": [
            "Sustained attention tunneling (narrow gaze distribution)",
            "High control input variability",
            "Delayed or missed checklists/actions",
            "Elevated pupil dilation / blink rate (if measured)",
            "Frequent corrective maneuvers"
        ],
        "recommendations": [
            "Reduce cognitive load: postpone non-essential tasks and alerts.",
            "Enable or increase automation assistance (autopilot, envelope protection).",
            "Provide concise, high-salience visual and auditory cues for critical parameters.",
            "Request brief pilot rest / breathing break if operationally permissible.",
            "Assign a co-pilot or supervisor to assist immediate tasks."
        ]
    },

    1: {
        "title": "Medium Awareness",
        "description": "Pilot demonstrates moderate situational awareness: performance is acceptable but requires monitoring and occasional support.",
        "risk": "Moderate risk — maintain supervision and soft interventions.",
        "factors": [
            "Stable overall control with intermittent variability",
            "Occasional lapses in scan or delayed responses",
            "Higher-than-normal workload during complex segments",
            "Periods of effective automation use mixed with manual corrections"
        ],
        "recommendations": [
            "Provide targeted prompts to sustain effective scan patterns (short reminders).",
            "Reduce task switching and avoid introducing new complex tasks.",
            "Keep automation modes clearly displayed and confirm pilot understanding.",
            "Use soft warnings rather than hard alerts to avoid overloading the pilot.",
            "Schedule brief check-ins or crew resource management prompts."
        ]
    },

    2: {
        "title": "High Awareness",
        "description": "Pilot exhibits high situational awareness and stable control. Performance metrics indicate efficient workload handling and scan behavior.",
        "risk": "Low risk — continue monitoring and log for training/benchmarking.",
        "factors": [
            "Consistent gaze distribution and systematic scan pattern",
            "Low control input volatility",
            "Timely task completion and appropriate automation use",
            "Low stress indicators and stable physiological signs (if available)"
        ],
        "recommendations": [
            "Maintain current procedures and monitor for deviations.",
            "Record scenario for later review and use as a positive training case.",
            "Avoid unnecessary interventions that could disturb current flow.",
            "Continue periodic monitoring to ensure sustained performance."
        ]
    }
}

# -------------------------
# Auth models
# -------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(300), nullable=False)

    is_admin = db.Column(db.Boolean, default=False)
    is_active_flag = db.Column(db.Boolean, default=True)  # admin approval flag

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    # Flask-Login required method
    def is_active(self):
        return self.is_active_flag
    
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# create DB if not exists
with app.app_context():
    db.create_all()
    # create default admin if none
    if User.query.filter_by(is_admin=True).first() is None:
        admin = User(username="admin", email="admin@example.com", is_admin=True)
        admin.set_password("adminpass")
        db.session.add(admin)
        db.session.commit()
        print("Created default admin admin@example.com / adminpass (change asap)")

# -------------------------
# Helper functions
# -------------------------
def compute_hf_score(class_id, conf):
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
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    ax.set_ylim(0, 100)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

def write_history(row):
    df = pd.read_csv(PRED_HISTORY_CSV)
    df.loc[len(df)] = row
    df.to_csv(PRED_HISTORY_CSV, index=False)

# -------------------------
# Prediction logic (uses loaded model/scalers if available)
# -------------------------
def predict_from_csvs(gaze_df, control_df):
    # Basic validation (same as original)
    if gaze_df.shape != (20,4) or control_df.shape != (20,4):
        raise ValueError("CSV inputs must be exactly 20 rows × 4 columns.")

    # ----------------------------------------------------
    # MODEL PREDICTION / FALLBACK
    # ----------------------------------------------------
    if model is None or g_scaler is None or c_scaler is None:
        # fallback: simple heuristic
        avg_g = gaze_df.values.mean()
        avg_c = control_df.values.mean()

        score = int(max(0, min(100, (50 + (avg_g - avg_c) * 10))))

        if score >= 70:
            class_id, conf = 2, 0.92
        elif score >= 40:
            class_id, conf = 1, 0.67
        else:
            class_id, conf = 0, 0.83

    else:
        g = g_scaler.transform(gaze_df.values).reshape(1,20,4)
        c = c_scaler.transform(control_df.values).reshape(1,20,4)

        probs = model.predict([g, c])
        class_id = int(np.argmax(probs))
        conf = float(np.max(probs))

        score = compute_hf_score(class_id, conf)

    # ----------------------------------------------------
    # HUMAN FACTOR METRICS
    # ----------------------------------------------------
    color = risk_color(score)
    base_info = hf_explanation[class_id]

    hf_metrics = {
        "Attention": score,
        "Workload": 100 - score,
        "Stability": score,
        "Scan": score,
        "Stress": 100 - score
    }

    # ----------------------------------------------------
    # CUSTOM DYNAMIC RECOMMENDATIONS BLOCK
    # ----------------------------------------------------
    dynamic_suggestions = []

    # Low awareness → more warnings
    if class_id == 0:
        dynamic_suggestions.extend([
            "Critical: Pilot workload too high; immediate assistance recommended.",
            "Consider reducing cockpit task load and enabling higher automation.",
            "Increase visual/auditory cues to regain situational awareness.",
            "Encourage short breathing cycle to lower cognitive overload."
        ])

    # Medium awareness
    elif class_id == 1:
        dynamic_suggestions.extend([
            "Pilot awareness is stable but requires monitoring.",
            "Introduce periodic prompts to maintain task engagement.",
            "Avoid unnecessary complex tasks during this phase."
        ])

    # High awareness
    else:
        dynamic_suggestions.extend([
            "Pilot is performing optimally.",
            "Maintain current workflow; no intervention required.",
            "Regular monitoring recommended for consistency."
        ])

    # Extra dynamic suggestions based on data
    if hf_metrics["Workload"] > 60:
        dynamic_suggestions.append("Detected high workload from control variability ― consider automation support.")

    if hf_metrics["Stress"] > 60:
        dynamic_suggestions.append("Stress indicators elevated ― offer guidance or reduce task density.")

    if hf_metrics["Scan"] < 50:
        dynamic_suggestions.append("Gaze scan inconsistency detected ― recommend refocusing strategies.")

    # ----------------------------------------------------
    # Generate radar chart
    # ----------------------------------------------------
    radar_file = f"radar_{int(datetime.utcnow().timestamp())}.png"
    radar_path = os.path.join(STATIC_DIR, radar_file)
    create_radar(hf_metrics, radar_path)

    # ----------------------------------------------------
    # Save to history CSV
    # ----------------------------------------------------
    write_history({
        "timestamp": datetime.utcnow().isoformat(),
        "sample_id": -1,
        "pred_class": class_id,
        "pred_label": label_map[class_id],
        "confidence": conf,
        "hf_score": score
    })

    # ----------------------------------------------------
    # FINAL RETURN STRUCTURE (NEW IMPROVED VERSION)
    # ----------------------------------------------------
    return {
        "pred_label": label_map[class_id],
        "confidence": conf,
        "hf_score": score,
        "color": color,
        "hf": base_info,
        "radar_file": radar_file,
        "hf_metrics": hf_metrics,
        "dynamic_suggestions": dynamic_suggestions,
        "awareness_level": class_id
    }

# -------------------------
# Routes: public pages
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_page")
@login_required
def predict_page():
    return render_template("predict_page.html")

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    gaze_file = request.files.get("gaze_csv")
    control_file = request.files.get("control_csv")

    if not gaze_file or not control_file:
        flash("Please upload both gaze and control CSV files.", "danger")
        return redirect(url_for("predict_page"))

    try:
        gaze_df = pd.read_csv(gaze_file)
        control_df = pd.read_csv(control_file)
    except Exception as e:
        flash("Failed to read CSV files: " + str(e), "danger")
        return redirect(url_for("predict_page"))

    try:
        result = predict_from_csvs(gaze_df, control_df)
    except Exception as e:
        flash("Prediction error: " + str(e), "danger")
        return redirect(url_for("predict_page"))

    return render_template("result_hf.html", pred=result)

@app.route("/download_pdf", methods=["POST"])
@login_required
def download_pdf():
    df = pd.read_csv(PRED_HISTORY_CSV)
    last = df.iloc[-1]
    class_id = int(last["pred_class"])
    hf = hf_explanation[class_id]

    radar_file = sorted(os.listdir(STATIC_DIR))[-1]
    radar_path = os.path.join(STATIC_DIR, radar_file)

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 760, "Pilot Situation Awareness Report")
    y = 720
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Timestamp: {last['timestamp']}"); y -= 20
    c.drawString(50, y, f"Prediction: {last['pred_label']}"); y -= 20
    c.drawString(50, y, f"Confidence: {last['confidence']:.4f}"); y -= 20
    c.drawString(50, y, f"Human Factor Score: {last['hf_score']}"); y -= 30
    c.drawString(50, y, hf["description"]); y -= 40
    c.drawImage(radar_path, 50, y-250, width=500, height=250)
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name="prediction_report.pdf",
                     mimetype="application/pdf")

# -------------------------
# Authentication routes
# -------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        if not (username and email and password):
            flash("All fields required.", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "warning")
            return redirect(url_for("register"))
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Account created. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if not user:
            flash("User does not exist.", "danger")
            return redirect(url_for("login"))

        if not user.is_active_flag:
            flash("Your account is blocked. Contact admin.", "danger")
            return redirect(url_for("login"))

        if user.check_password(password):
            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for("index"))
        else:
            flash("Incorrect password.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("index"))

# -------------------------
# Admin utilities
# -------------------------
def admin_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not getattr(current_user, "is_admin", False):
            flash("Admin access required.", "danger")
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    return wrapper

@app.route("/admin_dashboard")
@admin_required
def admin_dashboard():
    # show some quick stats
    df = pd.read_csv(PRED_HISTORY_CSV)
    total = len(df)
    last = df.tail(1).to_dict(orient="records")
    return render_template("admin_dashboard.html", total=total, last=last)

@app.route("/admin_history")
@admin_required
def admin_history():
    df = pd.read_csv(PRED_HISTORY_CSV)
    history_json = df.tail(200).to_json(orient="records")
    return render_template("admin_history.html", history_json=history_json)

# --------------------------------------------------------
# ADMIN ACCESS DECORATOR
# --------------------------------------------------------
def admin_required(func):
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for("login"))

        if not getattr(current_user, "is_admin", False):
            flash("Admin access required!", "danger")
            return redirect(url_for("index"))

        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


# --------------------------------------------------------
# USER DASHBOARD
# --------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/dashboard/latest")
@login_required
def dashboard_latest():
    df = pd.read_csv(PRED_HISTORY_CSV)
    return df.tail(50).to_json(orient="records")


# --------------------------------------------------------
# ADMIN USER MANAGEMENT
# --------------------------------------------------------
@app.route("/admin/users")
@admin_required
def admin_users():
    users = User.query.all()
    return render_template("admin_users.html", users=users)


@app.route("/admin/user/approve/<int:user_id>")
@admin_required
def admin_approve_user(user_id):
    user = User.query.get_or_404(user_id)
    user.is_active_flag = True
    db.session.commit()
    flash("User approved.", "success")
    return redirect(url_for("admin_users"))


@app.route("/admin/user/block/<int:user_id>")
@admin_required
def admin_block_user(user_id):
    user = User.query.get_or_404(user_id)
    user.is_active_flag = False
    db.session.commit()
    flash("User blocked.", "warning")
    return redirect(url_for("admin_users"))


@app.route("/admin/user/delete/<int:user_id>")
@admin_required
def admin_delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash("User deleted.", "danger")
    return redirect(url_for("admin_users"))


# --------------------------------------------------------
# ADMIN METRICS DASHBOARD (ML METRICS + MODEL STATS)
# --------------------------------------------------------
@app.route("/admin/metrics")
@admin_required
def admin_metrics():
    confusion_img = "metrics/confusion_matrix.png"
    loss_img = "metrics/training_testing_plots.png"

    report_path = os.path.join(STATIC_DIR, "metrics", "classification_report.txt")
    cls_table = []
    accuracy = None
    labels = []
    supports = []

    # Parse classification report if available
    if os.path.exists(report_path):
        txt = open(report_path, "r").read().strip()

        for line in txt.splitlines():
            line = line.strip()

            # class rows: <class> <precision> <recall> <f1> <support>
            if len(line) > 0 and line[0].isdigit():
                parts = line.split()
                try:
                    cls = parts[0]
                    precision = float(parts[1])
                    recall = float(parts[2])
                    f1 = float(parts[3])
                    support = int(parts[4])
                    cls_table.append({
                        "class": cls,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "support": support
                    })
                    labels.append(cls)
                    supports.append(support)
                except:
                    continue

            # accuracy
            if line.lower().startswith("accuracy"):
                parts = line.split()
                for p in parts:
                    try:
                        val = float(p)
                        if 0 <= val <= 1:
                            accuracy = val
                            break
                    except:
                        continue

        if not cls_table:
            cls_raw = open(report_path, "r").read()
        else:
            cls_raw = None

    else:
        cls_raw = None

    # Confusion matrix (fallback = diagonal)
    if supports:
        matrix = [[0]*len(supports) for _ in supports]
        for i, s in enumerate(supports):
            matrix[i][i] = s
    else:
        matrix = None

    if accuracy is None and supports:
        total = sum(supports)
        accuracy = sum(supports) / total if total > 0 else 0.0

    # Model summary (if TF/Keras model exists)
    try:
        buf = io.StringIO()
        model.summary(print_fn=lambda x: buf.write(x + "\n"))
        model_summary = buf.getvalue()
        buf.close()
    except Exception:
        model_summary = None

    return render_template(
        "admin_metrics.html",
        available=True,
        confusion_image=confusion_img,
        loss_image=loss_img,
        cls_table=cls_table,
        cls_raw=cls_raw,
        matrix=matrix,
        labels=labels,
        accuracy=accuracy,
        model_summary=model_summary
    )


# --------------------------------------------------------
# ADMIN METRICS → CHART.JS DATA (PIE/DONUT/BAR/LINE)
# --------------------------------------------------------
@app.route("/admin/metrics/data")
@admin_required
def metrics_data():

    total_users = User.query.count()
    approved_users = User.query.filter_by(approved=True).count()
    blocked_users = User.query.filter_by(approved=False).count()

    total_predictions = Prediction.query.count()
    high_risk = Prediction.query.filter_by(pred_class=0).count()
    medium_risk = Prediction.query.filter_by(pred_class=1).count()
    low_risk = Prediction.query.filter_by(pred_class=2).count()

    # HF Score ranges
    range_low = Prediction.query.filter(Prediction.hf_score <= 30).count()
    range_mid = Prediction.query.filter(Prediction.hf_score.between(31, 60)).count()
    range_high = Prediction.query.filter(Prediction.hf_score >= 61).count()

    # Timeline data (last 30 predictions)
    recent_preds = Prediction.query.order_by(Prediction.timestamp.desc()).limit(30).all()
    time_labels = [p.timestamp.strftime("%H:%M") for p in reversed(recent_preds)]
    time_values = [1 for _ in time_labels]

    return {
        "total_users": total_users,
        "approved_users": approved_users,
        "total_predictions": total_predictions,
        "high_risk": high_risk,

        "class_counts": [low_risk, medium_risk, high_risk],

        "hf_ranges": [range_low, range_mid, range_high],

        "user_status": [approved_users, blocked_users],

        "time_labels": time_labels,
        "time_values": time_values
    }


# --------------------------------------------------------
# ADMIN DATASET PAGE
# --------------------------------------------------------
@app.route("/admin/dataset")
@admin_required
def admin_dataset():
    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".csv")]
    return render_template("admin_dataset.html", files=files)
@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/contactus")
def contactus():
    return render_template("contactus.html")
# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
