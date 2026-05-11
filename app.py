import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_PATH = "model/gui_model.pkl"
SCALER_PATH = "model/gui_scaler.pkl"

FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaperlessBilling",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAPERLESS_MAP = {"Yes": 1, "No": 0}
CHURN_MAP = {"No": 0, "Yes": 1}


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #080c14;
        --panel: rgba(18, 24, 38, 0.88);
        --panel-soft: rgba(24, 32, 49, 0.72);
        --border: rgba(148, 163, 184, 0.18);
        --text: #e5eefb;
        --muted: #95a3b8;
        --green: #22c55e;
        --red: #ef4444;
        --cyan: #38bdf8;
        --amber: #f59e0b;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 18% 8%, rgba(56, 189, 248, 0.20), transparent 28%),
            radial-gradient(circle at 82% 16%, rgba(34, 197, 94, 0.14), transparent 30%),
            linear-gradient(135deg, #070b12 0%, #101827 52%, #07111c 100%);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(9, 14, 24, 0.98), rgba(16, 24, 39, 0.98));
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 1.5rem;
        max-width: 1320px;
    }

    .hero {
        padding: 2rem;
        border: 1px solid var(--border);
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(34, 197, 94, 0.10)),
            rgba(15, 23, 42, 0.78);
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.25rem;
    }

    .hero h1 {
        margin: 0 0 0.7rem 0;
        font-size: clamp(2.1rem, 5vw, 4.4rem);
        line-height: 1.03;
        font-weight: 800;
        letter-spacing: 0;
        color: #f8fbff;
    }

    .hero p {
        margin: 0;
        color: #b7c5d8;
        font-size: 1.02rem;
        max-width: 900px;
    }

    .glass-card {
        padding: 1.15rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--panel);
        box-shadow: 0 16px 45px rgba(0, 0, 0, 0.24);
        transition: transform 180ms ease, border-color 180ms ease, box-shadow 180ms ease;
        height: 100%;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.36);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.34);
    }

    .metric-card {
        padding: 1rem;
        border: 1px solid var(--border);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.72));
        min-height: 118px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        margin-top: 0.4rem;
    }

    .metric-note {
        color: #8fa1b8;
        font-size: 0.84rem;
        margin-top: 0.25rem;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0.3rem 0 0.8rem 0;
    }

    .status-card {
        padding: 1.35rem;
        border-radius: 20px;
        margin-top: 1rem;
        border: 1px solid transparent;
    }

    .status-churn {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18), rgba(127, 29, 29, 0.55));
        border-color: rgba(248, 113, 113, 0.34);
    }

    .status-stay {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.18), rgba(20, 83, 45, 0.54));
        border-color: rgba(74, 222, 128, 0.34);
    }

    .status-title {
        font-size: 1.55rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.4rem;
    }

    .status-subtitle {
        color: #cbd5e1;
        font-size: 0.98rem;
    }

    .sidebar-box {
        padding: 1rem;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid var(--border);
        margin-bottom: 0.8rem;
    }

    .sidebar-title {
        font-weight: 800;
        font-size: 1.12rem;
        margin-bottom: 0.6rem;
    }

    .sidebar-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        padding: 0.55rem 0;
        font-size: 0.9rem;
    }

    .sidebar-row span:first-child {
        color: #9fb0c7;
    }

    .sidebar-row span:last-child {
        color: #f8fafc;
        font-weight: 700;
        text-align: right;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #a9b7ca;
        font-weight: 700;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.28), rgba(34, 197, 94, 0.20));
        color: #ffffff;
    }

    .stButton > button {
        width: 100%;
        min-height: 3.25rem;
        border: 0;
        border-radius: 14px;
        background: linear-gradient(135deg, #38bdf8, #22c55e);
        color: #06121f;
        font-weight: 800;
        font-size: 1.04rem;
        box-shadow: 0 14px 35px rgba(34, 197, 94, 0.20);
        transition: transform 160ms ease, filter 160ms ease, box-shadow 160ms ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
        box-shadow: 0 18px 42px rgba(56, 189, 248, 0.24);
        color: #06121f;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
    }

    .footer {
        text-align: center;
        color: #8da0b8;
        padding: 1.25rem 0 0.5rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load trained GUI model and scaler."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_dataset():
    """Load raw Telco Customer Churn dataset."""
    return pd.read_csv(DATA_PATH)


def preprocess_for_gui(data):
    """Apply the same preprocessing used by the 5-feature GUI model."""
    prepared = data.copy()
    prepared["TotalCharges"] = pd.to_numeric(prepared["TotalCharges"], errors="coerce")
    prepared = prepared.dropna(subset=FEATURES + ["Churn"])

    prepared["Contract"] = prepared["Contract"].map(CONTRACT_MAP)
    prepared["PaperlessBilling"] = prepared["PaperlessBilling"].map(PAPERLESS_MAP)
    prepared["Churn"] = prepared["Churn"].map(CHURN_MAP)
    prepared = prepared.dropna(subset=FEATURES + ["Churn"])

    X = prepared[FEATURES].copy()
    y = prepared["Churn"].astype(int)
    return prepared, X, y


@st.cache_data
def calculate_dataset_summary(data):
    """Return high-level dataset statistics for dashboard cards."""
    churn_counts = data["Churn"].value_counts()
    return {
        "total_samples": int(len(data)),
        "churn_count": int(churn_counts.get("Yes", 0)),
        "non_churn_count": int(churn_counts.get("No", 0)),
        "feature_count": int(data.shape[1]),
    }


def evaluate_model(model, scaler, X, y):
    """Evaluate saved GUI model on the notebook-style holdout split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_test_scaled = X_test.copy()
    X_test_scaled[NUMERIC_FEATURES] = scaler.transform(X_test_scaled[NUMERIC_FEATURES])

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Non-Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix, len(X_test)


def render_metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(summary, accuracy):
    st.sidebar.markdown(
        """
        <div class="sidebar-box">
            <div class="sidebar-title">📌 Project Overview</div>
            <div class="sidebar-row"><span>Project Name</span><span>Customer Churn Prediction</span></div>
            <div class="sidebar-row"><span>Dataset Name</span><span>Telco Customer Churn</span></div>
            <div class="sidebar-row"><span>Model Used</span><span>Logistic Regression</span></div>
            <div class="sidebar-row"><span>Accuracy</span><span>{accuracy:.2%}</span></div>
        </div>
        """.format(accuracy=accuracy),
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-box">
            <div class="sidebar-title">🧠 ML Techniques</div>
            <div class="sidebar-row"><span>Missing Values</span><span>Handled</span></div>
            <div class="sidebar-row"><span>TotalCharges</span><span>Numeric</span></div>
            <div class="sidebar-row"><span>Encoding</span><span>Binary + Ordinal</span></div>
            <div class="sidebar-row"><span>Scaling</span><span>MinMaxScaler</span></div>
            <div class="sidebar-row"><span>Split</span><span>Train/Test</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="sidebar-box">
            <div class="sidebar-title">📊 Dataset Stats</div>
            <div class="sidebar-row"><span>Total Samples</span><span>{total:,}</span></div>
            <div class="sidebar-row"><span>Churn</span><span>{churn:,}</span></div>
            <div class="sidebar-row"><span>Non-Churn</span><span>{non_churn:,}</span></div>
        </div>
        """.format(
            total=summary["total_samples"],
            churn=summary["churn_count"],
            non_churn=summary["non_churn_count"],
        ),
        unsafe_allow_html=True,
    )


def build_prediction_frame(tenure, monthly_charges, total_charges, contract, paperless_billing):
    """Create one-row input DataFrame in the exact feature order expected by the model."""
    return pd.DataFrame(
        [[
            tenure,
            monthly_charges,
            total_charges,
            CONTRACT_MAP[contract],
            PAPERLESS_MAP[paperless_billing],
        ]],
        columns=FEATURES,
    )


def scale_input(input_df, scaler):
    """Scale only numerical columns with the saved GUI scaler."""
    scaled = input_df.copy()
    scaled[NUMERIC_FEATURES] = scaler.transform(scaled[NUMERIC_FEATURES])
    return scaled


def render_probability_bar(probability, prediction):
    color = "#ef4444" if prediction == 1 else "#22c55e"
    st.markdown(
        f"""
        <div style="margin-top: 1rem;">
            <div style="display:flex; justify-content:space-between; color:#cbd5e1; font-weight:700; margin-bottom:0.45rem;">
                <span>Prediction Probability</span>
                <span>{probability:.2%}</span>
            </div>
            <div style="height: 16px; border-radius: 999px; background: rgba(148, 163, 184, 0.18); overflow: hidden;">
                <div style="height: 16px; width: {probability * 100:.2f}%; background: {color}; border-radius: 999px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-Churn", "Churn"],
        yticklabels=["Non-Churn", "Churn"],
        linewidths=0.7,
        linecolor="#1e293b",
        ax=ax,
    )
    ax.set_xlabel("Predicted", color="#e5e7eb", fontweight="bold")
    ax.set_ylabel("Actual", color="#e5e7eb", fontweight="bold")
    ax.set_title("Confusion Matrix", color="#f8fafc", fontweight="bold", pad=14)
    ax.tick_params(colors="#cbd5e1")
    st.pyplot(fig, use_container_width=True)


def main():
    try:
        model, scaler = load_artifacts()
        raw_data = load_dataset()
    except FileNotFoundError as error:
        st.error(f"Required file not found: {error}")
        st.stop()
    except Exception as error:
        st.error(f"Unable to load model, scaler, or dataset: {error}")
        st.stop()

    prepared_data, X, y = preprocess_for_gui(raw_data)
    summary = calculate_dataset_summary(raw_data)
    accuracy, report, matrix, test_samples = evaluate_model(model, scaler, X, y)
    report_df = pd.DataFrame(report).transpose().round(3)

    render_sidebar(summary, accuracy)

    st.markdown(
        """
        <div class="hero">
            <h1>Customer Churn Prediction System</h1>
            <p>
                Professional machine learning dashboard for predicting whether a Telco customer is likely to leave the service,
                using a 5-feature Logistic Regression GUI model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    with metric_col_1:
        render_metric_card("Model Accuracy", f"{accuracy:.2%}", f"Evaluated on {test_samples:,} test samples")
    with metric_col_2:
        render_metric_card("Total Samples", f"{summary['total_samples']:,}", "Telco Customer Churn dataset")
    with metric_col_3:
        render_metric_card("Churn Customers", f"{summary['churn_count']:,}", "Customers who left")
    with metric_col_4:
        render_metric_card("Non-Churn Customers", f"{summary['non_churn_count']:,}", "Customers who stayed")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_prediction, tab_model, tab_dataset = st.tabs(["🔮 Prediction", "📈 Model Info", "🗂️ Dataset Info"])

    with tab_prediction:
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown('<div class="section-title">Customer Input Form</div>', unsafe_allow_html=True)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            with st.form("prediction_form"):
                form_col_1, form_col_2 = st.columns(2)
                with form_col_1:
                    tenure = st.number_input(
                        "Tenure",
                        min_value=0,
                        max_value=100,
                        value=12,
                        step=1,
                        help="Number of months the customer has stayed with the company.",
                    )
                    total_charges = st.number_input(
                        "TotalCharges",
                        min_value=0.0,
                        value=500.0,
                        step=10.0,
                        format="%.2f",
                    )
                    paperless_billing = st.selectbox("PaperlessBilling", ["Yes", "No"])

                with form_col_2:
                    monthly_charges = st.number_input(
                        "MonthlyCharges",
                        min_value=0.0,
                        value=70.0,
                        step=1.0,
                        format="%.2f",
                    )
                    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

                submitted = st.form_submit_button("Predict Churn")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="glass-card">
                    <div style="color:#95a3b8; font-weight:600; margin-bottom:0.5rem;">Current Model</div>
                    <div style="font-size:1.6rem; font-weight:800; color:#f8fafc;">Logistic Regression</div>
                    <div style="color:#a9b7ca; margin-top:0.45rem;">
                        Features: tenure, MonthlyCharges, TotalCharges, Contract, PaperlessBilling
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if submitted:
                with st.spinner("Analyzing customer churn risk..."):
                    progress = st.progress(0)
                    for value in range(0, 101, 20):
                        time.sleep(0.06)
                        progress.progress(value)

                input_df = build_prediction_frame(
                    tenure,
                    monthly_charges,
                    total_charges,
                    contract,
                    paperless_billing,
                )
                scaled_input = scale_input(input_df, scaler)
                prediction = int(model.predict(scaled_input)[0])
                probabilities = model.predict_proba(scaled_input)[0]
                probability = float(probabilities[prediction])

                if prediction == 1:
                    st.markdown(
                        """
                        <div class="status-card status-churn">
                            <div class="status-title">⚠️ Customer likely to churn</div>
                            <div class="status-subtitle">Risk status: High attention recommended.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="status-card status-stay">
                            <div class="status-title">✅ Customer likely to stay</div>
                            <div class="status-subtitle">Risk status: Customer appears stable.</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                render_probability_bar(probability, prediction)

                prob_col_1, prob_col_2 = st.columns(2)
                with prob_col_1:
                    st.metric("Stay Probability", f"{probabilities[0]:.2%}")
                with prob_col_2:
                    st.metric("Churn Probability", f"{probabilities[1]:.2%}")
            else:
                st.info("Enter customer information and click Predict Churn to view the result.")

    with tab_model:
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        model_col_1, model_col_2 = st.columns([0.95, 1.05], gap="large")

        with model_col_1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{accuracy:.2%}")
            st.metric("Test Samples", f"{test_samples:,}")
            st.metric("Model Type", "Logistic Regression")
            st.metric("Scaler", "MinMaxScaler")
            st.markdown("</div>", unsafe_allow_html=True)

        with model_col_2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            render_confusion_matrix(matrix)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
        st.dataframe(report_df, use_container_width=True)

    with tab_dataset:
        st.markdown('<div class="section-title">Dataset Information</div>', unsafe_allow_html=True)
        data_col_1, data_col_2, data_col_3 = st.columns(3)
        with data_col_1:
            st.metric("Total Samples", f"{summary['total_samples']:,}")
        with data_col_2:
            st.metric("Cleaned Samples", f"{len(prepared_data):,}")
        with data_col_3:
            st.metric("Columns", f"{summary['feature_count']}")

        chart_col_1, chart_col_2 = st.columns([0.8, 1.2], gap="large")
        with chart_col_1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            churn_summary = pd.DataFrame(
                {
                    "Status": ["Non-Churn", "Churn"],
                    "Count": [summary["non_churn_count"], summary["churn_count"]],
                }
            )
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            sns.barplot(data=churn_summary, x="Status", y="Count", palette=["#22c55e", "#ef4444"], ax=ax)
            ax.set_title("Churn Distribution", color="#f8fafc", fontweight="bold", pad=14)
            ax.set_xlabel("")
            ax.set_ylabel("Customers", color="#cbd5e1")
            ax.tick_params(colors="#cbd5e1")
            sns.despine(ax=ax)
            st.pyplot(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col_2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.dataframe(raw_data.head(12), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">GUI Features</div>', unsafe_allow_html=True)
        st.dataframe(prepared_data[FEATURES + ["Churn"]].head(20), use_container_width=True)

    st.markdown('<div class="footer">Machine Learning Final Project</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
