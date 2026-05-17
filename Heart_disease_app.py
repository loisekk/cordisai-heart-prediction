import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CordisAI — Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global / background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark teal mesh background */
.stApp {
    background:
        radial-gradient(ellipse at 20% 20%, rgba(0,180,140,0.12) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 80%, rgba(0,100,180,0.10) 0%, transparent 60%),
        linear-gradient(135deg, #0a1628 0%, #0d1f2d 50%, #071a20 100%);
    background-attachment: fixed;
}

/* Grid overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,200,150,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,150,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 22, 40, 0.95) !important;
    border-right: 1px solid rgba(0,200,150,0.15);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00c896 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(0,200,150,0.07);
    border: 1px solid rgba(0,200,150,0.2);
    border-radius: 12px;
    padding: 14px 18px !important;
    backdrop-filter: blur(8px);
}
[data-testid="metric-container"] label { color: #8ab4c8 !important; font-size: 0.78rem !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #e8f5ff !important; font-size: 1.6rem !important; }

/* ── Cards / containers ── */
.card {
    background: rgba(13,31,45,0.85);
    border: 1px solid rgba(0,200,150,0.18);
    border-radius: 16px;
    padding: 24px 28px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}
.card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #00c896;
    margin-bottom: 6px;
    letter-spacing: 0.02em;
}

/* ── Prediction result boxes ── */
.result-high {
    background: linear-gradient(135deg, rgba(220,38,38,0.18), rgba(239,68,68,0.08));
    border: 1.5px solid rgba(220,38,38,0.55);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, rgba(0,200,150,0.18), rgba(16,185,129,0.08));
    border: 1.5px solid rgba(0,200,150,0.55);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
}
.result-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 8px; }
.result-sub   { font-size: 0.92rem; color: #8ab4c8; }

/* ── Inputs ── */
[data-testid="stSlider"] > div > div > div > div { background: #00c896 !important; }
.stSelectbox > div > div { background: rgba(13,31,45,0.9) !important; border-color: rgba(0,200,150,0.25) !important; }
[data-testid="stNumberInput"] > div > div > input { background: rgba(13,31,45,0.9) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: rgba(10,22,40,0.7); border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #8ab4c8; border-radius: 8px; }
.stTabs [aria-selected="true"] { background: rgba(0,200,150,0.2) !important; color: #00c896 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00c896, #00a07c) !important;
    color: #071a20 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 32px !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0,200,150,0.35) !important; }

/* ── Dividers ── */
hr { border-color: rgba(0,200,150,0.15) !important; }

/* ── Headings ── */
h1, h2, h3 { color: #e8f5ff !important; }
p, li, span { color: #b0cfe0; }

/* ── Risk gauge labels ── */
.gauge-label { font-size: 0.82rem; color: #8ab4c8; text-align: center; margin-top: 4px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a1628; }
::-webkit-scrollbar-thumb { background: #00c896; border-radius: 3px; }

/* ── Section headers ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.15rem; font-weight: 600; color: #e8f5ff;
    margin-bottom: 16px;
}
.section-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #00c896;
    box-shadow: 0 0 8px #00c896;
    flex-shrink: 0;
}

/* alert banner */
.info-banner {
    background: rgba(0,200,150,0.1);
    border-left: 3px solid #00c896;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #8ab4c8;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("KNN_Heart_Prediction.pkl")
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

@st.cache_data
def load_data():
    df = pd.read_csv("heart.xls")
    ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
    df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean).round(2)
    bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
    df['RestingBP']   = df['RestingBP'].replace(0, bp_mean).round(2)
    return df

model, scaler, MODEL_COLS = load_model()
df = load_data()

NUMERICAL = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

PLOT_CFG = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#8ab4c8', family='Inter'),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─── Helper: build input vector ──────────────────────────────────────────────
def build_input(inputs: dict) -> pd.DataFrame:
    row = {c: 0 for c in MODEL_COLS}
    row['Age']        = inputs['Age']
    row['RestingBP']  = inputs['RestingBP']
    row['Cholesterol']= inputs['Cholesterol']
    row['FastingBS']  = inputs['FastingBS']
    row['MaxHR']      = inputs['MaxHR']
    row['Oldpeak']    = inputs['Oldpeak']
    if inputs['Sex'] == 'M':        row['Sex_M'] = 1
    cpt = inputs['ChestPainType']
    if cpt == 'ATA': row['ChestPainType_ATA'] = 1
    elif cpt == 'NAP': row['ChestPainType_NAP'] = 1
    elif cpt == 'TA':  row['ChestPainType_TA']  = 1
    recg = inputs['RestingECG']
    if recg == 'Normal': row['RestingECG_Normal'] = 1
    elif recg == 'ST':   row['RestingECG_ST']    = 1
    if inputs['ExerciseAngina'] == 'Y': row['ExerciseAngina_Y'] = 1
    sl = inputs['ST_Slope']
    if sl == 'Flat': row['ST_Slope_Flat'] = 1
    elif sl == 'Up': row['ST_Slope_Up']   = 1
    return pd.DataFrame([row])

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 20px 0;'>
      <div style='font-size:2.4rem;'>🫀</div>
      <div style='font-size:1.25rem; font-weight:700; color:#e8f5ff; letter-spacing:0.04em;'>CordisAI</div>
      <div style='font-size:0.78rem; color:#8ab4c8; margin-top:4px;'>Heart Disease Risk Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='card-title'>🧬 Patient Information</div>", unsafe_allow_html=True)

    # Demographics
    st.markdown("**Demographics**")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50, step=1)
    with col_s2:
        sex = st.selectbox("Sex", ["M", "F"])

    st.markdown("**Vitals**")
    resting_bp  = st.slider("Resting BP (mmHg)", 80, 200, 120, 1)
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200, 1)
    max_hr      = st.slider("Max Heart Rate", 60, 220, 150, 1)
    oldpeak     = st.slider("Oldpeak (ST depression)", 0.0, 6.5, 1.0, 0.1)

    st.markdown("**Clinical Findings**")
    chest_pain    = st.selectbox("Chest Pain Type",
        ["ASY","ATA","NAP","TA"],
        help="ASY=Asymptomatic · ATA=Atypical Angina · NAP=Non-Anginal · TA=Typical Angina")
    fasting_bs    = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x else "No")
    resting_ecg   = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
    exercise_ang  = st.selectbox("Exercise-Induced Angina", ["N","Y"], format_func=lambda x: "Yes" if x=="Y" else "No")
    st_slope      = st.selectbox("ST Slope", ["Up","Flat","Down"])

    st.markdown("---")
    predict_btn = st.button("🔍  Analyze Risk", use_container_width=True)

# ─── Assemble patient dict ────────────────────────────────────────────────────
patient = dict(
    Age=age, Sex=sex, ChestPainType=chest_pain,
    RestingBP=resting_bp, Cholesterol=cholesterol,
    FastingBS=fasting_bs, RestingECG=resting_ecg,
    MaxHR=max_hr, ExerciseAngina=exercise_ang,
    Oldpeak=oldpeak, ST_Slope=st_slope,
)

# ─── Main Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 10px 0 6px 0;'>
  <div style='font-size:0.78rem; letter-spacing:0.15em; color:#00c896; font-weight:600; margin-bottom:6px;'>
    CLINICAL DECISION SUPPORT
  </div>
  <h1 style='font-size:2.2rem; font-weight:700; margin:0; color:#e8f5ff;'>
    Cardiovascular Risk Intelligence by CordisAI
  </h1>
  <p style='color:#8ab4c8; margin-top:6px; font-size:0.95rem;'>
    KNN-powered prediction trained on 918 patients · 86.4% accuracy · 88.2% F1
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Top KPI row ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Dataset Size",   "918 patients")
k2.metric("Model Accuracy", "86.4%")
k3.metric("F1 Score",       "88.2%")
k4.metric("Features Used",  "15")
k5.metric("Algorithm",      "KNN (k=5)")

st.markdown("---")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Risk Prediction",
    "📊  Dataset Insights",
    "📈  Model Performance",
    "ℹ️  Feature Reference",
])

# ══════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════
with tab1:

    if predict_btn or True:          # always show layout; update on click
        X_input = build_input(patient)
        X_scaled = scaler.transform(X_input[MODEL_COLS])

        prediction = model.predict(X_scaled)[0]
        proba      = model.predict_proba(X_scaled)[0]
        risk_pct   = proba[1] * 100

        col_res, col_gauge = st.columns([1.4, 1])

        with col_res:
            if prediction == 1:
                st.markdown(f"""
                <div class='result-high'>
                  <div class='result-title' style='color:#f87171;'>⚠️ High Risk Detected</div>
                  <div style='font-size:3rem; font-weight:800; color:#ef4444; margin:8px 0;'>{risk_pct:.1f}%</div>
                  <div class='result-sub'>Probability of Heart Disease</div>
                  <div style='margin-top:14px; font-size:0.88rem; color:#f87171; background:rgba(239,68,68,0.12);
                       border-radius:8px; padding:10px 14px;'>
                    Clinical review recommended. This patient shows elevated cardiovascular risk markers.
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-low'>
                  <div class='result-title' style='color:#34d399;'>✅ Low Risk Profile</div>
                  <div style='font-size:3rem; font-weight:800; color:#00c896; margin:8px 0;'>{risk_pct:.1f}%</div>
                  <div class='result-sub'>Probability of Heart Disease</div>
                  <div style='margin-top:14px; font-size:0.88rem; color:#34d399; background:rgba(0,200,150,0.1);
                       border-radius:8px; padding:10px 14px;'>
                    No immediate concern. Routine monitoring and healthy lifestyle advised.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Risk breakdown bar
            st.markdown("<br>", unsafe_allow_html=True)
            fig_bar = go.Figure(go.Bar(
                x=[proba[0]*100, proba[1]*100],
                y=["No Disease", "Heart Disease"],
                orientation='h',
                marker_color=['#00c896', '#ef4444'],
                text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                textposition='inside',
            ))
            fig_bar.update_layout(**PLOT_CFG, height=120,
                xaxis=dict(range=[0,100], showgrid=False, visible=False),
                yaxis=dict(showgrid=False),
                title=dict(text="Class Probabilities", font=dict(size=13, color='#8ab4c8')),
                bargap=0.35,
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

        with col_gauge:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                number=dict(suffix="%", font=dict(size=36, color='#e8f5ff')),
                delta=dict(reference=50, valueformat=".1f"),
                gauge=dict(
                    axis=dict(range=[0,100], tickwidth=1, tickcolor='#8ab4c8',
                              tickfont=dict(color='#8ab4c8')),
                    bar=dict(color='#ef4444' if prediction==1 else '#00c896', thickness=0.28),
                    bgcolor='rgba(13,31,45,0.8)',
                    borderwidth=0,
                    steps=[
                        dict(range=[0,30],  color='rgba(0,200,150,0.15)'),
                        dict(range=[30,60], color='rgba(251,191,36,0.12)'),
                        dict(range=[60,100],color='rgba(239,68,68,0.15)'),
                    ],
                    threshold=dict(line=dict(color='white', width=2), thickness=0.8, value=50),
                ),
                title=dict(text="Risk Score", font=dict(size=14, color='#8ab4c8')),
            ))
            fig_gauge.update_layout(**PLOT_CFG, height=270)
            st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

            # Risk tier
            tier = "🔴 Critical" if risk_pct>70 else ("🟡 Moderate" if risk_pct>40 else "🟢 Low")
            st.markdown(f"""
            <div style='text-align:center; background:rgba(13,31,45,0.6); border:1px solid rgba(0,200,150,0.15);
                 border-radius:10px; padding:10px;'>
              <div style='font-size:0.75rem; color:#8ab4c8;'>Risk Tier</div>
              <div style='font-size:1.2rem; font-weight:700; color:#e8f5ff; margin-top:2px;'>{tier}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Patient summary radar ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><div class='section-dot'></div>Patient Vitals vs Population Average</div>", unsafe_allow_html=True)

        radar_cols = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
        pop_mean   = df[radar_cols].mean()
        pat_vals   = [patient[c] for c in radar_cols]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(pop_mean / pop_mean.max() * 100),
            theta=radar_cols + [radar_cols[0]],
            fill='toself', name='Population Avg',
            line_color='#8ab4c8', fillcolor='rgba(138,180,200,0.12)',
        ))
        # normalize patient
        pat_norm = [v/m*100 for v,m in zip(pat_vals, pop_mean.max()*np.ones(5))]
        fig_radar.add_trace(go.Scatterpolar(
            r=pat_norm + [pat_norm[0]],
            theta=radar_cols + [radar_cols[0]],
            fill='toself', name='This Patient',
            line_color='#00c896' if prediction==0 else '#ef4444',
            fillcolor='rgba(0,200,150,0.12)' if prediction==0 else 'rgba(239,68,68,0.12)',
        ))
        fig_radar.update_layout(
            **PLOT_CFG, height=340,
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0,120], color='#8ab4c8',
                                gridcolor='rgba(138,180,200,0.12)'),
                angularaxis=dict(color='#8ab4c8', gridcolor='rgba(138,180,200,0.12)'),
            ),
            legend=dict(font=dict(color='#8ab4c8')),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════
# TAB 2 — DATASET INSIGHTS
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'><div class='section-dot'></div>Dataset Overview</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",      f"{len(df):,}")
    c2.metric("Heart Disease +ve",  f"{df['HeartDisease'].sum():,}  ({df['HeartDisease'].mean()*100:.1f}%)")
    c3.metric("Avg Age",            f"{df['Age'].mean():.1f} yr")
    c4.metric("Male Patients",      f"{(df['Sex']=='M').sum():,}  ({(df['Sex']=='M').mean()*100:.1f}%)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Distribution grid ──
    st.markdown("<div class='section-header'><div class='section-dot'></div>Numerical Distributions</div>", unsafe_allow_html=True)
    dist_col = st.selectbox("Select feature", NUMERICAL + ['Oldpeak'], label_visibility='collapsed')

    fig_dist = go.Figure()
    for label, grp, col in [(0,'No Disease','#00c896'),(1,'Heart Disease','#ef4444')]:
        sub = df[df['HeartDisease']==label][dist_col]
        fig_dist.add_trace(go.Histogram(
            x=sub, name=grp, nbinsx=30,
            marker_color=col, opacity=0.65,
            histnorm='probability density',
        ))
    fig_dist.update_layout(**PLOT_CFG, height=280, barmode='overlay',
        title=dict(text=f"Distribution of {dist_col} by Outcome", font=dict(size=13, color='#8ab4c8')),
        legend=dict(font=dict(color='#8ab4c8')),
        xaxis=dict(gridcolor='rgba(138,180,200,0.08)'),
        yaxis=dict(gridcolor='rgba(138,180,200,0.08)'),
    )
    st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})

    col_l, col_r = st.columns(2)

    with col_l:
        # Sex breakdown
        fig_sex = px.histogram(df, x='Sex', color='HeartDisease',
            barmode='group', color_discrete_map={0:'#00c896',1:'#ef4444'},
            title='Heart Disease by Sex', height=260)
        fig_sex.update_layout(**PLOT_CFG,
            xaxis=dict(gridcolor='rgba(138,180,200,0.08)'),
            yaxis=dict(gridcolor='rgba(138,180,200,0.08)'),
            legend=dict(title='', font=dict(color='#8ab4c8')),
        )
        st.plotly_chart(fig_sex, use_container_width=True, config={'displayModeBar': False})

    with col_r:
        # ChestPain breakdown
        fig_cp = px.histogram(df, x='ChestPainType', color='HeartDisease',
            barmode='group', color_discrete_map={0:'#00c896',1:'#ef4444'},
            title='Heart Disease by Chest Pain Type', height=260)
        fig_cp.update_layout(**PLOT_CFG,
            xaxis=dict(gridcolor='rgba(138,180,200,0.08)'),
            yaxis=dict(gridcolor='rgba(138,180,200,0.08)'),
            legend=dict(title='', font=dict(color='#8ab4c8')),
        )
        st.plotly_chart(fig_cp, use_container_width=True, config={'displayModeBar': False})

    # Correlation heatmap
    st.markdown("<div class='section-header'><div class='section-dot'></div>Correlation Matrix</div>", unsafe_allow_html=True)
    df_enc = pd.get_dummies(df, drop_first=True).astype(float)
    corr   = df_enc.corr(numeric_only=True)
    fig_hm = px.imshow(corr, color_continuous_scale='teal', text_auto='.2f',
                       title='Feature Correlation Heatmap', height=420)
    fig_hm.update_layout(**PLOT_CFG, coloraxis_colorbar=dict(tickfont=dict(color='#8ab4c8')))
    fig_hm.update_traces(textfont_size=9)
    st.plotly_chart(fig_hm, use_container_width=True, config={'displayModeBar': False})

    # Violin: Age by outcome
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_vio = px.violin(df, x='HeartDisease', y='Age', color='HeartDisease',
            box=True, points='outliers', color_discrete_map={0:'#00c896',1:'#ef4444'},
            title='Age Distribution by Outcome', height=290)
        fig_vio.update_layout(**PLOT_CFG,
            xaxis=dict(gridcolor='rgba(138,180,200,0.08)', tickvals=[0,1], ticktext=['Negative','Positive']),
            yaxis=dict(gridcolor='rgba(138,180,200,0.08)'),
            legend=dict(title='', font=dict(color='#8ab4c8')),
        )
        st.plotly_chart(fig_vio, use_container_width=True, config={'displayModeBar': False})

    with col_v2:
        fig_box = px.box(df, x='HeartDisease', y='Cholesterol', color='HeartDisease',
            color_discrete_map={0:'#00c896',1:'#ef4444'},
            title='Cholesterol vs Outcome', height=290)
        fig_box.update_layout(**PLOT_CFG,
            xaxis=dict(gridcolor='rgba(138,180,200,0.08)', tickvals=[0,1], ticktext=['Negative','Positive']),
            yaxis=dict(gridcolor='rgba(138,180,200,0.08)'),
            legend=dict(title='', font=dict(color='#8ab4c8')),
        )
        st.plotly_chart(fig_box, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
with tab3:
    results_data = [
        {'Model': 'Logistic Regression', 'Accuracy': 0.8696, 'F1': 0.8857},
        {'Model': 'KNN ★',               'Accuracy': 0.8641, 'F1': 0.8815},
        {'Model': 'Naive Bayes',          'Accuracy': 0.8533, 'F1': 0.8683},
        {'Model': 'Decision Tree',        'Accuracy': 0.7609, 'F1': 0.7864},
        {'Model': 'SVM',                  'Accuracy': 0.8478, 'F1': 0.8679},
    ]
    df_res = pd.DataFrame(results_data)
    df_res['Accuracy %'] = (df_res['Accuracy']*100).round(2)
    df_res['F1 %']       = (df_res['F1']*100).round(2)

    st.markdown("<div class='section-header'><div class='section-dot'></div>Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-banner'>★ KNN selected as production model. Balanced performance with low overfitting risk.</div>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name='Accuracy', x=df_res['Model'], y=df_res['Accuracy %'],
            marker_color='#00c896', text=df_res['Accuracy %'].apply(lambda x: f"{x}%"),
            textposition='outside',
        ))
        fig_cmp.add_trace(go.Bar(
            name='F1 Score', x=df_res['Model'], y=df_res['F1 %'],
            marker_color='#3b82f6', text=df_res['F1 %'].apply(lambda x: f"{x}%"),
            textposition='outside',
        ))
        fig_cmp.update_layout(**PLOT_CFG, height=320, barmode='group',
            title=dict(text="Accuracy & F1 Score", font=dict(size=13, color='#8ab4c8')),
            yaxis=dict(range=[70,95], gridcolor='rgba(138,180,200,0.08)'),
            xaxis=dict(gridcolor='rgba(138,180,200,0.08)'),
            legend=dict(font=dict(color='#8ab4c8')),
        )
        st.plotly_chart(fig_cmp, use_container_width=True, config={'displayModeBar': False})

    with col_m2:
        # Scatter: accuracy vs f1
        fig_sc = px.scatter(df_res, x='Accuracy %', y='F1 %', text='Model',
            size='Accuracy %', color='Model',
            color_discrete_sequence=['#00c896','#f59e0b','#3b82f6','#ef4444','#8b5cf6'],
            title='Accuracy vs F1 Score', height=320)
        fig_sc.update_traces(textposition='top center', textfont=dict(color='#e8f5ff', size=10))
        fig_sc.update_layout(**PLOT_CFG,
            xaxis=dict(range=[74,90], gridcolor='rgba(138,180,200,0.08)'),
            yaxis=dict(range=[77,91], gridcolor='rgba(138,180,200,0.08)'),
            showlegend=False,
        )
        st.plotly_chart(fig_sc, use_container_width=True, config={'displayModeBar': False})

    # Radar comparison
    st.markdown("<div class='section-header'><div class='section-dot'></div>Radar — Model Profiles</div>", unsafe_allow_html=True)
    fig_rad = go.Figure()
    colors  = ['#00c896','#f59e0b','#3b82f6','#ef4444','#8b5cf6']
    cats    = ['Accuracy', 'F1 Score']
    for i, row in df_res.iterrows():
        fig_rad.add_trace(go.Scatterpolar(
            r=[row['Accuracy %'], row['F1 %'], row['Accuracy %']],
            theta=cats + [cats[0]],
            fill='toself', name=row['Model'],
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)].replace('#','rgba(').rstrip(')') if False else f"rgba(0,0,0,0.05)",
        ))
    fig_rad.update_layout(**PLOT_CFG, height=350,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[75,92], color='#8ab4c8',
                            gridcolor='rgba(138,180,200,0.12)'),
            angularaxis=dict(color='#8ab4c8', gridcolor='rgba(138,180,200,0.12)'),
        ),
        legend=dict(font=dict(color='#8ab4c8')),
    )
    st.plotly_chart(fig_rad, use_container_width=True, config={'displayModeBar': False})

    # Metrics table
    st.markdown("<div class='section-header'><div class='section-dot'></div>Detailed Metrics Table</div>", unsafe_allow_html=True)
    display_df = df_res[['Model','Accuracy %','F1 %']].rename(columns={'Accuracy %':'Accuracy (%)','F1 %':'F1 Score (%)'})
    st.dataframe(
        display_df.style
            .highlight_max(subset=['Accuracy (%)','F1 Score (%)'], color='rgba(0,200,150,0.25)')
            .format({'Accuracy (%)': '{:.2f}', 'F1 Score (%)': '{:.2f}'}),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════
# TAB 4 — FEATURE REFERENCE
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'><div class='section-dot'></div>Feature Glossary</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-banner'>All features are derived from standard clinical assessments. Values outside normal ranges may elevate risk scores.</div>", unsafe_allow_html=True)

    features = [
        ("Age", "Years", "20–100", "Older age increases cardiovascular risk."),
        ("Sex", "M / F", "—", "Male patients show higher prevalence in this dataset."),
        ("ChestPainType", "ASY/ATA/NAP/TA", "—", "ASY (asymptomatic) is most associated with heart disease."),
        ("RestingBP", "mmHg", "80–200", "Normal <120. Hypertension >140 increases risk significantly."),
        ("Cholesterol", "mg/dL", "100–600", "Desirable <200; 200–239 borderline; >240 high."),
        ("FastingBS", "0 / 1", "—", "1 = fasting blood sugar >120 mg/dL (possible diabetes indicator)."),
        ("RestingECG", "Normal/ST/LVH", "—", "ST-wave or LVH abnormalities indicate cardiac stress."),
        ("MaxHR", "bpm", "60–220", "Lower max HR under exercise may signal poor cardiac reserve."),
        ("ExerciseAngina", "Y / N", "—", "Chest pain during exercise is a strong risk marker."),
        ("Oldpeak", "mm", "0.0–6.5", "ST depression induced by exercise. >2 considered significant."),
        ("ST_Slope", "Up/Flat/Down", "—", "Flat/Down slope during exercise associated with ischemia."),
    ]
    cols_hdr = st.columns([1.5, 1, 1, 3])
    for hdr, width in zip(["Feature", "Units", "Normal Range", "Clinical Note"], [1.5,1,1,3]):
        pass

    for feat, unit, rng, note in features:
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 3])
        c1.markdown(f"<span style='color:#00c896; font-weight:600;'>{feat}</span>", unsafe_allow_html=True)
        c2.markdown(f"<span style='color:#8ab4c8; font-size:0.88rem;'>{unit}</span>", unsafe_allow_html=True)
        c3.markdown(f"<span style='color:#8ab4c8; font-size:0.88rem;'>{rng}</span>", unsafe_allow_html=True)
        c4.markdown(f"<span style='color:#b0cfe0; font-size:0.88rem;'>{note}</span>", unsafe_allow_html=True)
        st.markdown("<hr style='margin:4px 0; border-color:rgba(0,200,150,0.08);'>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><div class='section-dot'></div>Preprocessing Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
      <ol style='color:#b0cfe0; line-height:2;'>
        <li>Zero-value imputation for <code>Cholesterol</code> and <code>RestingBP</code> using column mean.</li>
        <li>One-hot encoding of categorical features (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope).</li>
        <li>Standard scaling (μ=0, σ=1) applied to all numerical features before KNN inference.</li>
        <li>15 final model features after encoding and feature selection.</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8ab4c8; font-size:0.8rem; padding:10px 0 20px 0;'>
  🫀 <strong style='color:#00c896;'>CordisAI</strong> · For educational & research use only ·
  Not a substitute for professional medical advice · KNN model trained on UCI Heart Disease dataset
</div>
""", unsafe_allow_html=True)