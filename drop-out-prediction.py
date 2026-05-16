import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict — Student Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — Aurora gradient dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:      #0a0e1a;
    --bg-card:      #111827;
    --bg-card2:     #161d2e;
    --border:       rgba(99,179,237,0.15);
    --accent1:      #63b3ed;   /* sky blue */
    --accent2:      #9f7aea;   /* violet */
    --accent3:      #f6ad55;   /* amber */
    --grad-hero: linear-gradient(135deg, #0f2027 0%, #1a1040 40%, #0f2027 100%);
    --grad-card: linear-gradient(135deg, rgba(99,179,237,0.05) 0%, rgba(159,122,234,0.05) 100%);
    --text-primary:   #e2e8f0;
    --text-secondary: #94a3b8;
    --dropout-color:  #fc8181;
    --enrolled-color: #f6ad55;
    --graduate-color: #68d391;
    --font: 'Plus Jakarta Sans', sans-serif;
    --mono: 'JetBrains Mono', monospace;
}

/* ── Base ── */
html, body, [class*="css"], .stApp {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font) !important;
}

.stApp { background: var(--bg-base) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--font) !important; }

/* ── Sidebar header ── */
.sidebar-header {
    background: linear-gradient(135deg, rgba(99,179,237,0.15) 0%, rgba(159,122,234,0.15) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 16px;
    margin-bottom: 24px;
    text-align: center;
}
.sidebar-header h2 {
    font-size: 1.05rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f2027 0%, #1a1040 50%, #0f1f3d 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 24px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 40%, rgba(99,179,237,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 60%, rgba(159,122,234,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-icon { font-size: 3.5rem; margin-bottom: 12px; }
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #63b3ed 0%, #9f7aea 50%, #f6ad55 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0 0 12px;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-secondary);
    font-weight: 400;
    max-width: 600px;
}

/* ── Metric Cards ── */
.metric-row { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 18px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent-line, linear-gradient(90deg, var(--accent1), var(--accent2)));
}
.metric-card .metric-icon { font-size: 1.6rem; margin-bottom: 8px; }
.metric-card .metric-value {
    font-size: 2rem; font-weight: 800;
    background: var(--accent-grad, linear-gradient(90deg, var(--accent1), var(--accent2)));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-card .metric-label { font-size: 0.8rem; color: var(--text-secondary); font-weight: 500; margin-top: 2px; }

/* ── Section Title ── */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 28px 0 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Result Box ── */
.result-box {
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin: 20px 0;
    border: 2px solid;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at center, var(--result-glow, rgba(99,179,237,0.1)) 0%, transparent 70%);
    pointer-events: none;
}
.result-icon { font-size: 4rem; margin-bottom: 12px; }
.result-title {
    font-size: 2.2rem; font-weight: 800;
    margin-bottom: 8px;
}
.result-subtitle { font-size: 1rem; color: var(--text-secondary); }

.result-dropout { border-color: var(--dropout-color); --result-glow: rgba(252,129,129,0.12); }
.result-dropout .result-title { color: var(--dropout-color); }
.result-enrolled { border-color: var(--enrolled-color); --result-glow: rgba(246,173,85,0.12); }
.result-enrolled .result-title { color: var(--enrolled-color); }
.result-graduate { border-color: var(--graduate-color); --result-glow: rgba(104,211,145,0.12); }
.result-graduate .result-title { color: var(--graduate-color); }

/* ── Probability bars ── */
.prob-row { margin: 10px 0; }
.prob-label { 
    display: flex; justify-content: space-between; 
    font-size: 0.85rem; font-weight: 600; 
    margin-bottom: 5px; 
}
.prob-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 99px; height: 10px; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Feature importance bars ── */
.fi-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.fi-label { font-size: 0.8rem; color: var(--text-secondary); min-width: 190px; text-align: right; }
.fi-bar-bg { flex: 1; height: 8px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
.fi-bar-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent1), var(--accent2)); }
.fi-val { font-size: 0.75rem; font-family: var(--mono); color: var(--accent1); min-width: 45px; }

/* ── Confusion matrix ── */
.cm-table { width: 100%; border-collapse: collapse; }
.cm-table th, .cm-table td {
    border: 1px solid var(--border);
    padding: 12px 16px;
    text-align: center;
    font-size: 0.85rem;
}
.cm-table th { background: rgba(99,179,237,0.1); font-weight: 700; }
.cm-diag { background: rgba(104,211,145,0.15); font-weight: 800; color: var(--graduate-color); }

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--font) !important;
}
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-family: var(--font) !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,179,237,0.2), rgba(159,122,234,0.2)) !important;
    color: var(--text-primary) !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #63b3ed, #9f7aea) !important;
    color: #0a0e1a !important;
    font-family: var(--font) !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Info card ── */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.info-card h4 { font-size: 0.9rem; font-weight: 700; color: var(--accent1); margin: 0 0 6px; }
.info-card p { font-size: 0.85rem; color: var(--text-secondary); margin: 0; line-height: 1.5; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.3); border-radius: 99px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden; }

/* ── Metric card extras ── */
.metric-header { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.metric-header .metric-icon { font-size: 1.3rem; margin: 0; }
.metric-header .metric-label { font-size: 0.82rem; color: var(--text-secondary); font-weight: 600; margin: 0; }
.metric-big { font-size: 2.2rem; font-weight: 800; line-height: 1; margin-bottom: 4px;
    background: var(--accent-grad, linear-gradient(90deg, var(--accent1), var(--accent2)));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.metric-sublabel { font-size: 0.72rem; color: var(--text-secondary); margin-bottom: 2px; }
.metric-pct { font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px; }
.metric-spark { width: 100%; height: 36px; display: block; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CONSTANTS — ALL UNIQUE VALUES
# ─────────────────────────────────────────────
MARITAL_STATUS = ['Divorced','Facto Union','Legally Separated','Married','Single','Widower']
GENDER         = ['Female','Male']
INTL_STUDENT   = ['No','Yes']
DISPLACED      = ['No','Yes']
EDU_SPECIAL    = ['No','Yes']
COURSES = ['Advertising and Marketing Management','Agronomy','Animation and Multimedia Design',
           'Basic Education','Biofuel Production Technologies','Communication Design','Equiniculture',
           'Informatics Engineering','Journalism and Communication','Management','Management (Evening)',
           'Nursing','Oral Hygiene','Social Service','Social Service (Evening)','Tourism','Veterinary Nursing']
CLASS_SCHED    = ['Daytime','Evening']
PREV_QUAL = ['10th Year of Schooling','10th Year – Not Completed','11th Year – Not Completed',
             '12th Year – Not Completed','Basic Education 2nd Cycle','Basic Education 3rd Cycle',
             'Frequency of Higher Education',"Higher Education – Bachelor's Degree",
             'Higher Education – Degree','Higher Education – Degree (1st Cycle)',
             'Higher Education – Doctorate',"Higher Education – Master's (2nd Cycle)",
             "Higher Education – Master's Degree",'Other – 11th Year of Schooling',
             'Professional Higher Technical Course','Secondary Education','Technological Specialization Course']
MOM_QUAL = ['10th Year of Schooling','2nd Cycle of the General High School Course',
            '2nd Year Complementary High School Course','7th Year (Old)','7th Year of Schooling',
            'Basic Education 1st Cycle (4th/5th Year)','Basic Education 3rd Cycle or Equivalent',
            'Can Read Without 4th Year of Schooling','Frequency of Higher Education',
            'General Course of Administration and Commerce',"Higher Education – Bachelor's Degree",
            'Higher Education – Degree','Higher Education – Degree (1st Cycle)',
            'Higher Education – Doctorate','Higher Education – Doctorate (3rd Cycle)',
            "Higher Education – Master's Degree",'Other – 11th Year of Schooling',
            'Secondary Education – 12th Year or Equivalent','Technological Specialization Course','Unknown']
DAD_QUAL = ['10th Year of Schooling','2nd Cycle of the General High School Course',
            '2nd Year Complementary High School Course','7th Year (Old)','7th Year of Schooling',
            '9th Year – Not Completed','Basic Education 1st Cycle (4th/5th Year)',
            'Basic Education 3rd Cycle or Equivalent','Can Read Without 4th Year of Schooling',
            'Cannot Read or Write','Frequency of Higher Education','General Commerce Course',
            'General Course of Administration and Commerce',"Higher Education – Bachelor's Degree",
            'Higher Education – Degree','Higher Education – Degree (1st Cycle)',
            'Higher Education – Doctorate','Higher Education – Doctorate (3rd Cycle)',
            "Higher Education – Master's Degree","Higher Education – Master's Degree (2nd Cycle)",
            'Other – 11th Year of Schooling','Secondary Education – 12th Year or Equivalent',
            'Specialized Higher Studies Course','Technological Specialization Course','Unknown']
MOM_OCC = ['(Blank / Unknown)','(Not Applicable)','Administrative Staff',
           'Data. Accounting. Statistical & Financial Services Operators',
           'Farmers & Skilled Workers in Agriculture. Fisheries. and Forestry',
           'Health Professionals','Health Technicians – Intermediate Level',
           'Information and Communication Technology Specialists',
           'Installation and Machine Operators & Assembly Workers',
           'Intermediate Level Science & Engineering Technicians',
           'Intermediate Level Technicians and Professions',
           'Intermediate Level Technicians – Legal. Social. Sports. Cultural',
           'Meal Preparation Assistants','Office Workers. Secretaries & Data Processing Operators',
           'Other Administrative Support Staff','Other Situation','Personal Care Workers and the Like',
           'Personal Service Workers','Personal Services. Security & Safety Workers. and Sellers',
           'Representatives of Legislative Power & Executive Bodies','Sellers',
           'Skilled Construction Workers (excl. Electricians)',
           'Skilled Workers in Industry. Construction. and Craftsmen',
           'Skilled Workers in Printing. Precision Instruments & Jewellery',
           'Specialists in Intellectual and Scientific Activities',
           'Street Vendors (excl. Food) and Street Service Providers','Student','Teachers',
           'Unskilled Workers','Unskilled Workers in Agriculture & Fisheries',
           'Unskilled Workers in Extractive Industry & Construction',
           'Workers in Food Processing. Woodworking & Clothing Industries']
DAD_OCC = ['(Blank / Unknown)','(Not Applicable)','Administrative Staff','Armed Forces Officers',
           'Armed Forces Sergeants','Assembly Workers',
           'Data. Accounting. Statistical & Financial Services Operators',
           'Directors of Administrative and Commercial Services','Domestic Service Workers',
           'Farmers & Skilled Workers in Agriculture. Fisheries. and Forestry',
           'Farmers. Livestock Keepers. Fishermen. Hunters (Subsistence)',
           'Finance. Accounting & Administrative Relations Specialists',
           'Fixed Plant and Machine Operators','Health Professionals',
           'Health Technicians – Intermediate Level',
           'Hotel. Catering. Trade. and Other Services Directors',
           'Information and Communication Technology Technicians',
           'Installation and Machine Operators & Assembly Workers',
           'Intermediate Level Science & Engineering Technicians',
           'Intermediate Level Technicians and Professions',
           'Intermediate Level Technicians – Legal. Social. Sports. Cultural',
           'Market-Oriented Farmers & Skilled Agricultural Workers','Meal Preparation Assistants',
           'Office Workers. Secretaries & Data Processing Operators',
           'Other Administrative Support Staff','Other Armed Forces Personnel','Other Situation',
           'Personal Care Workers and the Like','Personal Service Workers',
           'Personal Services. Security & Safety Workers. and Sellers',
           'Protection and Security Services Personnel',
           'Representatives of Legislative Power & Executive Bodies','Sellers',
           'Skilled Construction Workers (excl. Electricians)',
           'Skilled Workers in Electricity and Electronics',
           'Skilled Workers in Industry. Construction. and Craftsmen',
           'Skilled Workers in Metallurgy and Metalworking',
           'Specialists in Intellectual and Scientific Activities',
           'Specialists in Physical Sciences. Mathematics & Engineering',
           'Street Vendors (excl. Food) and Street Service Providers','Student','Teachers',
           'Unskilled Workers','Unskilled Workers in Extractive Industry & Construction',
           'Vehicle Drivers and Mobile Equipment Operators',
           'Workers in Food Processing. Woodworking & Clothing Industries']
SCHOLARSHIP    = ['No','Yes']
DEBTOR         = ['No','Yes']
TUITION        = ['No','Yes']

FEATURES = ['Marital Status','Gender','Age at Enrollment','International Student','Displaced',
            'Educational Special Needs','Course','Class Schedule','Previous Qualification',
            "Mother's Qualification","Father's Qualification","Mother's Occupation",
            "Father's Occupation",'Scholarship Holder','Debtor','Tuition Fees Up to Date',
            'Sem 1 – Grade','Sem 2 – Grade']

TARGET_CLASSES = ['Dropout','Enrolled','Graduate']

ENCODERS_MAP = {
    'Marital Status': MARITAL_STATUS,
    'Gender': GENDER,
    'International Student': INTL_STUDENT,
    'Displaced': DISPLACED,
    'Educational Special Needs': EDU_SPECIAL,
    'Course': COURSES,
    'Class Schedule': CLASS_SCHED,
    'Previous Qualification': PREV_QUAL,
    "Mother's Qualification": MOM_QUAL,
    "Father's Qualification": DAD_QUAL,
    "Mother's Occupation": MOM_OCC,
    "Father's Occupation": DAD_OCC,
    'Scholarship Holder': SCHOLARSHIP,
    'Debtor': DEBTOR,
    'Tuition Fees Up to Date': TUITION,
}


# ─────────────────────────────────────────────
# MODEL TRAINING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Training Random Forest model…")
def train_model():
    df = pd.read_excel('data_prediksi_dropout_labeled.xlsx')
    X = df[FEATURES].copy()
    y = df['Target'].copy()

    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == 'str':
            classes = ENCODERS_MAP.get(c, sorted(X[c].fillna('Unknown').astype(str).unique().tolist()))
            mapping = {v: i for i, v in enumerate(classes)}
            X[c] = X[c].fillna('Unknown').astype(str).map(mapping).fillna(0)
        else:
            X[c] = X[c].fillna(0)

    X_np = X.values.astype(float)
    target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    y_enc = np.array([target_mapping.get(str(v), 0) for v in y])

    X_train, X_test, y_train, y_test = train_test_split(X_np, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fi = rf.feature_importances_
    report = classification_report(y_test, y_pred, target_names=TARGET_CLASSES, output_dict=True)

    return rf, acc, cm, fi, report, len(df), df


def encode_input(vals: dict) -> np.ndarray:
    row = []
    for feat in FEATURES:
        v = vals[feat]
        if feat in ENCODERS_MAP:
            classes = ENCODERS_MAP[feat]
            row.append(float(classes.index(v)) if v in classes else 0.0)
        else:
            row.append(float(v))
    return np.array(row).reshape(1, -1)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
rf_model, accuracy, conf_matrix, feat_importance, class_report, n_samples, df_raw = train_model()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size:2rem;">📊</div>
        <h2>Early Dropout<br>Prediction</h2>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px;">Student Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", ["🏠 Overview", "🔮 Predict Student", "📋 Batch Predict", "ℹ️ About"],
                    label_visibility="collapsed")

    st.markdown("""
    <style>
    /* Hilangkan bulatan radio */
    [data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* Container menu */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 4px !important;
        padding: 4px 0 !important;
    }

    /* Tiap item menu */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        display: flex !important;
        align-items: center !important;
        padding: 11px 16px !important;
        border-radius: 12px !important;
        cursor: pointer !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        transform: translateX(0px) !important;
        border: 1px solid transparent !important;
        margin: 1px 0 !important;
        position: relative !important;
        overflow: hidden !important;
    }

    /* Animasi hover — geser kanan + highlight */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: linear-gradient(135deg, rgba(99,179,237,0.12), rgba(159,122,234,0.08)) !important;
        color: #e2e8f0 !important;
        transform: translateX(4px) !important;
        border-color: rgba(99,179,237,0.2) !important;
        box-shadow: 0 2px 12px rgba(99,179,237,0.1) !important;
    }

    /* Item yang sedang aktif/dipilih */
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(99,179,237,0.2), rgba(159,122,234,0.15)) !important;
        color: #e2e8f0 !important;
        border-color: rgba(99,179,237,0.35) !important;
        font-weight: 700 !important;
        transform: translateX(4px) !important;
        box-shadow: 0 2px 16px rgba(99,179,237,0.15), inset 0 0 0 1px rgba(99,179,237,0.1) !important;
    }

    /* Garis kiri biru untuk item aktif */
    [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked)::before {
        content: '' !important;
        position: absolute !important;
        left: 0 !important; top: 20% !important; bottom: 20% !important;
        width: 3px !important;
        background: linear-gradient(180deg, #63b3ed, #9f7aea) !important;
        border-radius: 99px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:32px;padding:0 8px;">
        <div style="text-align:center;font-size:3.5rem;margin-bottom:4px;">👩‍💻👨‍💻</div>
        <div style="
            background: linear-gradient(135deg, rgba(99,179,237,0.08), rgba(159,122,234,0.08));
            border: 1px solid rgba(99,179,237,0.15);
            border-radius: 14px;
            padding: 14px 12px;
            text-align: center;
        ">
            <div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;line-height:1.5;">
                Support students.<br>Empower futures.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── Counts
    target_counts  = df_raw['Target'].value_counts()
    dropout_count  = int(target_counts.get('Dropout', 0))
    enrolled_count = int(target_counts.get('Enrolled', 0))
    graduate_count = int(target_counts.get('Graduate', 0))
    dropout_pct    = dropout_count  / n_samples * 100
    enrolled_pct   = enrolled_count / n_samples * 100
    graduate_pct   = graduate_count / n_samples * 100
    # ── Page header (mirip "Overview" di gambar)
    st.markdown("""
    <div class="hero-banner" style="margin-bottom:28px;">
        <div class="hero-icon">👨‍🎓👩‍🎓</div>
        <div class="hero-title">Student Dropout<br>Early Warning System</div>
        <div class="hero-sub">Identify at-risk students before it's too late</div>
    </div>
    <div style="margin-bottom:20px;">
        <div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;">Overview</div>
        <div style="font-size:0.85rem;color:#94a3b8;margin-top:2px;">
            Monitor student status and early prediction insights
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sparkline helper
    def spark(points, color):
        return f'<svg viewBox="0 0 120 35" preserveAspectRatio="none" style="width:100%;height:38px;display:block;margin-top:10px;"><polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/></svg>'

    spark_total = spark("0,28 15,22 30,25 45,18 60,21 75,14 90,17 105,10 120,13", "#63b3ed")
    spark_grad  = spark("0,28 15,22 30,25 45,18 60,21 75,14 90,17 105,10 120,13", "#68d391")
    spark_risk  = spark("0,24 15,28 30,20 45,26 60,18 75,24 90,16 105,22 120,18", "#f6ad55")
    spark_drop  = spark("0,20 15,26 30,18 45,28 60,16 75,26 90,14 105,24 120,19", "#fc8181")
    spark_acc   = spark("0,30 15,24 30,27 45,18 60,22 75,13 90,16 105,9  120,11", "#9f7aea")

    # ── 5 metric cards
    st.markdown(f"""
    <div style="display:flex;gap:14px;margin-bottom:28px;flex-wrap:wrap;">

      <div style="flex:1;min-width:150px;background:#111827;border:1px solid rgba(99,179,237,0.15);
           border-radius:16px;padding:20px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
             background:linear-gradient(90deg,#63b3ed,#9f7aea);"></div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <span style="font-size:1.4rem;">👥</span>
          <span style="font-size:0.8rem;color:#94a3b8;font-weight:600;">Total Students</span>
        </div>
        <div style="font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#63b3ed,#9f7aea);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;">
          {n_samples:,}</div>
        <div style="font-size:0.72rem;color:#94a3b8;margin-top:3px;">All students</div>
        {spark_total}
      </div>

      <div style="flex:1;min-width:150px;background:#111827;border:1px solid rgba(99,179,237,0.15);
           border-radius:16px;padding:20px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
             background:linear-gradient(90deg,#68d391,#9ae6b4);"></div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <span style="font-size:1.4rem;">👩‍🎓</span>
          <span style="font-size:0.8rem;color:#94a3b8;font-weight:600;">Low Risk</span>
        </div>
        <div style="font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#68d391,#9ae6b4);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;">
          {graduate_count:,}</div>
        <div style="font-size:0.72rem;color:#94a3b8;margin-top:3px;">{graduate_pct:.1f}% of total</div>
        {spark_grad}
      </div>

      <div style="flex:1;min-width:150px;background:#111827;border:1px solid rgba(99,179,237,0.15);
           border-radius:16px;padding:20px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
             background:linear-gradient(90deg,#f6ad55,#fbd38d);"></div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <span style="font-size:1.4rem;">⚡</span>
          <span style="font-size:0.8rem;color:#94a3b8;font-weight:600;">At Risk</span>
        </div>
        <div style="font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#f6ad55,#fbd38d);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;">
          {enrolled_count:,}</div>
        <div style="font-size:0.72rem;color:#94a3b8;margin-top:3px;">{enrolled_pct:.1f}% of total</div>
        {spark_risk}
      </div>

      <div style="flex:1;min-width:150px;background:#111827;border:1px solid rgba(99,179,237,0.15);
           border-radius:16px;padding:20px 18px;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
             background:linear-gradient(90deg,#fc8181,#f687b3);"></div>
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
          <span style="font-size:1.4rem;">⚠️</span>
          <span style="font-size:0.8rem;color:#94a3b8;font-weight:600;">Dropout</span>
        </div>
        <div style="font-size:2.2rem;font-weight:800;background:linear-gradient(90deg,#fc8181,#f687b3);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;">
          {dropout_count:,}</div>
        <div style="font-size:0.72rem;color:#94a3b8;margin-top:3px;">{dropout_pct:.1f}% of total</div>
        {spark_drop}
      </div>

          

    </div>
    """, unsafe_allow_html=True)

    # ── Row 2: Pie chart | Top Risk Factors | Risk by Program
    col1, col2, col3 = st.columns([1.1, 1, 1.2])

    with col1:
        st.markdown('<div class="section-title">🥧 Dropout Risk Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4, 3.8))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        sizes  = [graduate_count, enrolled_count, dropout_count]
        colors = ['#68d391', '#f6ad55', '#fc8181']
        wedges, _ = ax.pie(
            sizes, colors=colors, startangle=90,
            wedgeprops={'linewidth': 2.5, 'edgecolor': '#111827', 'width': 0.55},
        )
        # Legend manual
        legend_items = [
            ('🟢 Low Risk', graduate_count, '#68d391', f'{graduate_pct:.1f}%'),
            ('🟡 At Risk',  enrolled_count,  '#f6ad55', f'{enrolled_pct:.1f}%'),
            ('🔴 Dropout',  dropout_count,   '#fc8181', f'{dropout_pct:.1f}%'),
        ]
        for i, (lbl, cnt, clr, pct) in enumerate(legend_items):
            ax.text(1.15, 0.35 - i*0.38, f'● {lbl}', color=clr, fontsize=8.5,
                    transform=ax.transAxes, fontweight='bold')
            ax.text(1.15, 0.22 - i*0.38, f'{cnt:,} students', color='#94a3b8', fontsize=7.5,
                    transform=ax.transAxes)
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown('<div class="section-title">📋 Top Risk Factors</div>', unsafe_allow_html=True)
        fi_sorted = sorted(zip(FEATURES, feat_importance), key=lambda x: -x[1])[:5]
        max_fi = fi_sorted[0][1]
        risk_html = ""
        for feat, imp in fi_sorted:
            pct = imp / max_fi * 100
            risk_html += f"""
            <div style="margin-bottom:14px;">
              <div style="font-size:0.82rem;color:#e2e8f0;font-weight:500;margin-bottom:5px;">{feat}</div>
              <div style="display:flex;align-items:center;gap:10px;">
                <div style="flex:1;height:8px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden;">
                  <div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,#63b3ed,#9f7aea);border-radius:99px;"></div>
                </div>
                <span style="font-size:0.75rem;color:#63b3ed;font-family:monospace;min-width:36px;">{pct:.0f}%</span>
              </div>
            </div>"""
        st.markdown(risk_html, unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-title">📊 Risk by Program</div>', unsafe_allow_html=True)
        prog = df_raw.groupby('Course')['Target'].value_counts().unstack(fill_value=0)
        prog['Total'] = prog.sum(axis=1)
        prog = prog.sort_values('Total', ascending=False).head(5)

        fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
        fig2.patch.set_facecolor('#111827')
        ax2.set_facecolor('#111827')

        n_courses = len(prog)
        courses_short = [c[:16] + '…' if len(c) > 16 else c for c in prog.index]
        y_pos = np.arange(n_courses)

        do_v = prog.get('Dropout',  pd.Series([0]*n_courses)).values.astype(float)
        en_v = prog.get('Enrolled', pd.Series([0]*n_courses)).values.astype(float)
        gr_v = prog.get('Graduate', pd.Series([0]*n_courses)).values.astype(float)
        totals = do_v + en_v + gr_v
        totals[totals == 0] = 1

        # stacked horizontal bar (pct)
        ax2.barh(y_pos, gr_v/totals*100, color='#68d391', height=0.55, label='Low Risk')
        ax2.barh(y_pos, en_v/totals*100, color='#f6ad55', height=0.55, label='At Risk',
                 left=gr_v/totals*100)
        ax2.barh(y_pos, do_v/totals*100, color='#fc8181', height=0.55, label='Dropout',
                 left=(gr_v+en_v)/totals*100)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(courses_short, color='white', fontsize=7.5)
        ax2.tick_params(colors='white', labelsize=7.5)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Percentage (%)', color='#94a3b8', fontsize=7.5)
        ax2.spines[['top','right','left','bottom']].set_color('#1e293b')
        ax2.xaxis.label.set_color('#94a3b8')
        legend = ax2.legend(fontsize=7, labelcolor='white', facecolor='#111827',
                            edgecolor='#1e293b', loc='lower right')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Row 3: Recent At-Risk Students | Tagline footer
    st.markdown('<div class="section-title">🚨 Recent At-Risk Students (Sample)</div>', unsafe_allow_html=True)

    at_risk_df = df_raw[df_raw['Target'] == 'Enrolled'].head(4)[
        ['Course', 'Sem 1 – Grade', 'Sem 2 – Grade', 'Scholarship Holder', 'Debtor']
    ].copy()
    at_risk_df.index = range(1, len(at_risk_df)+1)

    col_a, col_b = st.columns([1.6, 1])
    with col_a:
        st.dataframe(at_risk_df, use_container_width=True)
    with col_b:
        st.markdown(f"""
        <div style="background:#111827;border:1px solid rgba(99,179,237,0.15);border-radius:16px;
             padding:20px;height:100%;">
          <div style="font-size:0.85rem;color:#94a3b8;font-weight:600;margin-bottom:12px;">
            📊 Intervention Overview
          </div>
          <div style="font-size:0.82rem;color:#e2e8f0;margin-bottom:8px;">
            🎯 At-Risk Identified &nbsp;
            <span style="float:right;font-weight:700;color:#f6ad55;">{enrolled_count:,}</span>
          </div>
          <div style="font-size:0.82rem;color:#e2e8f0;margin-bottom:8px;">
            ✅ Graduates &nbsp;
            <span style="float:right;font-weight:700;color:#68d391;">{graduate_count:,}</span>
          </div>
          <div style="font-size:0.82rem;color:#e2e8f0;margin-bottom:8px;">
            ⚠️ Dropouts &nbsp;
            <span style="float:right;font-weight:700;color:#fc8181;">{dropout_count:,}</span>
          </div>
          <hr style="border-color:rgba(99,179,237,0.15);margin:12px 0;">
          <div style="font-size:0.75rem;color:#63b3ed;font-weight:600;text-align:center;margin-top:8px;">
            👩‍💻👨‍💻 Support students. Empower futures.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer tagline
    st.markdown("""
    <div style="margin-top:24px;padding:14px 20px;background:#111827;border:1px solid rgba(99,179,237,0.12);
         border-radius:12px;display:flex;align-items:center;gap:10px;">
      <span style="font-size:1.1rem;">👩‍🎓</span>
      <span style="font-size:0.82rem;color:#94a3b8;">
        <strong style="color:#63b3ed;">Early detection.</strong> &nbsp;
        <strong style="color:#9f7aea;">Timely intervention.</strong> &nbsp;
        <strong style="color:#68d391;">Better outcomes.</strong>
      </span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PREDICT STUDENT
# ─────────────────────────────────────────────
elif page == "🔮 Predict Student":
    st.markdown("""
    <div style="margin-bottom:28px;">
        <div style="font-size:2rem;font-weight:800;background:linear-gradient(90deg,#63b3ed,#9f7aea);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
            🔮 Predict Student Outcome
        </div>
        <div style="color:#94a3b8;margin-top:6px;">Fill in student details to get an AI-powered prediction</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        # ── Section 1: Personal Info
        st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            marital = st.selectbox("Marital Status", MARITAL_STATUS, index=4)
            gender  = st.selectbox("Gender", GENDER, index=1)
        with c2:
            age     = st.number_input("Age at Enrollment", min_value=17, max_value=70, value=20)
            intl    = st.selectbox("International Student", INTL_STUDENT, index=0)
        with c3:
            displaced = st.selectbox("Displaced", DISPLACED, index=0)
            edu_special = st.selectbox("Educational Special Needs", EDU_SPECIAL, index=0)

        # ── Section 2: Academic
        st.markdown('<div class="section-title">📚 Academic Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            course      = st.selectbox("Course", COURSES, index=11)
            class_sched = st.selectbox("Class Schedule", CLASS_SCHED, index=0)
        with c2:
            prev_qual   = st.selectbox("Previous Qualification", PREV_QUAL, index=15)
            scholarship = st.selectbox("Scholarship Holder", SCHOLARSHIP, index=0)
        with c3:
            sem1 = st.number_input("Sem 1 – Grade(0-20)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
            sem2 = st.number_input("Sem 2 – Grade(0-20)", min_value=0.0, max_value=20.0, value=13.0, step=0.5)

        # ── Section 3: Family
        st.markdown('<div class="section-title">👨‍👩‍👧 Family Background</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            mom_qual = st.selectbox("Mother's Qualification", MOM_QUAL, index=17)
            mom_occ  = st.selectbox("Mother's Occupation", MOM_OCC, index=2)
        with c2:
            dad_qual = st.selectbox("Father's Qualification", DAD_QUAL, index=21)
            dad_occ  = st.selectbox("Father's Occupation", DAD_OCC, index=36)

        # ── Section 4: Financial
        st.markdown('<div class="section-title">💰 Financial Status</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            debtor  = st.selectbox("Debtor", DEBTOR, index=0)
        with c2:
            tuition = st.selectbox("Tuition Fees Up to Date", TUITION, index=1)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Predict Outcome")

    if submitted:
        user_vals = {
            'Marital Status': marital, 'Gender': gender, 'Age at Enrollment': age,
            'International Student': intl, 'Displaced': displaced,
            'Educational Special Needs': edu_special, 'Course': course,
            'Class Schedule': class_sched, 'Previous Qualification': prev_qual,
            "Mother's Qualification": mom_qual, "Father's Qualification": dad_qual,
            "Mother's Occupation": mom_occ, "Father's Occupation": dad_occ,
            'Scholarship Holder': scholarship, 'Debtor': debtor,
            'Tuition Fees Up to Date': tuition, 'Sem 1 – Grade': sem1, 'Sem 2 – Grade': sem2,
        }

        X_input = encode_input(user_vals)
        pred    = rf_model.predict(X_input)[0]
        proba   = rf_model.predict_proba(X_input)[0]
        pred_label = TARGET_CLASSES[pred]

        # Result display
        RESULT_MAP = {
            'Dropout':  ('⚠️', 'result-dropout',  'High risk of dropout — early intervention recommended.'),
            'Enrolled': ('📚', 'result-enrolled', 'Student is likely to remain enrolled this semester.'),
            'Graduate': ('🏆', 'result-graduate', 'Student is on a strong graduation trajectory!'),
        }
        icon, css_class, sub = RESULT_MAP[pred_label]
        conf = proba[pred] * 100

        st.markdown(f"""
        <div class="result-box {css_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-title">{pred_label}</div>
            <div class="result-subtitle">{sub}</div>
            <div style="margin-top:16px;font-size:0.85rem;color:#94a3b8;">
        </div>
        """, unsafe_allow_html=True)



# ─────────────────────────────────────────────
# PAGE: BATCH PREDICT
# ─────────────────────────────────────────────
elif page == "📋 Batch Predict":
    st.markdown("""
    <div style="font-size:2rem;font-weight:800;background:linear-gradient(90deg,#63b3ed,#9f7aea);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:8px;">
        📋 Batch Prediction
    </div>
    <div style="color:#94a3b8;margin-bottom:28px;">Upload an Excel/CSV file to predict outcomes for multiple students at once</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>📌 File Requirements</h4>
        <p>Upload a file with these exact columns: Marital Status, Gender, Age at Enrollment, 
        International Student, Displaced, Educational Special Needs, Course, Class Schedule, 
        Previous Qualification, Mother's Qualification, Father's Qualification, Mother's Occupation, 
        Father's Occupation, Scholarship Holder, Debtor, Tuition Fees Up to Date, Sem 1 – Grade, Sem 2 – Grade</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload student data file", type=["xlsx", "csv"])

    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df_batch = pd.read_csv(uploaded)
            else:
                df_batch = pd.read_excel(uploaded)

            missing = [c for c in FEATURES if c not in df_batch.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                X_batch = df_batch[FEATURES].copy()
                for c in X_batch.columns:
                    if c in ENCODERS_MAP:
                        classes = ENCODERS_MAP[c]
                        mapping = {v: i for i, v in enumerate(classes)}
                        X_batch[c] = X_batch[c].fillna('Unknown').astype(str).map(mapping).fillna(0)
                    else:
                        X_batch[c] = X_batch[c].fillna(0)

                X_np_batch = X_batch.values.astype(float)
                preds = rf_model.predict(X_np_batch)
                probas = rf_model.predict_proba(X_np_batch)

                df_result = df_batch.copy()
                df_result['Prediction'] = [TARGET_CLASSES[p] for p in preds]
                df_result['Confidence'] = [f"{probas[i][p]*100:.1f}%" for i, p in enumerate(preds)]
                df_result['P(Dropout)'] = [f"{probas[i][0]*100:.1f}%" for i in range(len(preds))]
                df_result['P(Enrolled)'] = [f"{probas[i][1]*100:.1f}%" for i in range(len(preds))]
                df_result['P(Graduate)'] = [f"{probas[i][2]*100:.1f}%" for i in range(len(preds))]

                # Summary
                pred_counts = pd.Series([TARGET_CLASSES[p] for p in preds]).value_counts()
                n = len(preds)
                do_pct = pred_counts.get('Dropout', 0)/n
                en_pct = pred_counts.get('Enrolled', 0)/n
                gr_pct = pred_counts.get('Graduate', 0)/n

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card" style="--accent-line:linear-gradient(90deg,#fc8181,#f687b3);--accent-grad:linear-gradient(90deg,#fc8181,#f687b3);">
                        <div class="metric-icon">⚠️</div>
                        <div class="metric-value">{pred_counts.get('Dropout',0)}</div>
                        <div class="metric-label">PREDICTED DROPOUT ({do_pct:.0%})</div>
                    </div>
                    <div class="metric-card" style="--accent-line:linear-gradient(90deg,#f6ad55,#fbd38d);--accent-grad:linear-gradient(90deg,#f6ad55,#fbd38d);">
                        <div class="metric-icon">📚</div>
                        <div class="metric-value">{pred_counts.get('Enrolled',0)}</div>
                        <div class="metric-label">PREDICTED ENROLLED ({en_pct:.0%})</div>
                    </div>
                    <div class="metric-card" style="--accent-line:linear-gradient(90deg,#68d391,#9ae6b4);--accent-grad:linear-gradient(90deg,#68d391,#9ae6b4);">
                        <div class="metric-icon">🏆</div>
                        <div class="metric-value">{pred_counts.get('Graduate',0)}</div>
                        <div class="metric-label">PREDICTED GRADUATE ({gr_pct:.0%})</div>
                    </div>
                    <div class="metric-card" style="--accent-line:linear-gradient(90deg,#63b3ed,#9f7aea);--accent-grad:linear-gradient(90deg,#63b3ed,#9f7aea);">
                        <div class="metric-icon">📊</div>
                        <div class="metric-value">{n}</div>
                        <div class="metric-label">TOTAL STUDENTS</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-title">📋 Prediction Results</div>', unsafe_allow_html=True)
                st.dataframe(df_result, use_container_width=True, hide_index=True)

                # Download
                buf = io.BytesIO()
                df_result.to_excel(buf, index=False)
                st.download_button("⬇️ Download Results as Excel", buf.getvalue(),
                                   file_name="predictions.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#94a3b8;">
            <div style="font-size:3rem;margin-bottom:12px;">📂</div>
            <div style="font-size:1rem;">Upload an Excel or CSV file to start batch prediction</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("""
    <div style="font-size:2rem;font-weight:800;background:linear-gradient(90deg,#63b3ed,#9f7aea);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:28px;">
        ℹ️ About EduPredict
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Purpose</h4>
            <p>EduPredict is an AI-powered early warning system that uses machine learning to identify 
            students at risk of dropping out — enabling timely intervention by educators and counselors.</p>
        </div>
        <div class="info-card">
            <h4>📊 Dataset</h4>
            <p>4,424 student records with 18 features spanning demographics, academic history, family background, 
            and financial status. Three outcome classes: Dropout, Enrolled, and Graduate.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="info-card">
            <h4>🔬 Features Used (18)</h4>
            <p>
            • Marital Status &nbsp;• Gender &nbsp;• Age at Enrollment<br>
            • International Student &nbsp;• Displaced<br>
            • Educational Special Needs &nbsp;• Course<br>
            • Class Schedule &nbsp;• Previous Qualification<br>
            • Mother's & Father's Qualification<br>
            • Mother's & Father's Occupation<br>
            • Scholarship Holder &nbsp;• Debtor<br>
            • Tuition Fees Up to Date<br>
            • Sem 1 Grade &nbsp;• Sem 2 Grade
            </p>
        </div>
        <div class="info-card">
            <h4>⚡ Technical Stack</h4>
            <p>Python · Streamlit · scikit-learn · pandas · NumPy</p>
        </div>
        <div class="info-card">
            <h4>⚠️ Ethical Note</h4>
            <p>Predictions are decision-support tools, not deterministic labels. 
            Human judgment should always be applied when interpreting results and designing interventions. 
            Model accuracy is ~81% — there will be misclassifications.</p>
        </div>
        """, unsafe_allow_html=True)
