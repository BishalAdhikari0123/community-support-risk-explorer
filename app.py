from pathlib import Path
import sys

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from uk_risk.data import FEATURES, make_demo_dataset, validate_dataset  # noqa: E402
from uk_risk.modeling import explain_row, train_model  # noqa: E402

st.set_page_config(page_title="Community Support Risk Explorer", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #16324f; --muted: #475569; --sky: #b9e4f4; --blue: #005a8d; --green: #176b55; --amber: #b54708; --cream: #f5f7fa; --paper: #ffffff; --line: #cbd5e1; }
    .stApp { background: radial-gradient(circle at 100% 0%, #e5f2f8 0, transparent 29rem), var(--cream); color: var(--ink); }
    .block-container { max-width: 1440px; padding: 2.2rem 3.5rem 3rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    p, label, .stMarkdown, .stCaption { font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: #e8f1f6; border-right: 1px solid #c8d9e4; }
    [data-testid="stSidebar"] h2 { font-size: 1rem; }
    .topline { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.4rem; color: var(--muted); font-size: .8rem; letter-spacing: .04em; text-transform: uppercase; }
    .brand { color: var(--blue); font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: 0; }
    .hero { background: #162a2c; color: #f5f8f3; padding: 2.5rem 2.8rem 2.7rem; border-radius: 16px; margin-bottom: 1.1rem; box-shadow: 0 16px 35px rgba(20, 35, 40, .12); position: relative; overflow: hidden; }
    .hero:after { content: ''; position: absolute; width: 18rem; height: 18rem; border: 1px solid rgba(185, 228, 244, .42); border-radius: 50%; right: -4rem; top: -8rem; }
    .hero h1 { color: #f5f8f3; font-size: clamp(2.2rem, 4vw, 4rem); line-height: .98; max-width: 760px; margin: .35rem 0 .9rem; letter-spacing: 0; }
    .eyebrow { color: var(--sky); font-weight: 700; font-size: .72rem; letter-spacing: .11em; text-transform: uppercase; }
    .hero p { color: #c9d8d1; max-width: 680px; font-size: 1rem; line-height: 1.55; margin: 0; }
    .hero-meta { display: flex; gap: 1.5rem; margin-top: 1.9rem; color: #c9d8d1; font-size: .78rem; }
    .hero-meta strong { color: #fff; display: block; font-size: .95rem; margin-top: .2rem; }
    [data-testid="stMetric"] { background: var(--paper); border: 1px solid var(--line); padding: 1rem 1.15rem; border-radius: 10px; box-shadow: 0 4px 14px rgba(20, 35, 40, .035); }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3 { color: var(--ink) !important; }
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p { color: var(--ink); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: var(--ink) !important; }
    .hero h1, .hero p, .hero .hero-meta { color: #f5f8f3 !important; }
    .hero .eyebrow { color: var(--sky) !important; }
    .hero .hero-meta strong { color: #ffffff !important; }
    .section-label { color: var(--green); font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; margin: 1.8rem 0 .55rem; }
    .status-panel { background: var(--paper); border: 1px solid var(--line); border-left: 5px solid var(--amber); padding: 1.1rem 1.2rem; border-radius: 10px; margin: .5rem 0 1.25rem; }
    .status-panel strong { color: var(--ink); font-family: 'Space Grotesk', sans-serif; }
    .status-panel span { color: var(--muted); font-size: .9rem; }
    .note { color: var(--muted); font-size: .82rem; line-height: 1.5; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { color: var(--muted); padding: .7rem 0; }
    .stTabs [aria-selected="true"] { color: var(--blue); }
    @media (max-width: 800px) { .block-container { padding: 1.2rem 1rem 2rem; } .hero { padding: 1.8rem 1.5rem 2rem; } .hero-meta { gap: .8rem; flex-wrap: wrap; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    frame = make_demo_dataset()
    validate_dataset(frame)
    return frame


@st.cache_resource
def load_model(frame):
    return train_model(frame)


frame = load_data()
result = load_model(frame)
frame["pressure_label"] = frame["high_pressure"].map({0: "Lower pressure", 1: "Higher pressure"})

st.markdown(
    """
                <div class="topline"><span class="brand">COMMUNITY / SUPPORT EXPLORER</span><span>Public interest data science · 2026 demo</span></div>
    <section class="hero">
            <div class="eyebrow">Portfolio project / public interest data science</div>
      <h1>Where could cost-of-living pressure be highest?</h1>
      <p>An explainable prioritisation view for local teams planning further research and support. Explore the model estimate, its drivers, and the uncertainty behind it.</p>
            <div class="hero-meta"><div>Coverage<strong>180 demo areas</strong></div><div>Model<strong>Explainable baseline</strong></div><div>Purpose<strong>Prioritise investigation</strong></div></div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='eyebrow'>Explore the model</div>", unsafe_allow_html=True)
    st.header("Select an area")
    selected = st.selectbox("Authority", frame["authority"].sort_values().tolist())
    st.divider()
    st.subheader("Model context")
    st.caption("Logistic regression with standardised aggregate indicators. Holdout evaluation uses 25% of the synthetic demo data.")
    st.warning("Demo data only. This tool must not be used to decide individual eligibility or access to services.")

row = frame.loc[frame["authority"] == selected].iloc[0]
probability = float(result.model.predict_proba(row[FEATURES].to_frame().T)[0, 1])
label = "Higher priority for investigation" if probability >= 0.5 else "Lower priority for investigation"

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.metric("Selected area risk", f"{probability:.0%}")
metric_b.metric("Model ROC-AUC", f"{result.metrics['roc_auc']:.2f}")
metric_c.metric("Average precision", f"{result.metrics['average_precision']:.2f}")
metric_d.metric("Areas above 50%", f"{(frame['high_pressure'].mean()):.0%}")
st.markdown(f'<div class="status-panel"><strong>{selected}</strong><br><span>{label} | {probability:.0%} estimated risk probability. This is a model estimate, not a measured fact.</span></div>', unsafe_allow_html=True)

overview_tab, detail_tab = st.tabs(["Risk landscape", "Authority detail"])
with overview_tab:
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Risk landscape")
        chart = px.scatter(
            frame, x="median_rent_income_ratio", y="unemployment_pct", color="pressure_label",
            size="food_insecurity_pct", hover_name="authority",
            color_discrete_map={"Lower pressure": "#356a8a", "Higher pressure": "#b54708"},
            labels={"median_rent_income_ratio": "Rent / income ratio", "unemployment_pct": "Unemployment (%)", "pressure_label": "Model group"},
            height=410,
        )
        chart.update_layout(plot_bgcolor="#fffefa", paper_bgcolor="#fffefa", legend_title_text="Synthetic target", font_family="DM Sans", margin={"l": 10, "r": 10, "t": 20, "b": 10})
        chart.update_traces(marker={"line": {"width": 1.2, "color": "#ffffff"}})
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("What moves the model?")
        importance = result.importance.copy()
        importance["feature"] = importance["feature"].str.replace("_", " ").str.title()
        bar = px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", color_discrete_sequence=["#005a8d"], height=390)
        bar.update_layout(plot_bgcolor="#fffefa", paper_bgcolor="#fffefa", font_family="DM Sans", xaxis_title="Mean permutation importance", yaxis_title="", margin={"l": 10, "r": 10, "t": 20, "b": 10})
        st.plotly_chart(bar, use_container_width=True)

with detail_tab:
    st.subheader(f"Why does the model estimate {probability:.0%} for {selected}?")
    explanation = explain_row(result, row)
    explanation["feature"] = explanation["feature"].str.replace("_", " ").str.title()
    explanation["direction"] = explanation["contribution"].map(lambda value: "Raises estimate" if value >= 0 else "Lowers estimate")
    explanation["contribution"] = explanation["contribution"].round(3)
    st.dataframe(explanation[["feature", "contribution", "direction"]], hide_index=True, use_container_width=True)
    with st.expander("Inspect selected area inputs"):
        display = row[FEATURES].rename(lambda value: value.replace("_", " ").title()).to_frame("Value")
        st.dataframe(display, use_container_width=True)

st.markdown('<p class="note">Method note: the included dataset is deterministic synthetic data created for demonstration. Replace it with versioned official statistics and validate locally before drawing conclusions.</p>', unsafe_allow_html=True)
