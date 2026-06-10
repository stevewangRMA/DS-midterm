import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score

# ---------- Page config ----------
st.set_page_config(page_title="Medicare Spending Analysis", layout="wide")

# 6 care categories = our features X; they sum to ~ total spend (y)
FEATURES = ["hospital_snf", "physician", "outpatient", "home_health", "hospice", "dme"]
LABELS = {
    "hospital_snf": "Hospital & SNF",
    "physician": "Physician",
    "outpatient": "Outpatient",
    "home_health": "Home health",
    "hospice": "Hospice",
    "dme": "Medical equipment",
    "total_spend": "Total spend",
}

# ---------- Load data ----------
def load_data():
    df = pd.read_csv("medicare_county_2014_clean.csv")
    # fill the few missing category values with the column median
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())
    return df

# ---------- Train model ----------
def train_model(df):
    X = df[FEATURES]
    y = df["total_spend"]
    # 80% train / 20% test for an honest evaluation
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {
        "r2": r2_score(y_te, pred),
        "mae": mean_absolute_error(y_te, pred),
        "mse": mean_squared_error(y_te, pred),
        "evs": explained_variance_score(y_te, pred),
    }
    return model, metrics

df = load_data()
model, metrics = train_model(df)

# ---------- Sidebar: navigation ----------
st.sidebar.title("Medicare Spending Project")
page = st.sidebar.radio(
    "Go to page",
    ["1. Introduction (Business Case)", "2. Data Visualization", "3. Prediction Model"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: Dartmouth Atlas of Health Care, 2014 (via data.world)")


# =====================================================================
# Page 1: Introduction
# =====================================================================
if page.startswith("1"):
    st.title("Per-Enrollee Medicare Spending by U.S. County")
    st.subheader("The problem we are solving")
    st.markdown(
        """
        **Medicare** is the U.S. federal health insurance program for seniors, spending
        over a trillion dollars a year. Yet **counties** differ enormously in per-person
        cost: even after adjusting for age, sex and race, some counties spend under
        \\$5,000 per enrollee while others spend more than \\$15,000.

        **Business / societal question:** What drives a county's per-enrollee Medicare
        spending, and which counties spend far more than their care mix would predict
        (potential targets for efficiency review or audit)?

        We use a **linear regression** model to predict total per-enrollee spend from
        6 categories of care. This reveals the **main cost drivers** and lets us use
        **residuals** to flag "overspending" counties.
        """
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Counties", f"{len(df):,}")
    c2.metric("Avg total spend", f"\\${df['total_spend'].mean():,.0f}")
    c3.metric("Min spend", f"\\${df['total_spend'].min():,.0f}")
    c4.metric("Max spend", f"\\${df['total_spend'].max():,.0f}")
    c5.metric("States covered", f"{df['state'].nunique()}")

    st.subheader("What the dataset looks like")
    st.markdown(
        "Each row is a **county**. `total_spend` is the target (y); the 6 care categories "
        "are the features (X). All values are **per enrollee, age/sex/race-adjusted** dollars."
    )
    show = df[["county_name", "state", "enrollees", "total_spend"] + FEATURES].copy()
    st.dataframe(show.head(15), use_container_width=True)

    with st.expander("Column definitions"):
        st.markdown(
            """
            - **county_name / state**: county name / state
            - **enrollees**: number of Medicare enrollees in the county
            - **total_spend**: total per-enrollee reimbursement (Parts A & B) -- target y
            - **hospital_snf**: hospital + skilled nursing facility spend
            - **physician**: physician services spend
            - **outpatient**: outpatient facility spend
            - **home_health**: home health agency spend
            - **hospice**: hospice spend
            - **dme**: durable medical equipment (wheelchairs, oxygen, etc.) spend
            """
        )


# =====================================================================
# Page 2: Data Visualization
# =====================================================================
elif page.startswith("2"):
    st.title("Data Visualization: Key Insights")

    # --- Insight 1: distribution of total spend ---
    st.subheader("1. Distribution of per-enrollee total spend")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(df["total_spend"], bins=40, kde=True, color="#4C72B0", ax=ax)
    ax.axvline(df["total_spend"].mean(), color="red", linestyle="--",
               label=f"Mean ${df['total_spend'].mean():,.0f}")
    ax.set_xlabel("Total spend per enrollee ($)"); ax.set_ylabel("Number of counties"); ax.legend()
    st.pyplot(fig)
    st.caption("Most counties cluster at $8,000-$10,000, but a long right tail shows a few unusually high counties.")

    # --- Insight 2: average spend by category ---
    st.subheader("2. Where is the money spent?")
    means = df[FEATURES].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh([LABELS[c] for c in means.index], means.values, color="#55A868")
    for i, v in enumerate(means.values):
        ax.text(v, i, f" ${v:,.0f}", va="center")
    ax.set_xlabel("Average spend per enrollee ($)")
    st.pyplot(fig)
    st.caption("Hospital & SNF is the largest cost, followed by physician and outpatient -- these three dominate total spend.")

    # --- Insight 3: correlation heatmap ---
    st.subheader("3. Which category relates most strongly to total spend?")
    corr = df[["total_spend"] + FEATURES].corr()
    corr.index = [LABELS[c] for c in corr.index]
    corr.columns = [LABELS[c] for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    st.pyplot(fig)
    st.caption("Hospital & SNF has the strongest correlation with total spend -- the main cost driver.")

    # --- Insight 4: most / least expensive states ---
    st.subheader("4. Which states are most / least expensive per enrollee?")
    state_avg = df.groupby("state")["total_spend"].mean().sort_values()
    topbot = pd.concat([state_avg.head(10), state_avg.tail(10)])
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#55A868"] * 10 + ["#C44E52"] * 10
    ax.barh(topbot.index, topbot.values, color=colors)
    ax.set_xlabel("Average total spend per enrollee ($)")
    ax.set_title("Cheapest 10 states (green) vs Most expensive 10 (red)")
    st.pyplot(fig)

    # --- Insight 5: interactive scatter ---
    st.subheader("5. Explore: a category vs total spend")
    pick = st.selectbox("Pick a care category", FEATURES, format_func=lambda c: LABELS[c])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(df[pick], df["total_spend"], alpha=0.3, s=12, color="#4C72B0")
    ax.set_xlabel(f"{LABELS[pick]} spend per enrollee ($)")
    ax.set_ylabel("Total spend per enrollee ($)")
    r = df[pick].corr(df["total_spend"])
    ax.set_title(f"correlation r = {r:.2f}")
    st.pyplot(fig)


# =====================================================================
# Page 3: Prediction Model
# =====================================================================
elif page.startswith("3"):
    st.title("Prediction Model: Linear Regression")
    st.markdown(
        "Model: **total spend = w1·hospital + w2·physician + ... + w6·equipment + b**. "
        "Trained on 80% of counties, evaluated on the remaining 20%."
    )

    # --- Model performance (same metrics as the course sample) ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (test set)", f"{metrics['r2']:.3f}",
              help="1 = perfect; the features explain almost all of total spend")
    c2.metric("MAE", f"${metrics['mae']:,.0f}",
              help="Average dollar error of the prediction")
    c3.metric("MSE", f"{metrics['mse']:,.0f}",
              help="Mean squared error")
    c4.metric("Explained variance", f"{metrics['evs']*100:.1f}%",
              help="% of variance in total spend explained by the model")

    # --- Coefficients ---
    st.subheader("Cost drivers (regression coefficients)")
    coef = pd.Series(model.coef_, index=[LABELS[c] for c in FEATURES]).sort_values()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(coef.index, coef.values, color="#8172B3")
    ax.set_xlabel("Increase in total spend per +$1 in category")
    st.pyplot(fig)
    st.caption("All coefficients are close to 1: each category adds roughly 1:1 to total spend, as expected.")

    # --- Interactive predictor ---
    st.subheader("Try it: set the spends and predict the total")
    st.markdown("Move the sliders to define a hypothetical county; the model predicts its total per-enrollee spend.")
    cols = st.columns(3)
    inputs = {}
    for i, f in enumerate(FEATURES):
        lo, hi, med = float(df[f].min()), float(df[f].max()), float(df[f].median())
        inputs[f] = cols[i % 3].slider(LABELS[f], lo, hi, med)
    x_new = pd.DataFrame([inputs])[FEATURES]
    pred = model.predict(x_new)[0]
    st.success(f"Predicted total spend per enrollee: **${pred:,.0f}**")

    # --- Residual analysis: flag overspending counties (the "solve a problem" part) ---
    st.subheader("Which counties spend more than the model predicts?")
    st.markdown(
        "We predict each county's total spend, then look at actual − predicted (the residual). "
        "**A large residual** means the county spends well above what its care mix implies -- "
        "a prime target for cost review / audit."
    )
    work = df.copy()
    work["predicted"] = model.predict(work[FEATURES])
    work["residual_overspend"] = work["total_spend"] - work["predicted"]
    top = work.sort_values("residual_overspend", ascending=False).head(15)
    st.dataframe(
        top[["county_name", "state", "total_spend", "predicted", "residual_overspend"]]
        .rename(columns={"county_name": "County", "state": "State",
                         "total_spend": "Actual spend", "predicted": "Predicted",
                         "residual_overspend": "Residual (overspend)"})
        .round(0),
        use_container_width=True,
    )

    # --- Actual vs predicted scatter ---
    st.subheader("Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(work["total_spend"], work["predicted"], alpha=0.3, s=12, color="#4C72B0")
    lims = [work["total_spend"].min(), work["total_spend"].max()]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual total spend ($)"); ax.set_ylabel("Predicted total spend ($)")
    ax.legend()
    st.pyplot(fig)
    st.caption("The closer a point is to the red line, the better the prediction; points far above the line are overspending counties.")
