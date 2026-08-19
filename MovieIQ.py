import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from scipy import stats
import ast
from collections import Counter

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="MovieIQ — Film Success Predictor",
    page_icon="🎬",
    layout="wide"
)

# ── Load & prepare data ──────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    def parse_genres(g):
        try:
            items = ast.literal_eval(g)
            return [x['name'] for x in items] if isinstance(items, list) else []
        except:
            return []

    df['genres_list'] = df['genres'].apply(parse_genres)
    df['primary_genre'] = df['genres_list'].apply(lambda x: x[0] if x else 'Unknown')
    df['success'] = (df['revenue'] > df['budget']).astype(int)
    return df

@st.cache_resource
def train_model(df):
    features = ['budget', 'popularity', 'runtime', 'vote_average']
    X = df[features]
    y = df['success']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'cm': confusion_matrix(y_test, y_pred)
    }
    return model, metrics, features

df = load_data()
model, metrics, features = train_model(df)
all_genres = sorted(set(g for gl in df['genres_list'] for g in gl))

# ── SIDEBAR ──────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/clapper-board.png", width=70)
st.sidebar.title("🎬 MovieIQ Filters")

selected_genres = st.sidebar.multiselect(
    "Filter by Genre",
    options=all_genres,
    default=[]
)
min_vote = st.sidebar.slider("Minimum Vote Average", 0.0, 10.0, 0.0, 0.1)
show_only_success = st.sidebar.checkbox("Show only Successful Movies")

# Apply filters
filtered = df.copy()
if selected_genres:
    filtered = filtered[filtered['genres_list'].apply(
        lambda x: any(g in x for g in selected_genres))]
if min_vote > 0:
    filtered = filtered[filtered['vote_average'] >= min_vote]
if show_only_success:
    filtered = filtered[filtered['success'] == 1]

# ── HEADER ───────────────────────────────────────────────
st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.markdown("**Analyse and predict whether a movie will succeed based on budget, popularity, runtime, and votes.**")
st.markdown("---")

# ── KPI CARDS ────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📽️ Movies", f"{len(filtered):,}")
col2.metric("✅ Successful", f"{filtered['success'].sum():,}")
col3.metric("📈 Success Rate", f"{filtered['success'].mean()*100:.1f}%")
col4.metric("💰 Avg Budget", f"${filtered['budget'].mean()/1e6:.1f}M")
col5.metric("💵 Avg Revenue", f"${filtered['revenue'].mean()/1e6:.1f}M")

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🔬 Statistical Tests", "🤖 Model Results", "🎯 Predict a Movie"])

# ── TAB 1: EDA ────────────────────────────────────────────
with tab1:
    st.header("Exploratory Data Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Budget vs Revenue")
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = filtered['success'].map({1:'#2ecc71', 0:'#e74c3c'})
        ax.scatter(filtered['budget']/1e6, filtered['revenue']/1e6,
                   c=colors, alpha=0.5, s=25, edgecolors='none')
        max_b = filtered['budget'].max()/1e6
        ax.plot([0, max_b], [0, max_b], 'k--', linewidth=1.5, label='Break-even')
        ax.set_xlabel('Budget (USD Millions)')
        ax.set_ylabel('Revenue (USD Millions)')
        ax.set_title('Budget vs Revenue', fontweight='bold')
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#2ecc71',label='Successful'),
                            Patch(color='#e74c3c',label='Not Successful'),
                            plt.Line2D([0],[0],color='black',linestyle='--',label='Break-even')])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Green dots above the dashed break-even line are successful movies (revenue > budget).")

    with col_b:
        st.subheader("Success Rate by Genre")
        genre_data = []
        for g in all_genres:
            mask = filtered['genres_list'].apply(lambda x: g in x)
            sub = filtered[mask]
            if len(sub) >= 5:
                genre_data.append({'Genre': g, 'Success Rate': sub['success'].mean()*100,
                                   'Count': len(sub)})
        if genre_data:
            gdf = pd.DataFrame(genre_data).sort_values('Success Rate', ascending=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            bar_colors = ['#2ecc71' if r > 60 else '#e67e22' if r > 50 else '#e74c3c'
                          for r in gdf['Success Rate']]
            ax.barh(gdf['Genre'], gdf['Success Rate'], color=bar_colors, edgecolor='white')
            ax.set_xlabel('Success Rate (%)')
            ax.set_title('Success Rate by Genre', fontweight='bold')
            ax.axvline(x=60, color='black', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption("Green bars = success rate above 60%. Dashed line marks the 60% threshold.")

    st.subheader("Feature Distributions: Successful vs Not Successful")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat, title in zip(axes,
                                ['popularity','runtime','vote_average'],
                                ['Popularity','Runtime (min)','Vote Average']):
        s = filtered[filtered['success']==1][feat]
        f = filtered[filtered['success']==0][feat]
        if len(s) > 0 and len(f) > 0:
            bp = ax.boxplot([s, f], patch_artist=True, widths=0.5,
                             medianprops={'color':'black','linewidth':2})
            bp['boxes'][0].set_facecolor('#2ecc71')
            bp['boxes'][1].set_facecolor('#e74c3c')
            ax.set_xticklabels(['Successful','Not Successful'])
            ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Popularity shows the clearest separation between successful and unsuccessful movies.")

    st.subheader("Correlation Heatmap")
    corr = filtered[['budget','revenue','popularity','runtime','vote_average','success']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, square=True, ax=ax)
    ax.set_title('Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Note: Revenue is excluded from the model to prevent data leakage (it defines success).")

# ── TAB 2: STATS ─────────────────────────────────────────
with tab2:
    st.header("Statistical Tests")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("T-Test: Popularity vs Success")
        pop_s = df[df['success']==1]['popularity']
        pop_f = df[df['success']==0]['popularity']
        t_stat, p_val = stats.ttest_ind(pop_s, pop_f)

        st.write(f"**Null Hypothesis:** Mean popularity is the same for both groups")
        st.write(f"**T-statistic:** {t_stat:.4f}")
        st.write(f"**P-value:** {p_val:.6f}")
        if p_val < 0.05:
            st.success("✅ REJECT null hypothesis (p < 0.05) — Popularity IS significantly different between successful and unsuccessful movies")
        else:
            st.warning("❌ FAIL TO REJECT null hypothesis")
        st.write(f"Mean popularity — Successful: **{pop_s.mean():.2f}**")
        st.write(f"Mean popularity — Not Successful: **{pop_f.mean():.2f}**")

    with col2:
        st.subheader("Chi-Square: Genre vs Success")
        ct = pd.crosstab(df['primary_genre'], df['success'])
        chi2, p_chi, dof, _ = stats.chi2_contingency(ct)

        st.write(f"**Null Hypothesis:** Genre and success are independent")
        st.write(f"**Chi-square stat:** {chi2:.4f}")
        st.write(f"**Degrees of freedom:** {dof}")
        st.write(f"**P-value:** {p_chi:.6f}")
        if p_chi < 0.05:
            st.success("✅ REJECT null hypothesis — Genre IS associated with movie success")
        else:
            st.warning("❌ FAIL TO REJECT null hypothesis")

    st.markdown("---")
    st.info("**What is a p-value?** A p-value measures the probability of seeing your data by chance if the null hypothesis were true. A p-value < 0.05 means there is less than a 5% chance the result occurred randomly — we take this as evidence to reject the null hypothesis.")

# ── TAB 3: MODEL ─────────────────────────────────────────
with tab3:
    st.header("🤖 Random Forest Model Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.1f}%")
    col2.metric("🔍 Precision", f"{metrics['precision']*100:.1f}%")
    col3.metric("📡 Recall", f"{metrics['recall']*100:.1f}%")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Not Successful','Successful'],
                    yticklabels=['Not Successful','Successful'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Feature Importance")
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        colors_fi = ['#2ecc71','#3498db','#e67e22','#9b59b6']
        ax.barh(importances.index, importances.values, color=colors_fi, edgecolor='white')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("How Random Forest Works")
    st.markdown("""
    A **Random Forest** is an ensemble of 100 decision trees:
    1. Each tree is trained on a random sample of the data (bootstrap sampling)
    2. Each tree independently predicts "Success" or "Not Success" for a new movie
    3. The final prediction is the **majority vote** across all 100 trees
    4. This approach reduces overfitting and is more robust than a single decision tree
    """)

# ── TAB 4: PREDICTION ────────────────────────────────────
with tab4:
    st.header("🎯 Predict a Movie's Success")
    st.markdown("Enter the details of a movie below and MovieIQ will predict whether it will be successful.")

    col1, col2 = st.columns(2)
    with col1:
        budget_input = st.number_input("Budget (USD)", min_value=1_000_000,
                                        max_value=500_000_000, value=50_000_000, step=1_000_000)
        popularity_input = st.slider("Popularity Score", 0.0, 100.0, 50.0, 0.1)

    with col2:
        runtime_input = st.number_input("Runtime (minutes)", min_value=60,
                                         max_value=300, value=120, step=1)
        vote_input = st.slider("Expected Vote Average", 0.0, 10.0, 6.5, 0.1)

    st.markdown("---")
    if st.button("🎬 Predict Success", use_container_width=True):
        input_data = pd.DataFrame([[budget_input, popularity_input, runtime_input, vote_input]],
                                   columns=features)
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.success(f"## ✅ SUCCESSFUL")
            st.write(f"**Confidence: {probability[1]*100:.1f}%** — The model predicts this movie will earn more than its budget.")
        else:
            st.error(f"## ❌ NOT SUCCESSFUL")
            st.write(f"**Confidence: {probability[0]*100:.1f}%** — The model predicts this movie will not recoup its budget.")

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Budget", f"${budget_input/1e6:.1f}M")
        col_r2.metric("Popularity", f"{popularity_input:.1f}")
        col_r3.metric("Runtime", f"{runtime_input} min")
        col_r4.metric("Vote Avg", f"{vote_input:.1f}")

        st.progress(probability[1])
        st.caption(f"Success probability: {probability[1]*100:.1f}%")

# ── FOOTER ───────────────────────────────────────────────
st.markdown("---")
st.markdown("**MovieIQ** | Built with Python · scikit-learn · Streamlit | Dataset: movies.csv (2,000 films)")
