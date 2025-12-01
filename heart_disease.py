import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


st.set_page_config(
    page_title="Heart Disease EDA",
    layout="wide"
)

def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("styles.css")


st.markdown(
    '<h1 class="main-title">Heart Disease Analysis Dashboard</h1>',
    unsafe_allow_html=True
)


heart = pd.read_csv(
    "heart.csv"
)


st.sidebar.header("Controls")
show_raw = st.sidebar.checkbox("Show raw data (head)", value=True)
target_value = st.sidebar.selectbox(
    "Filter by Target (Heart Disease)",
    options=["All", 0, 1],
    index=0
)


if target_value != "All":
    df = heart[heart["target"] == target_value]
else:
    df = heart.copy()


tab_overview, tab_uni, tab_bi_cat, tab_bi_cont, tab_corr, tab_outliers, tab_conclusion = st.tabs(
    [
        "Overview",
        "Univariate",
        "Bivariate (Categorical)",
        "Continuous vs Target",
        "Correlation / Multivariate",
        "Outlier Detection",
        "Conclusion"
    ]
)


with tab_overview:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric(
        "Heart Disease % (target=1)",
        f"{(heart['target'].mean() * 100):.1f}%"
    )

    if show_raw:
        st.write("### First 5 rows of dataset")
        st.dataframe(df.head())

    st.markdown('</div>', unsafe_allow_html=True)


with tab_uni:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Univariate Analysis")

    
    st.markdown("**Target Variable Distribution**")
    fig, ax = plt.subplots(figsize=(4, 2.6))
    sns.countplot(data=heart, x="target", ax=ax)
    ax.set_title("Distribution of Target Variable")
    ax.set_xlabel("Target (0 = No Disease, 1 = Disease)")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Age Distribution**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.histplot(data=df, x="age", kde=True, ax=ax, bins=20)
        ax.set_title("Age Distribution")
        st.pyplot(fig)

    with col2:
        st.markdown("**Maximum Heart Rate (thalach) Distribution**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.histplot(data=df, x="thalach", kde=True, ax=ax, bins=20)
        ax.set_title("Thalach Distribution")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


with tab_bi_cat:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Bivariate Analysis – Categorical Variables")

    
    st.markdown("**Target vs Sex (Facet Countplot)**")
    g = sns.catplot(
        x='target',
        col='sex',
        data=heart,
        kind='count',
        height=2.8,
        aspect=1,
        hue='target'
    )
    g.fig.suptitle("Target vs Sex (0 = Female, 1 = Male)", y=1.05)
    g.fig.set_size_inches(6, 3)
    st.pyplot(g.fig)
    plt.close(g.fig)

    
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Target Count (Set3 Palette)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.countplot(data=heart, x='target', palette='Set3', ax=ax)
        ax.set_title("Target Count (Set3)")
        ax.set_xlabel("Target")
        st.pyplot(fig)

    with c2:
        st.markdown("**Target Count (Transparent Bars with Dark Edges)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.countplot(
            data=heart,
            x='target',
            facecolor=(0, 0, 0, 0),
            linewidth=2,
            edgecolor=sns.color_palette('dark', 3),
            ax=ax
        )
        ax.set_title("Target Count (Edge Styling)")
        ax.set_xlabel("Target")
        st.pyplot(fig)

    
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**Target vs Fasting Blood Sugar (fbs)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.countplot(data=heart, x='target', hue='fbs', ax=ax)
        ax.set_title("Target vs fbs")
        ax.set_xlabel("Target")
        ax.legend(title="fbs (>120 mg/dl)")
        st.pyplot(fig)

    with c4:
        st.markdown("**Target vs Exercise Induced Angina (exang)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.countplot(data=heart, x='target', hue='exang', ax=ax)
        ax.set_title("Target vs exang")
        ax.set_xlabel("Target")
        ax.legend(title="exang (1 = Yes)")
        st.pyplot(fig)

   
    st.markdown("### Chest Pain Type (cp) vs Target")

    
    st.markdown("**cp Value Counts**")
    cp_counts = heart['cp'].value_counts().sort_index()
    st.dataframe(cp_counts.rename("count"))

    
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("**cp Count (Set1 Palette)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.countplot(data=heart, x='cp', palette='Set1', ax=ax)
        ax.set_title("cp Count")
        ax.set_xlabel("cp")
        st.pyplot(fig)

    with c6:
        st.markdown("**cp vs Target (Countplot with Hue)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.countplot(data=heart, x='cp', hue='target', ax=ax)
        ax.set_title("cp vs Target")
        ax.set_xlabel("cp")
        ax.legend(title="target")
        st.pyplot(fig)

    
    st.markdown("**cp vs Target (Groupby Table)**")
    cp_target = heart.groupby('cp')['target'].value_counts().unstack(fill_value=0)
    st.dataframe(cp_target)

   
    st.markdown("**Target Distribution Faceted by cp**")
    g_cp = sns.catplot(
        data=heart,
        x='target',
        col='cp',
        hue='target',
        kind='count',
        height=2.6,
        aspect=0.9
    )
    g_cp.fig.suptitle("Target vs cp (Facet Countplot)", y=1.05)
    g_cp.fig.set_size_inches(7, 3)
    st.pyplot(g_cp.fig)
    plt.close(g_cp.fig)

    st.markdown('</div>', unsafe_allow_html=True)


with tab_bi_cont:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Continuous Features vs Target")

    
    cont_cols = ["age", "trestbps", "chol", "oldpeak"]

    for col in cont_cols:
        st.markdown(f"**{col} vs Target**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.boxplot(data=heart, x="target", y=col, ax=ax)
        ax.set_title(f"{col} vs Target")
        st.pyplot(fig)

    
    st.markdown("### Detailed Analysis: thalach vs Target")

   
    thalach_unique = heart['thalach'].nunique()
    st.metric("Unique thalach values", thalach_unique)

   
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Stripplot – thalach vs Target (No Jitter)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.stripplot(data=heart, x='target', y='thalach', palette='Set2', ax=ax)
        ax.set_title("thalach vs Target (Stripplot)")
        st.pyplot(fig)

    with c2:
        st.markdown("**Stripplot – thalach vs Target (With Jitter)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.stripplot(
            data=heart,
            x='target',
            y='thalach',
            jitter=0.01,
            palette='Set2',
            ax=ax
        )
        ax.set_title("thalach vs Target (Jitter)")
        st.pyplot(fig)

   
    st.markdown("**Boxplot – thalach vs Target**")
    fig, ax = plt.subplots(figsize=(4, 2.6))
    sns.boxplot(data=heart, x='target', y='thalach', palette='Set1', ax=ax)
    ax.set_title("thalach vs Target (Boxplot)")
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


with tab_corr:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Correlation & Multivariate Analysis")

    
    st.markdown("**Correlation Heatmap of Numeric Features**")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    corr = heart.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    
    st.markdown("**Correlation of Features with Target (Sorted)**")
    corr_with_target = corr['target'].sort_values(ascending=False)
    st.dataframe(corr_with_target.rename("correlation_with_target"))

    
    st.markdown("### Pairplot of Numerical Variables (Colored by Target)")
    num_var = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
    g_pair = sns.pairplot(
        heart[num_var],
        kind='scatter',
        diag_kind='hist',
        hue='target',
        palette='cubehelix'
    )
    g_pair.fig.set_size_inches(6, 4)
    st.pyplot(g_pair.fig)
    plt.close(g_pair.fig)

  
    st.markdown("### Age – Summary Statistics & Distribution")

 
    st.markdown("**Age Summary (describe)**")
    age_desc = heart['age'].describe()
    st.dataframe(age_desc.to_frame(name="age_stats"))

    
    st.markdown("**Age Distribution (Approximately Normal)**")
    fig, ax = plt.subplots(figsize=(4, 2.6))
    sns.histplot(heart['age'], bins=10, kde=True, ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)

    c1, c2 = st.columns(2)

    with c1:
        
        st.markdown("**Stripplot – age vs Target**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.stripplot(
            data=heart,
            x='target',
            y='age',
            hue='target',
            dodge=True,
            ax=ax
        )
        ax.set_title("age vs Target (Stripplot)")
        st.pyplot(fig)

    with c2:
        
        st.markdown("**Boxplot – age vs Target**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.boxplot(
            data=heart,
            x='target',
            y='age',
            hue='target',
            ax=ax
        )
        ax.set_title("age vs Target (Boxplot)")
        st.pyplot(fig)

   
    st.markdown("### Relationships Between Age, trestbps, chol and thalach")

    
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("**age vs trestbps (Scatter)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.scatterplot(data=heart, x='age', y='trestbps', ax=ax)
        ax.set_title("age vs trestbps")
        st.pyplot(fig)

    with c4:
        st.markdown("**age vs trestbps (Regression)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.regplot(data=heart, x='age', y='trestbps', ax=ax)
        ax.set_title("age vs trestbps (Reg)")
        st.pyplot(fig)

    c5, c6 = st.columns(2)

    with c5:
        st.markdown("**age vs chol (Scatter)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.scatterplot(data=heart, x='age', y='chol', ax=ax)
        ax.set_title("age vs chol")
        st.pyplot(fig)

    with c6:
        st.markdown("**age vs chol (Regression)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.regplot(data=heart, x='age', y='chol', ax=ax)
        ax.set_title("age vs chol (Reg)")
        st.pyplot(fig)

    c7, c8 = st.columns(2)

    with c7:
        st.markdown("**chol vs thalach (Scatter)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.scatterplot(data=heart, x='chol', y='thalach', ax=ax)
        ax.set_title("chol vs thalach")
        st.pyplot(fig)

    with c8:
        st.markdown("**chol vs thalach (Regression)**")
        fig, ax = plt.subplots(figsize=(4, 2.6))
        sns.regplot(data=heart, x='chol', y='thalach', ax=ax)
        ax.set_title("chol vs thalach (Reg)")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


with tab_outliers:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Outlier Detection (Numeric Features)")

    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    for col in numeric_cols:
        st.markdown(f"### {col} – Summary & Boxplot")

        
        desc = heart[col].describe()
        st.markdown("**Summary Statistics**")
        st.dataframe(desc.to_frame(name=f"{col}_stats"))

        
        st.markdown("**Boxplot (Outlier Detection)**")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        sns.boxplot(x=heart[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


with tab_conclusion:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("📌 Conclusion")

    st.markdown("""
### **End of Exploratory Data Analysis**

Our EDA journey has come to an end. In this dashboard, we have explored the **heart disease dataset** in detail.

We have:

- Performed **univariate, bivariate and multivariate** analysis  
- Visualized the distribution of key clinical features  
- Explored the interaction between the **target variable** and features such as  
  `age`, `cp`, `thalach`, `chol`, `trestbps`, and `oldpeak`  
- Discussed and visualized **outliers** using boxplots  
- Inspected **correlations** and relationships using pairplots and regression plots  

The main feature of interest is the **target variable** (0 = no disease, 1 = disease).  
We analyzed it alone and also in combination with other variables to understand patterns that might indicate a higher risk of heart disease.

We also touched on how to **detect missing data and outliers**, which is a crucial step before moving to modeling.

---

### **Final Note**

Exploratory Data Analysis (EDA) gives us deep intuition about the data.  
These insights are the foundation for:

- **Feature engineering**  
- **Model selection**  
- **Predictive analytics and risk scoring**

🚀 You are now ready to move from **EDA → ML modeling** for heart disease prediction.
""")


    st.markdown('</div>', unsafe_allow_html=True)

