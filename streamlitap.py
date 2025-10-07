import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st
import plotly.express as px

# =====================
# Streamlit Setup
# =====================
st.set_page_config(page_title="Market Basket Analysis", layout="wide")

# --- Load Data ---
@st.cache_data(show_spinner=True)
def load_data():
    df_raw = pd.read_csv(
        "https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/Groceries_dataset.csv")
    grouped_df = (
        df_raw.groupby(["Date", "Member_number"])["itemDescription"]
        .apply(list)
        .reset_index()
    )

    mlb = MultiLabelBinarizer()
    df = pd.DataFrame(
        mlb.fit_transform(grouped_df["itemDescription"]),
        columns=mlb.classes_,
        index=grouped_df.index,
    )
    return df_raw, grouped_df, df

df_raw, grouped_df, df = load_data()

# =====================
# Sidebar
# =====================
st.sidebar.title("⚙️ Navigation & Settings")
page = st.sidebar.radio("Go to", ["Apriori", "Association Rules"])

st.sidebar.markdown("---")
st.sidebar.header("📈 Algorithm Parameters")

# Show sliders conditionally
if page == "Apriori":
    min_support_apriori = st.sidebar.slider(
        "Apriori Min Support",
        0.001, 0.1, 0.01, 0.001,
        help="Minimum frequency of item combinations to be considered frequent. Lower values = more itemsets, but possibly less meaningful."
    )

elif page == "Association Rules":
    min_support_fp = st.sidebar.slider(
        "FP-Growth Min Support",
        0.001, 0.1, 0.02, 0.001,
        help="Minimum frequency threshold for FP-Growth algorithm."
    )
    min_confidence = st.sidebar.slider(
        "Min Confidence",
        0.1, 1.0, 0.7, 0.05,
        help="How often the consequent appears when the antecedent appears. Higher confidence = stronger rule reliability."
    )
    min_lift = st.sidebar.slider(
        "Min Lift",
        1.0, 3.0, 1.2, 0.1,
        help="Indicates how much more likely items are to occur together than by random chance. Lift > 1 means a positive association."
    )

# Recommended settings note
with st.sidebar.expander("💡 Recommended Settings", expanded=False):
    st.markdown(
        """
        ✅ **Suggested values for best results:**
        - Apriori Min Support: **0.01**
        - FP-Growth Min Support: **0.02**
        - Min Confidence: **0.7**
        - Min Lift: **1.2**

        These settings balance rule quantity and quality well for most datasets.
        """
    )


st.sidebar.markdown("---")
st.sidebar.header("👤 About the Creator")
st.sidebar.markdown(
    """
**Jonathan Wong Tze Syuen**  
📚 Data Science  

🔗 [Connect on LinkedIn](https://www.linkedin.com/in/jonathan-wong-2b9b39233/)

🔗 [Connect on Github](https://github.com/Excitedicecream)
"""
)

# =====================
# Dataset Overview
# =====================
with st.expander("📊 Dataset Summary"):
    st.write(f"**Total Transactions:** {len(grouped_df)}")
    st.write(f"**Unique Items:** {df.shape[1]}")
    st.write(f"**Date Range:** {df_raw['Date'].min()} → {df_raw['Date'].max()}")

# =====================
# Page 1: Apriori
# =====================
if page == "Apriori":
    st.title("🛍️ Market Basket Analysis — Apriori Algorithm")

    with st.expander("📂 Raw Data", expanded=False):
        st.dataframe(df_raw.head(20))

    with st.expander("👥 Grouped Transactions", expanded=False):
        st.dataframe(grouped_df.head(20))

    with st.expander("🔠 One-Hot Encoded Data", expanded=False):
        st.dataframe(df.head(20))

    st.subheader("🔍 Frequent Itemsets using Apriori")

    @st.cache_data(show_spinner=True)
    def run_apriori(data, min_support):
        freq_items = apriori(data, min_support=min_support, use_colnames=True)
        freq_items["length"] = freq_items["itemsets"].apply(len)
        freq_items["itemsets_str"] = freq_items["itemsets"].apply(lambda x: ", ".join(list(x)))
        return freq_items.sort_values("support", ascending=False, ignore_index=True)

    if st.button("Run Apriori"):
        with st.spinner("Finding frequent itemsets..."):
            frequent_itemsets = run_apriori(df, min_support_apriori)

        st.success(f"Found {len(frequent_itemsets)} itemsets!")
        st.dataframe(frequent_itemsets.head(20))

        # Visualization
        top_items = frequent_itemsets.head(20)
        fig = px.bar(
            top_items,
            x="itemsets_str",
            y="support",
            title="Top Frequent Itemsets (Apriori)",
            labels={"itemsets_str": "Items", "support": "Support"},
        )
        st.plotly_chart(fig, use_container_width=True)

# =====================
# Page 2: Association Rules (FP-Growth)
# =====================
elif page == "Association Rules":
    st.title("🔗 Association Rules — FP-Growth Algorithm")

    @st.cache_data(show_spinner=True)
    def run_fpgrowth_rules(data, min_support, min_conf, min_lift):
        freq_items = fpgrowth(data, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
        rules = rules[(rules["confidence"] >= min_conf)]
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        return rules.sort_values("lift", ascending=False, ignore_index=True)

    if st.button("Run FP-Growth"):
        with st.spinner("Generating association rules..."):
            rules = run_fpgrowth_rules(df, min_support_fp, min_confidence, min_lift)

        st.success(f"Generated {len(rules)} rules!")
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(20))

        # Visualization
        fig2 = px.scatter(
            rules,
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            hover_data=["antecedents", "consequents"],
            title="Association Rules (Support vs Confidence)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            """
            **How to read this chart:**
            - Each point = one association rule  
            - **Support** = frequency of the rule  
            - **Confidence** = likelihood of consequents appearing with antecedents  
            - **Lift** > 1 means a strong positive relationship
            """
        )

