import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.preprocessing import MultiLabelBinarizer
import streamlit as st

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Market Basket Analysis", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    df_raw = pd.read_csv(
        "https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/Groceries_dataset.csv"
    )
    grouped_df = df_raw.groupby("Member_number")["itemDescription"].apply(list).reset_index()
    mlb = MultiLabelBinarizer()
    df = pd.DataFrame(
        mlb.fit_transform(grouped_df["itemDescription"]),
        columns=mlb.classes_,
        index=grouped_df.index,
    )
    return df_raw, grouped_df, df

df_raw, grouped_df, df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Apriori", "Association Rules"])

# --- Parameters ---
st.sidebar.header("⚙️ Parameters")
min_support = st.sidebar.slider("Min Support", 0.001, 0.1, 0.01, 0.001)
min_conf = st.sidebar.slider("Min Confidence", 0.5, 1.0, 0.75, 0.05)
min_lift = st.sidebar.slider("Min Lift", 1.0, 3.0, 1.2, 0.1)

# =====================
# Page 1: Apriori
# =====================
if page == "Apriori":
    st.title("🛒 Apriori Frequent Itemsets")

    with st.expander("📂 Raw Data", expanded=False):
        st.dataframe(df_raw.head(20))

    with st.expander("👥 Grouped Transactions", expanded=False):
        st.dataframe(grouped_df.head(20))

    with st.expander("🔠 One-Hot Encoded Data", expanded=False):
        st.dataframe(df.head(20))

    @st.cache_data
    def run_apriori(data, min_support):
        freq_items = apriori(data, min_support=min_support, use_colnames=True)
        freq_items["length"] = freq_items["itemsets"].apply(len)
        freq_items["itemsets_str"] = freq_items["itemsets"].apply(lambda x: ", ".join(list(x)))
        return freq_items.sort_values("support", ascending=False)

    frequent_itemsets = run_apriori(df, min_support)

    st.subheader("📊 Frequent Itemsets")
    st.dataframe(frequent_itemsets[["support", "itemsets_str", "length"]].head(20))

    st.subheader("🥇 Top Single Items")
    top_items = frequent_itemsets[frequent_itemsets["length"] == 1]
    st.bar_chart(top_items.set_index("itemsets_str")["support"].head(10))

# =====================
# Page 2: Association Rules
# =====================
elif page == "Association Rules":
    st.title("🔗 Association Rules (FP-Growth)")

    @st.cache_data
    def run_fpgrowth_rules(data, min_support, min_conf, min_lift):
        freq_items = fpgrowth(data, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
        rules = rules[(rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)]
        return rules.sort_values("lift", ascending=False)

    rules = run_fpgrowth_rules(df, min_support, min_conf, min_lift)

    st.subheader("Rules")
    st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(20))
