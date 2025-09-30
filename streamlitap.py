import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules,apriori, fpmax, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer

# =====================
# Streamlit UI
# =====================
st.title("🛒 Market Basket Analysis with Apriori")

# Upload CSV
df_raw= pd.read_csv("https://raw.githubusercontent.com/Excitedicecream/CSV-Files/refs/heads/main/Groceries_dataset.csv")
st.dataframe(df_raw)


grouped_df = df_raw.groupby(["Member_number"])["itemDescription"].apply(list).reset_index()

st.subheader("Group Dataframe")
st.dataframe(grouped_df)

MultiLabelBinarizer = MultiLabelBinarizer()
df = pd.DataFrame(MultiLabelBinarizer.fit_transform(grouped_df["itemDescription"]), columns=MultiLabelBinarizer.classes_, index=grouped_df.index)
st.subheader("One-Hot Encoded Dataframe")
st.dataframe(df)



frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False)
max_len = frequent_itemsets['length'].max() 
st.subheader("Frequent Itemsets")
st.dataframe(frequent_itemsets)
st.write('From this we pull out what is the most common item and which of those items are commonly bought together')

st.subheader('commonly bought items')
top_item=frequent_itemsets[frequent_itemsets['length'] == 1]
st.dataframe(top_item)


frequent_itemsets_2 = fpgrowth(df, min_support=0.01, use_colnames=True)

st.dataframe(frequent_itemsets_2)
rules = association_rules(frequent_itemsets_2, metric="lift", min_threshold=1.2)
st.subheader("Association Rules")
st.dataframe(rules)
