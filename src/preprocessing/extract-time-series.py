import pandas as pd
import numpy as np
import config

FEATURES = [
    "sku",                   #0
    "sold_quantity",         #1
    "minutes_active",        #2
    "premium",               #3
    "fulfillment",           #4
    "cross_docking",         #5
    "drop_off",              #6
    "free_shipping",         #7
    "normalized_price",      #8
    "product_sales",         #9
    "product_market_price",  #10
    "product_min_price",     #11
    "product_count",         #12
    "product_min_sales",     #13
    "product_market_share",  #14
    "active",                #15
    "family_sales",          #16
    "family_market_share",   #17
    "family_market_price",   #18
    "family_count",          #19
    "item_sales",            #20
    "item_count",            #21
    "item_mean_price",       #22
    "current_price"          #23
]

# Read the data in csv
df_meli = pd.read_parquet(config.DATASET_DIRECTORY + '/train_data.parquet')
df_meli['date'] = pd.to_datetime(df_meli['date'])

# Add "active" column, this will be filled with zeros for missing rows.
df_meli["active"] = df_meli["minutes_active"] != 0

# One-hot encode categorical values
print("One-hot enconding categorical values...")
df_meli["premium"] = df_meli["listing_type"] == "premium"

df_meli["fulfillment"] = df_meli["shipping_logistic_type"] == "fulfillment"
df_meli["cross_docking"] = df_meli["shipping_logistic_type"] == "cross_docking"
df_meli["drop_off"] = df_meli["shipping_logistic_type"] == "drop_off"
df_meli["free_shipping"] = df_meli["shipping_payment"] == "free_shipping"

df_meli.drop(columns=["shipping_logistic_type", "listing_type"], inplace=True)

# Normalize currency by dividing for the price mean of each site
print("Normalizing prices...")
unique_currencies = df_meli["currency"].unique()
for cur in unique_currencies:
    subset = df_meli[df_meli["currency"] == cur]
    mean_price = np.sum(subset["sold_quantity"] * subset["current_price"]) / np.sum(subset["sold_quantity"])
    df_meli.loc[df_meli["currency"] == cur,"normalized_price"] = df_meli[df_meli["currency"] == cur]["current_price"] / mean_price

df_meli.drop(columns=["currency"], inplace=True)

# Merge with static data
print("Merging static data...")
df_static = pd.read_json(config.DATASET_DIRECTORY + '/items_static_metadata_full.jl', lines=True)
df_meli = pd.merge(df_meli,df_static,on="sku",how="left")

# Convert types to reduce memory usage
df_meli["item_domain_id"] = pd.factorize(df_meli["item_domain_id"])[0]
df_meli["site_id"] = pd.factorize(df_meli["site_id"])[0]
df_meli["product_family_id"] = pd.factorize(df_meli["product_family_id"])[0]
df_meli["product_id"] = pd.factorize(df_meli["product_id"])[0]
df_meli["sku"] = df_meli["sku"].astype("int32")
df_meli["sold_quantity"] = df_meli["sold_quantity"].astype("int32")
df_meli["item_id"] = df_meli["item_id"].astype("int32")

# Compute total sales and price mean by product
print("Computing aggregates at product_id lvl")
by_product_total_sales = df_meli.groupby(["date","site_id","product_id"])["sold_quantity"].sum().reset_index(name ='product_sales')
by_product_mean_price = df_meli.groupby(["date","site_id","product_id"])["normalized_price"].mean().reset_index(name ='product_market_price')
by_product_count = df_meli.groupby(["date","site_id","product_id"])["sku"].nunique().reset_index(name ='product_count')

# Compute minimum price by product
by_product_min_price = df_meli.groupby(["date","site_id","product_id"])["normalized_price"].min().reset_index(name ='product_min_price')
by_product_min_sales = df_meli.groupby(["date","site_id","product_id"])["sold_quantity"].min().reset_index(name ='product_min_sales')

# Merge
print("Merging aggregates at product level")
df_meli = pd.merge(df_meli,by_product_total_sales,on=["date","site_id","product_id"],how="left")
df_meli = pd.merge(df_meli,by_product_mean_price,on=["date","site_id","product_id"],how="left")
df_meli = pd.merge(df_meli,by_product_min_price,on=["date","site_id","product_id"],how="left")
df_meli = pd.merge(df_meli,by_product_count,on=["date","site_id","product_id"],how="left")
df_meli = pd.merge(df_meli,by_product_min_sales,on=["date","site_id","product_id"],how="left")

df_meli["product_sales"].fillna(df_meli["sold_quantity"],inplace=True)
df_meli["product_count"].fillna(1,inplace=True)
df_meli["product_market_price"].fillna(df_meli["normalized_price"],inplace=True)
df_meli["product_min_price"].fillna(df_meli["normalized_price"],inplace=True)
df_meli["product_min_sales"].fillna(df_meli["sold_quantity"],inplace=True)

# Transform market and min price to relative.
df_meli["product_market_price"] = df_meli["product_market_price"] / df_meli["normalized_price"]
df_meli["product_min_price"] = df_meli["product_min_price"] / df_meli["normalized_price"]

# We can also compute the market share at a product level.
df_meli["product_market_share"] = df_meli["sold_quantity"] / df_meli["product_sales"].clip(1e-5)

# Compute similar things at the product_family_id level.
print("Computing aggregates at product_family_id lvl")
by_family_total_sales = df_meli.groupby(["date","site_id","product_family_id"])["sold_quantity"].sum().reset_index(name ='family_sales')
by_family_mean_price = df_meli.groupby(["date","site_id","product_family_id"])["normalized_price"].mean().reset_index(name ='family_market_price')
by_family_count= df_meli.groupby(["date","site_id","product_family_id"])["sku"].nunique().reset_index(name ='family_count')

# merge
print("Merging aggregates at product_family_id level")
df_meli = pd.merge(df_meli,by_family_total_sales,on=["date","site_id","product_family_id"],how="left")
df_meli = pd.merge(df_meli,by_family_mean_price,on=["date","site_id","product_family_id"],how="left")
df_meli = pd.merge(df_meli,by_family_count,on=["date","site_id","product_family_id"],how="left")

df_meli["family_sales"].fillna(df_meli["sold_quantity"],inplace=True)
df_meli["family_market_price"].fillna(df_meli["normalized_price"],inplace=True)
df_meli["family_count"].fillna(1,inplace=True)

# Transform family market price to relative.
df_meli["family_market_price"] = df_meli["family_market_price"] / df_meli["normalized_price"]

# Market share at the family level.
df_meli["family_market_share"] = df_meli["sold_quantity"] / df_meli["family_sales"].clip(1e-5)

# Aggregates at item_id lvl.
print("Computing aggregates at item_id lvl")
by_item_total_sales = df_meli.groupby(["date","site_id","item_id"])["sold_quantity"].sum().reset_index(name ='item_sales')
by_item_mean_price = df_meli.groupby(["date","site_id","item_id"])["normalized_price"].mean().reset_index(name ='item_mean_price')
by_item_count = df_meli.groupby(["date","site_id","item_id"])["sku"].nunique().reset_index(name ='item_count')

# Merge
print("Merging aggregates at item_id lvl")
df_meli = pd.merge(df_meli,by_item_total_sales,on=["date","site_id","item_id"],how="left")
df_meli = pd.merge(df_meli,by_item_mean_price,on=["date","site_id","item_id"],how="left")
df_meli = pd.merge(df_meli,by_item_count,on=["date","site_id","item_id"],how="left")

df_meli["item_mean_price"].fillna(df_meli["normalized_price"],inplace=True)
df_meli["item_sales"].fillna(df_meli["sold_quantity"],inplace=True)
df_meli["item_count"].fillna(1,inplace=True)

df_meli.drop(columns=["site_id","product_id","item_domain_id","item_title","product_family_id","item_id"],inplace=True)

# extract time series
min_date = np.min(df_meli["date"])
max_date = np.max(df_meli["date"])

unique_skus = df_meli["sku"].max()
print("Unique skus:" + str(unique_skus))

processed_count = 0
processed_skus = 0
numpy_series = np.zeros(
    (unique_skus, (max_date - min_date).days + 1, len(FEATURES)))

def group_to_numpy_series(df):
    global processed_count
    global numpy_series
    sku = df["sku"].iloc[0]
    idx = pd.date_range(min_date, max_date)
    # Fill missing dates with 0s and extract features.
    feature_series = df.set_index("date").reindex(idx, fill_value=0)[
        FEATURES].to_numpy().astype(np.float32)
    numpy_series[sku-1, :, :] = feature_series
    processed_count += 1
    # Show progress every 10k processed skus.
    if processed_count % 10000 == 0:
        print("\rProcessed skus:" + str(processed_count), end="")
    return df["sold_quantity"].iloc[0]

df_meli.groupby("sku").apply(group_to_numpy_series)
print("")
print("Saving series.")
np.save(config.TIME_SERIES_PATH, numpy_series)