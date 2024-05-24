import pandas as pd

# very simp
def filter_group(group):
    # Calculate quantiles within each group
    c1 = group['price_per_sqr_meter'].quantile(0.95)

    filtered_group = group[(group['price_per_sqr_meter'] <= c1)] # & 

    c4 = filtered_group['price_per_sqr_meter'].quantile(0.01)
    
    final_filtered_group = filtered_group[filtered_group['price_per_sqr_meter'] > c4]
    
    return final_filtered_group

if __name__ == "__main__":

    data = pd.read_parquet("data/raw/house_price_data_20-05-2024.parquet")
    # make idempotent functions
    data = data.drop_duplicates()
    # further clean up some columns
    for column in ["home_size", "home_type", "municipality", "parish", "neighborhood"]:
        data[column] = data[column].str.capitalize().str.strip().str.rstrip(',')

    data = data.groupby(["municipality"]).apply(filter_group)
    # sanity check, there are no houses > T6
    data = data[data["home_size"].isin(['T2', 'T3', 'T4', 'T1', 'T5', 'T6', 'T0'])]
    data.to_parquet("data/processed/house_price_data_20-05-2024.parquet")