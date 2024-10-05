import pandas as pd

# Load the already-filtered dataset entirely (filtered so that the data only has entries for restaurants)
df_businesses = pd.read_json('/Users/mustafaelahi/Desktop/ECON1680_CODE/YELP_filtered_businesses.json', lines=True)
df_businesses_filtered = df_businesses[['business_id', 'is_open', 'stars']]

# Initialize an empty DataFrame for the merged data
merged_df = pd.DataFrame()

# Function to process each chunk
def process_chunk(chunk, df_businesses_filtered):
    # Select and rename the required columns from the chunk
    chunk_filtered = chunk[['business_id', 'stars', 'text']].rename(columns={'stars': 'review_stars'})
    # Merge with the business data
    return pd.merge(df_businesses_filtered, chunk_filtered, on='business_id', how='inner')

# Load and process the larger dataset in chunks
chunk_size = 5000 
for chunk in pd.read_json('/Users/mustafaelahi/Desktop/YELP OPEN DATASET/yelp_academic_dataset_review.json', lines=True, chunksize=chunk_size):
    merged_chunk = process_chunk(chunk, df_businesses_filtered)
    merged_df = pd.concat([merged_df, merged_chunk], ignore_index=True)

# Save the merged dataset
merged_df.to_json('/Users/mustafaelahi/Desktop/merged_dataset.json', orient='records', lines=True)
