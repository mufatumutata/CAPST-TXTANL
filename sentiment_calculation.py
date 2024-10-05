import pandas as pd
from transformers import pipeline

# Path to the cleaned/condensed dataset
df_path = '/Users/mustafaelahi/Desktop/draft_2_code/condensed_dataset.csv'
# New path for the dataset with sentiment
new_df_path = df_path.replace('condensed_dataset.csv', 'dataset_with_sentiment.csv')

# Load the sentiment analysis pipeline, specifying truncation=True to automatically handle long texts
sentiment_pipeline = pipeline("sentiment-analysis", truncation=True)

# Process the file in chunks
chunk_size = 500 

# Initialize variables to track progress
processed_rows = 0
total_rows = sum(1 for row in open(df_path, 'r', encoding='utf-8')) - 1  # Total rows minus header row

for chunk in pd.read_csv(df_path, chunksize=chunk_size):
    # Apply sentiment analysis individually, with truncation handled by the pipeline
    sentiments = []
    for text in chunk['clean text']:
        if pd.isna(text) or text == '':
            sentiments.append(None)
        else:
            result = sentiment_pipeline(text)
            sentiments.append(result[0]['label'])
    # save sentiment as its own data entry 
    chunk['sentiment'] = sentiments

    # Determine if this is the first chunk
    if processed_rows == 0:
        chunk.to_csv(new_df_path, mode='w', index=False)
    else:
        chunk.to_csv(new_df_path, mode='a', index=False, header=False)
    
    processed_rows += len(chunk)
    rows_left = total_rows - processed_rows

    print(f"Processed a chunk of size {len(chunk)}. Rows left to process: {rows_left}")

if rows_left == 0:
    print("Sentiment analysis completed and saved to new dataset.")
else:
    print(f"Sentiment analysis incomplete. {rows_left} rows left unprocessed.")
