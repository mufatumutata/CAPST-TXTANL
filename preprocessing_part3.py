import pandas as pd

chunk_size = 5000  # Number of lines to process at a time
df_path = '/Users/mustafaelahi/Desktop/draft_2_code/cleaned_dataset.csv'
output_path_csv = '/Users/mustafaelahi/Desktop/draft_2_code/condensed_dataset.csv'  # the output file path and name for the new CSV

# Process the file in chunks
first_chunk = True
for chunk in pd.read_csv(df_path, chunksize=chunk_size):
    # Filter out rows where 'clean text' has more than 512 characters
    chunk_filtered = chunk[chunk['clean text'].str.len() <= 512]
    if first_chunk:
        chunk_filtered.to_csv(output_path_csv, mode='w', index=False)  # Write the first chunk and create the file
        first_chunk = False
    else:
        chunk_filtered.to_csv(output_path_csv, mode='a', index=False, header=False)  # Append subsequent chunks without header
