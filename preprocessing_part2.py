import pandas as pd

chunk_size = 5000  # Number of lines to process at a time
df_path = '/Users/mustafaelahi/Desktop/draft_2_code/merged_dataset.json' #path to the dataset being used
output_path_csv = '/Users/mustafaelahi/Desktop/draft_2_code/cleaned_dataset.csv'  # the output file path and name for CSV

# clean the text by replacing a bunch of things with ' ' (space). also lowercasing and checking for links
def clean_text(text):
    text = text.replace('\r', ' ').replace('\n', ' ').replace('. ', ' ').replace(', ', ' ')
    tokens = text.split()
    tokens = [token for token in tokens if "https:" not in token]
    return ' '.join(tokens).lower()

# Process the file in chunks and write to CSV
first_chunk = True
for chunk in pd.read_json(df_path, lines=True, chunksize=chunk_size):
    chunk['clean text'] = chunk['text'].apply(clean_text)
    chunk.drop(columns=['text'], inplace=True)  # Drop the 'text' column
    if first_chunk:
        chunk.to_csv(output_path_csv, mode='w', index=False)  # Write the first chunk and create the file
        first_chunk = False
    else:
        chunk.to_csv(output_path_csv, mode='a', index=False, header=False)  # Append subsequent chunks without header
