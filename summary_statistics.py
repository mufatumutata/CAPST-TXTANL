import pandas as pd

data = pd.read_csv('/Users/mustafaelahi/Desktop/draft_2_code/dataset_with_sentiment.csv')

# Creating DataFrame
df = pd.DataFrame(data)

# Calculating Summary Statistics for 'is_open', 'stars', and 'review_stars'
summary_stats = df[['is_open', 'stars', 'review_stars']].describe().transpose()

# Adding Median to the summary
summary_stats['median'] = df[['is_open', 'stars', 'review_stars']].median()

# Calculating sentiment counts
sentiment_counts = df['sentiment'].value_counts()

# Preparing the final summary table
final_summary = summary_stats[['count', 'mean', 'std', 'min', '50%', 'max']]
final_summary.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Median', 'Max']
final_summary['Positive Sentiments'] = sentiment_counts.get('POSITIVE', 0)
final_summary['Negative Sentiments'] = sentiment_counts.get('NEGATIVE', 0)

# Display the final summary
print(final_summary)
