import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for density plots

# Load dataset
df = pd.read_csv('/Users/mustafaelahi/Desktop/draft_2_code/dataset_with_sentiment.csv')

# Sample 1000 random rows from the dataset
df_sampled = df.sample(n=1000, random_state=1680)

# Convert 'sentiment' to numerical values: POSITIVE -> 1, NEUTRAL -> 0, NEGATIVE -> -1
sentiment_mapping = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
df_sampled['sentiment'] = df_sampled['sentiment'].map(sentiment_mapping)

# Set N to the number of clusters 
N = 4

# Instances for clustering methods
km = KMeans(n_clusters=N, random_state=1680)
hc = AgglomerativeClustering(n_clusters=N, linkage='ward')

# Define the pairs of features to analyze
pairs = [('sentiment', 'is_open'), ('sentiment', 'stars'), ('sentiment', 'review_stars')]

for x, y in pairs:
    # Prepare data for this pair
    data_for_pair = df_sampled[[x, y]]

    # KMeans clustering
    km_labels = km.fit_predict(data_for_pair)
    # Agglomerative clustering
    hc_labels = hc.fit_predict(data_for_pair)

    # Plotting KMeans with density
    plt.figure(figsize=(10, 6))
    # Scatter plot for clusters
    plt.scatter(df_sampled[x], df_sampled[y], c=km_labels, cmap='viridis', alpha=0.6, edgecolor='k')
    # Density plot
    sns.kdeplot(x=df_sampled[x], y=df_sampled[y], cmap="inferno", shade=True, bw_adjust=0.7, alpha=0.3)
    plt.title(f'K-Means Clustering on {x} vs. {y} with Density')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

    # Plotting Agglomerative Clustering with density
    plt.figure(figsize=(10, 6))
    # Scatter plot for clusters
    plt.scatter(df_sampled[x], df_sampled[y], c=hc_labels, cmap='plasma', alpha=0.6, edgecolor='k')
    # Density plot
    sns.kdeplot(x=df_sampled[x], y=df_sampled[y], cmap="plasma", shade=True, bw_adjust=0.7, alpha=0.3)
    plt.title(f'Agglomerative Clustering on {x} vs. {y} with Density')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
