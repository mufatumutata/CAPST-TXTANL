import pandas as pd
import statsmodels.api as sm

# Load your dataset
df = pd.read_csv('/Users/mustafaelahi/Desktop/draft_2_code/dataset_with_sentiment.csv')

# Map sentiments to numerical values: POSITIVE -> 1, NEGATIVE -> 0
df['sentiment_num'] = df['sentiment'].map({'POSITIVE': 1, 'NEGATIVE': 0})

# Prepare the data
# Assuming 'sentiment' is already converted to 0s and 1s in 'sentiment_num'
X = df[['sentiment_num']]  # Predictor
y = df['stars']            # Response

# Add a constant to the predictor variable matrix (necessary for statsmodels)
X = sm.add_constant(X)

# Fit an OLS model
model = sm.OLS(y, X).fit()

# View the summary of the regression
print(model.summary())
