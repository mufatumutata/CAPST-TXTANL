Title/Research Q: "Impact of Online Review Sentiments on Restaurant Ratings and Longevity"

Description: 

preprocessing_part1.py :- Uses the 'YELP_filtered_businesses.json' file--which is already filtered to only contain restaurant data (from the previous ML project)--and the 'yelp_academic_dataset_review.json' to create one merged file with all the required and relevant data.

preprocessing_part2.py :- Cleans the review text by preprocessing: lowercasing and removing \r and \n etc. Creates a new column in the dataset called 'clean text' that replaces the 'text' column

preprocessing_part3.py :- Condenses the dataset by only including entries (rows) that have 'clean text' that is less than 512 characters; otherwise, sentiment analysis model is not able to classify data.

ols.py :- Runs an OLS regression on the sentiment and stars; to use OLS, we map positive sentiment to 1 and negatie sentiment to 0.

summary_statistics.py :- calculates the summary statistics shown in the data section of the research. Basically counts the number of negative and positive sentiments and calculates the defining characteristics of the dataset we are working with.

sentiment_calculation.py :- Calculates the sentiments for each of the reviews (either POSITIVE or NEGATIVE).

sentiment_clustering.py :- Uses K-Means and Agglomerative clustering to find patterns in the data between the sentiments, stars, and business continuity.

Bugs: None
