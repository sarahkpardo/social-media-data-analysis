# Twitter Information Operations Data Exploration

A set of Jupyter notebooks demonstrating pre-processing pipelines which provide a foundation for application of further analysis of [Twitter's information operations datasets](https://transparency.twitter.com/en/reports/information-operations.html). Contains utility procedures for correct handling of data types, cleaning tweet and user bio text strings, and querying dataset for relevant subsets of tweets. These notebooks also show basic usage for data exploration, retrieving summary statistics and visualizing various patterns and distributions.

# Installation

The conda environment file can be used to create an environment with the required modules using

```
conda create -n twitter_env --file twitter-env.txt
```

# To Do

- Compare campaign-associated tweets against a baseline of a more general set of tweets from the same time periods
- Cross-campaign analysis
- Apply clustering algorithms (e.g. k-means) to users and tweets
- More advanced algorithm such as BERT for tokenizing tweet text
- Add SQL database import/export
- Interface with deep learning library APIs
- More extensive plotting and visualization
