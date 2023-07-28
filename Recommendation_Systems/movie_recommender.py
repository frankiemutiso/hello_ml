import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def has_too_many_fields(row, expected_num_fields):
    return len(row.split(",")) > expected_num_fields


df = pd.read_csv(
    "Books_Data/Books.csv",
    encoding="latin-1",
    sep=";",
    on_bad_lines="skip",
    low_memory=False,
)

# Remove the entries with a duplicate book title from the dataset
df = df.drop_duplicates(subset="Book-Title")

# Select a sample size of 15000 to save of memory resources
sample_size = 15000

# Select a sample set from the dataset
# Random state is just a random number used to ensure the reproducibility of the sample set on subsequest requests
# Replace=False means that once a row is selected it's not returned to the dataset. Hence an item appears only once.
df = df.sample(n=sample_size, replace=False, random_state=490)

# Drop the index column
df = df.reset_index()
df = df.drop("index", axis=1)


# This step is to ensure that the author's name appears as a single name
def clean_author(author):
    return str(author).lower().replace(" ", "")


df["Book-Author"] = df["Book-Author"].apply(clean_author)
df["Book-Title"] = df["Book-Title"].str.lower()
df["Publisher"] = df["Publisher"].str.lower()

# When axis is 0 is means you intend to drop the entire row, when the axis is 1 you intend to drop the entire column
df2 = df.drop(
    ["ISBN", "Year-Of-Publication", "Image-URL-S", "Image-URL-M", "Image-URL-L"], axis=1
)

df2["data"] = df2[df2.columns[1:]].apply(
    lambda x: " ".join(x.dropna().astype(str)), axis=1
)

vect = CountVectorizer()
vectorized = vect.fit_transform(df2["data"])
similarities = cosine_similarity(vectorized)

df = pd.DataFrame(
    similarities, columns=df["Book-Title"], index=df["Book-Title"]
).reset_index()

# print(df.head())


input_book = "far beyond the stars (star trek deep space nine)"


recommendations = pd.DataFrame(df.nlargest(5, input_book)["Book-Title"])
recommendations = recommendations[recommendations["Book-Title"] != input_book]

file = open("output.txt", "w")

file.write(recommendations.to_string())
