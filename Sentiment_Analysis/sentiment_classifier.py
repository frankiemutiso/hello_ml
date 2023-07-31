from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import joblib

import random
import re, string


class SentimentClassifier:
    def __init__(self):
        self.positive_tweets = twitter_samples.strings("positive_tweets.json")
        self.negative_tweets = twitter_samples.strings("negative_tweets.json")

        self.positive_tweets_tokens = twitter_samples.tokenized("positive_tweets.json")
        self.negative_tweets_tokens = twitter_samples.tokenized("negative_tweets.json")

        self.stop_words = stopwords.words("english")
        self.classifier = None

        self.filename = "SentimentClassifier.joblib"

    def clean_sentences(self, tokens, stop_words=()):
        lemmatizer = WordNetLemmatizer()
        cleaned = []

        for token, tag in pos_tag(tokens):
            token = re.sub(
                "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|"
                "(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                token,
            )

            token = re.sub("(@[A-Za-z0-9_]+)", "", token)

            if tag.startswith("NN"):
                positive_tweets = "n"
            elif tag.startswith("VB"):
                positive_tweets = "v"
            else:
                positive_tweets = "a"

            token = lemmatizer.lemmatize(token, positive_tweets)

            if (
                len(token) > 0
                and token not in string.punctuation
                and token.lower() not in stop_words
            ):
                cleaned.append(token.lower())

        return cleaned

    def get_all_words(self, tokens_list):
        for tokens in tokens_list:
            for token in tokens:
                yield token

    def get_tweets_for_model(self, cleaned_tokens):
        for tokens in cleaned_tokens:
            yield dict([token, True] for token in tokens)

    def build_train_test_dataset(self):
        cleaned_positive_tweets_tokens = []
        cleaned_negative_tweets_tokens = []

        for line in self.positive_tweets_tokens:
            cleaned_positive_tweets_tokens.append(
                self.clean_sentences(line, self.stop_words)
            )

        for line in self.negative_tweets_tokens:
            cleaned_negative_tweets_tokens.append(
                self.clean_sentences(line, self.stop_words)
            )

        positive_tweets_tokens_for_model = self.get_tweets_for_model(
            cleaned_positive_tweets_tokens
        )
        negative_tweets_tokens_for_model = self.get_tweets_for_model(
            cleaned_negative_tweets_tokens
        )
        positive_tweetsitive_dataset = [
            (tweet_dict, "Positive") for tweet_dict in positive_tweets_tokens_for_model
        ]
        negative_tweetsative_dataset = [
            (tweet_dict, "Negative") for tweet_dict in negative_tweets_tokens_for_model
        ]

        dataset = positive_tweetsitive_dataset + negative_tweetsative_dataset

        random.shuffle(dataset)

        train_data = dataset[:7000]
        test_data = dataset[7000:]

        return (train_data, test_data)

    def train(self):
        (train, test) = self.build_train_test_dataset()

        classifier = NaiveBayesClassifier.train(train)

        return classifier

    def classify(self, tweet):
        classifier = self.train()

        cleaned_tweet = self.clean_sentences(word_tokenize(tweet))

        return classifier.classify(dict([token, True] for token in cleaned_tweet))
