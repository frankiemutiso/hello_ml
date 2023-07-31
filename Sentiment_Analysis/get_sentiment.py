from sentiment_classifier import SentimentClassifier

tweet = input("Enter a tweet: ")

if len(tweet) == 0:
    print("Error: Empty string")

classifier = SentimentClassifier()

print(classifier.classify(tweet))
