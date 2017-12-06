"""
Author:         CaptCorpMURICA
Project:        hitSongAnalysis
File:           hitSongAnalysis.py
Created:        12/4/2017, 8:39 PM
Description:    Perform sentiment analysis on the billboard top 100 songs from 1964-2015.
"""

# http://www.cs.cornell.edu/people/pabo/movie-review-data/

import pandas as pd
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

data = pd.read_excel('billboard_lyrics_1964-2015.xlsx')
# print(data.head())

# print(data.iloc[0].Rank)
# print(data.iloc[0].Song)
# print(data.iloc[0].Artist)
# print(data.iloc[0].Year)
# print(data.iloc[0].Lyrics)

data['NumChars'] = data['Lyrics'].str.len()

# Create polarity and subjectivity for sentiment analysis
i = 0
for lyric in data['Lyrics']:
    text = TextBlob(str(lyric))
    data.loc[i, 'polarity'] = text.sentiment.polarity
    data.loc[i, 'subjectivity'] = text.sentiment.subjectivity
    naiveBayes = TextBlob(str(text), analyzer=NaiveBayesAnalyzer()).sentiment
    data.loc[i, 'classification'], data.loc[i, 'p_pos'], data.loc[i, 'p_neg'] = naiveBayes
    i += 1

print(data.head())

# print(data.iloc[4000])

# print(TextBlob(data.loc[4000, 'Lyrics'], analyzer=NaiveBayesAnalyzer()).sentiment)

# x = TextBlob(data.loc[4000, 'Lyrics'], analyzer=NaiveBayesAnalyzer()).sentiment
# classification, p_pos, p_neg = x
# print("Classification: {}".format(classification))
# print("Positive Polarity: {}".format(p_pos))
# print("Negative Polarity: {}".format(p_neg))
