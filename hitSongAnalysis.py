"""
Author:         CaptCorpMURICA
Project:        hitSongAnalysis
File:           hitSongAnalysis.py
Created:        12/4/2017, 8:39 PM
Description:    Perform sentiment analysis on the billboard top 100 songs from 1964-2015.
"""

import pandas as pd
from textblob import TextBlob

data = pd.read_excel('billboard_lyrics_1964-2015.xlsx')
print(data.head())

print(data.iloc[0].Rank)
print(data.iloc[0].Song)
print(data.iloc[0].Artist)
print(data.iloc[0].Year)
print(data.iloc[0].Lyrics)

data['NumChars'] = data['Lyrics'].str.len()
print(data.head())

# Create polarity and subjectivity for sentiment analysis
i = 0
for lyric in data['Lyrics']:
    text = TextBlob(str(lyric))
    data.loc[i:(i + 1), 'polarity'] = text.sentiment.polarity
    data.loc[i:(i + 1), 'subjectivity'] = text.sentiment.subjectivity
    i += 1
