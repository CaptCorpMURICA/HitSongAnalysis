"""
Author:         Kevin Zufelt
Project:        hitSongAnalysis
File:           textBlob.py
Created:        12/5/2017, 10:02 PM
Description:    Tutorial on using the TextBlob Package
                https://textblob.readthedocs.io/en/dev/quickstart.html#create-a-textblob
                https://textblob.readthedocs.io/en/dev/advanced_usage.html#advanced
"""

from textblob import TextBlob

wiki = TextBlob("Python is a high-level, general-purpose programming language.")

print("=" * 50)

# Part-of-Speech Tagging
print(wiki.tags)

print("=" * 50)

# Noun Phrase Extraction
print(wiki.noun_phrases)

print("=" * 50)

# Sentiment Analysis
testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
print(testimonial.sentiment)
print(testimonial.sentiment.polarity)

print("=" * 50)

# Tokenization
zen = TextBlob("Beautiful is better than ugly. "
               "Explicit is better than implicit. "
               "Simple is better than complex.")
print(zen.words)
print(zen.sentences)

for sentence in zen.sentences:
    print(sentence.sentiment)

print("=" * 50)

# Words Inflection and Lemmatization
sentence = TextBlob('Use 4 spaces per indentation level.')
print(sentence.words)

print(sentence.words[2].singularize())
print(sentence.words[-1].pluralize())

from textblob import Word
w = Word("octopi")
print(w.lemmatize())

w = Word("went")
print(w.lemmatize("v")) # Pass in part of speech (verb)

print("=" * 50)

# WordNet Integration
from textblob import Word
from textblob.wordnet import VERB
word = Word("octopus")
print(word.synsets)
print(Word("hack").get_synsets(pos=VERB))
print(Word("octopus").definitions)

from textblob.wordnet import Synset
octopus = Synset('octopus.n.02')
shrimp = Synset('shrimp.n.03')
print(octopus.path_similarity(shrimp))

print("=" * 50)

# WordLists
animals = TextBlob("cat dog octopus")
print(animals.words)
print(animals.words.pluralize())

# Spelling Correction
b = TextBlob("I havv goood speling!")
print(b.correct())

from textblob import Word
w = Word('falibility')
print(w.spellcheck())

# Get Word and Noun Phrase Frequencies
monty = TextBlob("We are no longer the Knights who say Ni. "
                 "We are now the Knights who say Ekki ekki ekki PTANG.")
print(monty.word_counts['ekki'])
print(monty.words.count('ekki'))

# You can specify whether or not the search should be case-sensitive (default is False).
print(monty.words.count('ekki', case_sensitive=True))

print(wiki.noun_phrases.count('python'))

print("=" * 50)

# Translation and Language Detection
en_blob = TextBlob(u'Simple is better than complex.')
print(en_blob.translate(to='es'))

chinese_blob = TextBlob(u"美丽优于丑陋")
print(chinese_blob.translate(from_lang="zh-CN", to='en'))

b = TextBlob(u"بسيط هو أفضل من مجمع")
print(b.detect_language())

print("=" * 50)

# Parsing
b = TextBlob("And now for something completely different.")
print(b.parse())

print("=" * 50)

# TextBlobs Are Like Python Strings
print(zen[0:19])
print(zen.upper())
print(zen.find("Simple"))

apple_blob = TextBlob('apples')
banana_blob = TextBlob('bananas')
print(apple_blob < banana_blob)
print(apple_blob == 'apples')

print(apple_blob + ' and ' + banana_blob)
print("{0} and {1}".format(apple_blob, banana_blob))

print("=" * 50)

# n-grams
blob = TextBlob("Now is better than never.")
print(blob.ngrams(n=3))

print("=" * 50)

# Get Start and End Indices of Sentences
for s in zen.sentences:
    print(s)
    print("---- Starts at index {}, Ends at index {}".format(s.start, s.end))

print("=" * 50)
print("--- Advanced Techniques ---")
print("=" * 50)

# Sentiment Analysis

# The textblob.sentiments module contains two sentiment analysis implementations, PatternAnalyzer (based on the pattern
# library) and NaiveBayesAnalyzer (an NLTK classifier trained on a movie reviews corpus). The default implementation is
# PatternAnalyzer, but you can override the analyzer by passing another implementation into a TextBlob’s constructor.
# For instance, the NaiveBayesAnalyzer returns its result as a namedtuple of the form: Sentiment(classification, p_pos,
# p_neg).

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
blob = TextBlob("I love this library", analyzer=NaiveBayesAnalyzer())
print(blob.sentiment)

print("=" * 50)

# Tokenizers

# The words and sentences properties are helpers that use the textblob.tokenizers.WordTokenizer and
# textblob.tokenizers.SentenceTokenizer classes, respectively. You can use other tokenizers, such as those provided by
# NLTK, by passing them into the TextBlob constructor then accessing the tokens property.

from textblob import TextBlob
from nltk.tokenize import TabTokenizer
tokenizer = TabTokenizer()
blob = TextBlob("This is\ta rather tabby\tblob.", tokenizer=tokenizer)
print(blob.tokens)

# You can also use the tokenize([tokenizer]) method.

from textblob import TextBlob
from nltk.tokenize import BlanklineTokenizer
tokenizer = BlanklineTokenizer()
blob = TextBlob("A token\n\nof appreciation")
print(blob.tokenize(tokenizer))

print("=" * 50)

# Noun Phrase Chunkers

# TextBlob currently has two noun phrases chunker implementations, textblob.np_extractors.FastNPExtractor (default,
# based on Shlomi Babluki’s implementation from this blog post) and textblob.np_extractors.ConllExtractor, which uses
# the CoNLL 2000 corpus to train a tagger. You can change the chunker implementation (or even use your own) by
# explicitly passing an instance of a noun phrase extractor to a TextBlob’s constructor.

from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
extractor = ConllExtractor()
blob = TextBlob("Python is a high-level programming language.", np_extractor=extractor)
print(blob.noun_phrases)

print("=" * 50)

# POS Taggers

# TextBlob currently has two POS tagger implementations, located in textblob.taggers. The default is the PatternTagger
# which uses the same implementation as the pattern library. The second implementation is NLTKTagger which uses NLTK’s
# TreeBank tagger. Numpy is required to use the NLTKTagger. Similar to the tokenizers and noun phrase chunkers, you can
# explicitly specify which POS tagger to use by passing a tagger instance to the constructor.

from textblob import TextBlob
from textblob.taggers import NLTKTagger
nltk_tagger = NLTKTagger()
blob = TextBlob("Tag! You're It!", pos_tagger=nltk_tagger)
print(blob.pos_tags)

print("=" * 50)

# Parsers

from textblob import TextBlob
from textblob.parsers import PatternParser
blob = TextBlob("Parsing is fun.", parser=PatternParser())
print(blob.parse())

print("=" * 50)

# Blobber: A TextBlob Factory

# It can be tedious to repeatedly pass taggers, NP extractors, sentiment analyzers, classifiers, and tokenizers to
# multiple TextBlobs. To keep your code DRY, you can use the Blobber class to create TextBlobs that share the same
# models. First, instantiate a Blobber with the tagger, NP extractor, sentiment analyzer, classifier, and/or tokenizer
# of your choice.

from textblob import Blobber
from textblob.taggers import NLTKTagger
tb = Blobber(pos_tagger=NLTKTagger())
print(tb)

blob1 = tb("This is a blob.")
blob2 = tb("This is another blob.")
print(blob1.pos_tagger is blob2.pos_tagger)