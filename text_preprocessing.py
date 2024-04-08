import fasttext
import sys, os
import time
import re, string
import nltk
import spacy

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from stop_words import get_stop_words

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# # spaCy model download for English
nlp = spacy.load("en_core_web_lg")

# Words avoid for lemmatization
skip_words = ["BOWLING"]

# Debug or not debug
debug = not ('gunicorn' in os.environ.get('SERVER_SOFTWARE', ''))

# Initialize the WordNet Lemmatizer from nltk
wordnet_lemmatizer = WordNetLemmatizer()

# Define a function to get the WordNet POS tag for a word
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Define a function to check if a string is nonsense
def is_string_nonsense(string):
    if len(string) < 3:
        return True
    return False

# Define a function to truncate a string to a maximum length
def truncate_string(string, max_length):
    if len(string) <= max_length:
        return string
    trimmed_string = string[:max_length + 1]
    last_space = trimmed_string.rfind(' ')
    if last_space != -1:
        return trimmed_string[:last_space]
    return trimmed_string[:max_length]

# Define a function to transliterate non-Latin characters
def transliterate_non_latin(string):
    string = string.replace('´', "'")
    string = string.replace('\'S ', ' ')
    string = string.replace('Â´', "'")
    string = re.sub(r'[^\x20-\x7E]', '', string)
    return string

# Define a function to lemmatize words in a text using nltk
def lemmatize_words(text):
    words = nltk.word_tokenize(text)
    lemmatized_words = [wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(word)) if word not in skip_words else word for word in words]
    return ' '.join(lemmatized_words)

def lemmatize_with_spacy(text):
    # text to spaCy instance
    doc = nlp(text)
    # Lemmatization every token and concatenate them into string
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    return lemmatized_text

# Define a function to remove stop words from a text using nltk
def remove_stop_words(text):
    stop_words = set(get_stop_words('en'))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def remove_words_with_digits(text):
    # Replacing words with digits
    cleaned_text = re.sub(r'\b\w*\d\w*\b', '', text)
    return cleaned_text

# Define a function to prepare a description
def prepare_description(description):
    description = transliterate_non_latin(description.upper())
    description = remove_words_with_digits(description)
    description = lemmatize_with_spacy(description.lower())
    description = description.upper()
    description = re.sub(r'\d+', ' ', description)  # Replacing all digits by space
    description = re.sub(r'[^\w\s]', ' ', description, flags=re.UNICODE)  # Remove all symbols instead letters, digits and spaces
    description = re.sub(r'\s+', ' ', description)  # Replacing several spaces by one space
    description = truncate_string(description, 255)
    description = lemmatize_words(description)
    description = remove_stop_words(description)
    description = description.replace('T SHIRT', 'TSHIRT')
    description = re.sub(r'\b[a-z]\b', '', description, flags=re.IGNORECASE)
    description = re.sub(r'\s+', ' ', description)
    return description

def get_categories_dict(path_to_categories='./categories.txt'):
  categories = {}
  with open(f'./categories.txt', 'r') as f:
      lines = f.readlines()
      for line in lines:
          categories[line.split('\t')[0]] = line.split('\t')[1].strip()
  return categories