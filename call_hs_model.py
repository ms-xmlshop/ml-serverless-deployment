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
from text_preprocessing import prepare_description

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# spaCy model download for English
nlp = spacy.load("en_core_web_lg")

# Words avoid for lemmatization
skip_words = ["BOWLING"]

# Debug or not debug
debug = not ('gunicorn' in os.environ.get('SERVER_SOFTWARE', ''))

# Initialize the WordNet Lemmatizer from nltk
wordnet_lemmatizer = WordNetLemmatizer()

def get_categories_dict():
    categories = {}
    with open(f'./data/categories.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            categories[line.split('\t')[0]] = line.split('\t')[1].strip()
    return categories 

def call_hs2_model(input_text, predict_k=1):
    text = prepare_description(input_text)
    model_dir = '/root/external_copy/2'
    model = fasttext.load_model(f'{model_dir}/hs_model.bin')
    results = model.predict(text, k=predict_k)
    response_data = []
    categories = get_categories_dict()
    used_codes = []
    for i in range(len(results[0])):
        label = results[0][i].split('__l__')[1][0:2]
        probability = round(results[1][i], 4)

        if label in used_codes:
            continue
        used_codes.append(label)
        response_data_append = {
            'probability': probability,
            'code': label.strip(),
            'name': categories[label].strip(),
        }
        response_data.append(response_data_append)
    return response_data


if __name__ == '__main__':
    res = call_hs2_model('cotton hoodie')
    print(res)