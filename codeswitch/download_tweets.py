
import json
import re
import numpy as np
import pandas as pd

from twitterscraper import query_tweets
from twitterscraper.main import JSONEncoder
from typing import List
from fasttext.FastText import _FastText as FastTextModel

from utils import load_fasttext_lid_model, load_most_common_words, sparse_to_dense

candidates = [
    'She still doesn’t know who to believe , elle a fait partir les 2 and un autre mec est venu pour elle so elle s’en fou mtn& mourad lied because he’s a jealous piece of shit',
    'I’m sorry but Nas deserve to stay putain mon meeeec il peut partir dit',
    'C est l heure de partir! There is always a right moment to say goodbye when you are unwelcome somewhere. It’s the right time here and there will be another place to be liked and loved! Boris is an idiot',
    'im hesitant to walk out of the gym this sweaty and in this weather , ME VOY A PARTIR UN PULMON',
    'Elles pourrait se partir une agence qui recrute que des femmes. Le MI6 upside down WB Willy Smith shit Illuminati program for more love of God I do not fuckin know what happen with my country but i keep to win when i #Skycall with all of you womens around the Little Globby',
    'Fais ses course sur Friends for sale avant de partir....'
]

if __name__ == '__main__':

    model = load_fasttext_lid_model()
    words = load_most_common_words('fr')
    num_languages = len(model.get_labels())

    query = ' OR '.join(words[:40])

    # query tweets and store in pd.DataFrame
    tweets = query_tweets(f'"{query}"', limit=300, lang='en')
    df = pd.read_json(json.dumps(tweets, cls=JSONEncoder))
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x))

    labels, probabilities = model.predict(df['text'].tolist(), k=50)
    all_labels, all_probabilities = sparse_to_dense(labels, probabilities, model)

    for t in df['text']:
        print(model.predict(t, k=5))
        print(t)
        print()
