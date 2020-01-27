
import numpy as np

from os.path import join, dirname, basename
from pathlib import Path
from typing import List
from requests import get
from fasttext.FastText import load_model, _FastText as FastTextModel


def load_fasttext_lid_model(path: str = join(dirname(__file__), '..', 'models', 'ft_lid'),
                            url: str = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
                            ) -> FastTextModel:

    """ load fasttext lib model and download if necessary """

    name = basename(url)
    try:
        model = load_model(join(path, name))
    except ValueError:
        model_content = get(url).content
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(join(path, name), 'wb') as fh:
            fh.write(model_content)
        model = load_model(join(path, name))

    return model


def load_most_common_words(language: str,
                           path: str = join(dirname(__file__), '..', 'data', 'top_words'),
                           url: str = 'https://raw.githubusercontent.com/CodeBrauer/1000-most-common-words/master/'):

    """ Download lists of most common words.  See here: https://github.com/CodeBrauerjbkgyi fgr6h7eg31` """

    filenames = {
        'en': '1000-common-english-words.txt',
        'fr': '1000-most-common-french-words.txt',
    }

    data_file = join(path, filenames[language])
    try:
        with open(data_file, 'r') as fh:
            words = [word.strip() for word in fh.readlines()]
    except FileNotFoundError:
        # download data
        url = ''.join([url, filenames[language]])
        words = get(url).content.decode('utf-8').lower().split('\n')

        # write to file
        with open(data_file, 'w') as fh:
            for word in words:
                fh.write(f'{word}\n')

    return words


Labels = List[List[str]]
Probabilities = List[List[float]]


def sparse_to_dense(labels: Labels,
                    probabilities: Probabilities,
                    model: FastTextModel
                    ) -> (List[str], np.ndarray):

    """convert sparse FastText predictions to dense"""

    all_labels = np.array(model.get_labels())
    all_labels.sort()
    num_labels = len(all_labels)
    num_samples = len(labels)
    all_probabilities = np.zeros((num_samples, num_labels))

    for i, labels_, probabilities_ in zip(range(num_samples), labels, probabilities):
        # sort labels_ and probabilities_
        idx = np.argsort(labels_)
        print(labels_)
        print(idx)
        labels_ = labels_[idx]
        probabilities_ = probabilities_[idx]

        # get indices and insert into all_probabilities
        idx = np.searchsorted(all_labels, labels_)
        all_probabilities[i][idx] = probabilities_

    return all_labels, all_probabilities
