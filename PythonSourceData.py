# Downloading a text dataset for investigation
# Code reused from;
# Source: https://scikit-learn.org/stable/auto_examples/text/plot_document_ \
# clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause


from sklearn.datasets import fetch_20newsgroups
import numpy as np




# #############################################################################
# Load some categories from the training set
categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
# Uncomment the following to do the analysis on all the categories
# categories = None

print("Loading 20 newsgroups dataset for categories:")
print(categories)


dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)

print("%d documents retrieved." % len(dataset.data))
print("%d categories retrieved." % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]

# #############################################################################


#dataset is a 'Bunch' object - https://scikit-learn.org/stable/modules/ \
#generated/sklearn.utils.Bunch.html
# Bunches are an extension of dictionaries, allowing keys to be called lie
# a.b instead of a['b']

import random

dataset.keys()
#['data', 'filenames', 'target_names', 'target', 'DESCR', 'description']
random.sample(dataset.data, 1)
random.sample(dataset.target_names, 1)

