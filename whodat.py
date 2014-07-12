""" Messing around with scikit-learn.

Based on https://gist.github.com/ralphbean/4c2d4105ea2c7e407fb5
"""

import sys

import numpy as np
import scipy.sparse

import sklearn.linear_model
import sklearn.svm
import sklearn.metrics
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.utils.sparsefuncs

texts = {
    '7thchamberpart2.txt': 0,
    '7thchamber.txt': 0,
    'abettertomorrow.txt': 0,
    'aintnothintoeffwith.txt': 0,
    'ashighaswutangget.txt': 0,
    'bellsofwar.txt': 0,
    'blackshampoo.txt': 0,
    'bringdaruckus.txt': 0,
    'canitbeallsosimple.txt': 0,
    'cashstillrulesscaryhours.txt': 0,
    'clanindafront.txt': 0,
    'cream.txt': 0,
    'deadlymelody.txt': 0,
    'dogshit.txt': 0,
    'duckseason.txt': 0,
    'forheavenssake.txt': 0,
    'heaterz.txt': 0,
    'hellzwindstaff.txt': 0,
    'impossible.txt': 0,
    'intro.txt': 0,
    'itsyourz.txt': 0,
    'littleghettoboys.txt': 0,
    'maria.txt': 0,
    'methodman.txt': 0,
    'mysteryofchessboxin.txt': 0,
    'oldergods.txt': 0,
    'potus1.txt': 1,
    'potus2.txt': 1,
    'potus3.txt': 1,
    'potus4.txt': 1,
    'potus5.txt': 1,
    'potus6.txt': 1,
    'potus7.txt': 1,
    'projectsinternational.txt': 0,
    'protectyaneck.txt': 0,
    'reunited.txt': 0,
    'secondcoming.txt': 0,
    'severepunishment.txt': 0,
    'shame.txt': 0,
    'sunshower.txt': 0,
    'tearz.txt': 0,
    'thecity.txt': 0,
    'theclosing.txt': 0,
    'themgm.txt': 0,
    'theprojects.txt': 0,
    'triumph.txt': 0,
    'visionz.txt': 0,
    'wurevolution.txt': 0,
}

corpus = []
original_target = []
for fname, whodunnit in texts.items():
    with open('texts/' + fname, 'r') as f:
        content = f.read()
        corpus.append(content)
        original_target.append(whodunnit)

n_samples = len(corpus)
n_targets = len(set(texts.values()))

target = [[0 for i in range(n_targets)] for j in range(n_samples)]
for i in range(n_samples):
    target[i][original_target[i]] = 1.0

print "* shape of the corpus", len(corpus)

print "Convert text data into numerical vectors"
# http://scikit-learn.org/stable/modules/feature_extraction.html
vectorizer = sklearn.feature_extraction.text.CountVectorizer(
    stop_words='english',
    ngram_range=(1, 1),  #ngram_range=(1, 1) is the default
    dtype='double',
)
data = vectorizer.fit_transform(corpus)
print "* shape of the tfidf vectors", data.shape

# Save this to compute explained variance later
vectors = data

print "Reduce the dimensionality of the data"
pca = sklearn.decomposition.TruncatedSVD(n_components=50)
data = pca.fit_transform(data)

print "* shape of the pca components", data.shape
exp = np.var(data, axis=0)
full = sklearn.utils.sparsefuncs.mean_variance_axis0(vectors)[1].sum()
explained_variance_ratios = exp / full
confidence = sum(explained_variance_ratios)

if confidence < 0.8:
    print "explained variance ratio %f < 0.8.  Bailing." % confidence
    sys.exit(1)

print "Training a support vector machine on first half"
regression = sklearn.linear_model.LinearRegression()
regression.fit(data[:n_samples / 2], target[:n_samples / 2])

print "Now predict the value of on the second half"
expected = target[n_samples / 2:]
predicted = regression.predict(data[n_samples / 2:])

print(
    "Regression report for regression %s:\n%s\n"
    % (regression, sklearn.metrics.mean_squared_error(expected, predicted)))

print "expected"
print "--------"
print expected

print "predicted"
print "--------"
print predicted

for i in range(n_samples / 2):
    j = i + (n_samples / 2)
    fname, whodunnit = texts.items()[j]

    print "original:", fname, whodunnit
    print " ", expected[i]
    print " ", predicted[i]
