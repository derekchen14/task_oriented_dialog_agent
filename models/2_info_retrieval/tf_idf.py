'''
This is the tf-idf baseline for the chatbot.
Candidates loaded from dialog-babi-task6-dstc2-candidates.txt
Test data loaded from 'dialog-babi-task6-dstc2-tst.txt'

'''
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.internal.data_io import parse_dialogue_QA, parse_candidates
from scipy.sparse import dok_matrix
import numpy as np

DIRECTORY = 'datasets/restaurants'
TEST_FILES = 'dialog-babi-task6-dstc2-tst.txt'
RESPOND_CANDIDATE_FILE = 'dialog-babi-task6-dstc2-candidates.txt'

# Step 0. Build the tfidf vectors
MAX_FEATURES = 2000
vectorizer = TfidfVectorizer(max_df=1.0, max_features=MAX_FEATURES,
                             min_df=1, stop_words='english',
                             use_idf=True)


# Step 1. Generate the tf-idf weighted matrix for the candidates
with open(os.path.join(DIRECTORY, RESPOND_CANDIDATE_FILE)) as f:
    candidates_tf_idf = vectorizer.fit_transform(f)
# sparse matrix, [n_samples, n_features]. Tf-idf-weighted document-term matrix.

# Put all candidates into a list (and generate a list of string for future lookup)
with open(os.path.join(DIRECTORY, RESPOND_CANDIDATE_FILE)) as f:
    candidates = parse_candidates(f.readlines())


# Load the dialog in format of [Dialogue 1, Dialogue 2...]
# Each dialog is in format of [(u1, u2, u3...), (c1, c2, c3...]
with open(os.path.join(DIRECTORY, TEST_FILES)) as f:
    qa, _ = parse_dialogue_QA(f.readlines(), False)



all_queries = []
total_queries = 0
# Count total queries
for dialog in qa:
    queries, reponses = dialog
    for query in queries:
        all_queries.append(query)
        total_queries += 1


# Now let's load the test data into a sparse matrix by counting the frequency of each word
# The weights have been already added to the candidates. Frequency vectors will work for the test data
S = dok_matrix((total_queries, candidates_tf_idf.shape[1]), dtype=np.float32)

for i in range(len(all_queries)):
    for word in all_queries[i].split():
        if word in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[word]
            S[i, idx] += 1

# Calculate the cosine similarity between the test data and the candidates
cosine_similarity_matrix = cosine_similarity(S, candidates_tf_idf)
best_candidate_idx = np.argmax(cosine_similarity_matrix, axis=1)


# Output the examples and calculate the accuracy per response
match = 0
curr_idx = -1
for dialog in qa:
    queries, reponses = dialog
    for i in range(len(queries)):
        curr_idx += 1
        print 'Questions: ', queries[i]
        print 'True response: ', reponses[i]
        print 'TF-IDF response: ', candidates[best_candidate_idx[curr_idx]][0]
        if candidates[best_candidate_idx[curr_idx]][0].split() == reponses[i].split():
            match += 1


print match * 1.0/len(all_queries)








