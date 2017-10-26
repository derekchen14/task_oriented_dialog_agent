from sklearn.feature_extraction.text import TfidfVectorizer
import os

DIRECTORY = '../../datasets/restaurants'
TRAIN_FILES = 'dialog-babi-task6-dstc2-trn.txt'
DEV_FILES = 'dialog-babi-task6-dstc2-dev.txt'
RESPOND_CANDIDATE_FILE = 'dialog-babi-task6-dstc2-candidates.txt'

# Build the tfidf vectors
files_list = [os.path.join(DIRECTORY, TRAIN_FILES), os.path.join(DIRECTORY, RESPOND_CANDIDATE_FILE)]



MAX_FEATURES = 2000
vectorizer = TfidfVectorizer(max_df=0.5, max_features=MAX_FEATURES,
                             min_df=1, stop_words='english',
                             use_idf=True)

if not os.path.exists(os.path.join(DIRECTORY, 'train_and_candidate.txt')):
    print 'Wait... Generating tf-idf train txt...'
    file_iter = []
    with open(os.path.join(DIRECTORY, 'train_and_candidate.txt'), 'wb') as outfile:
        for file in files_list:
            with open(file, 'rb') as f:
                outfile.write(f.read())


with open(os.path.join(DIRECTORY, 'train_and_candidate.txt')) as f:
    X = vectorizer.fit_transform(f)
# sparse matrix, [n_samples, n_features]. Tf-idf-weighted document-term matrix.


