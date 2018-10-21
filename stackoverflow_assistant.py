import os
import sys
sys.path.append("..")
import utils
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_test, vectorizer_path):
    
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),
                                       token_pattern='(\S+)')
    tfidf_vectorizer.fit(X_train)
    X_train = tfidf_vectorizer.transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)
    with open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)
    return X_train, X_test


sample_size = 200000

dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
stackoverflow_df = pd.read_csv('data/tagged_posts.tsv', sep='\t').sample(sample_size, random_state=0)

print(dialogue_df.head())

print(stackoverflow_df.head())

# simple text processing
dialogue_df['text'] = dialogue_df['text'].apply(utils.text_prepare)
stackoverflow_df['title'] = stackoverflow_df['title'].apply(utils.text_prepare)

X = np.concatenate([dialogue_df['text'].values, stackoverflow_df['title'].values])
y = ['dialogue'] * dialogue_df.shape[0] + ['stackoverflow'] * stackoverflow_df.shape[0]

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.9,random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

X_train_tfidf, X_test_tfidf = tfidf_features(X_train, X_test, './tfidf_vectorizer.pkl')
intent_recognizer = LogisticRegression(penalty='l2', C=10, random_state=0)
intent_recognizer.fit(X_train_tfidf, y_train)

# Check test accuracy.
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))

pickle.dump(intent_recognizer, open(utils.RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))

X = stackoverflow_df['title'].values
y = stackoverflow_df['tag'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))

vectorizer = pickle.load(open(utils.RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))

X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)

tag_classifier=OneVsRestClassifier(LogisticRegression(penalty='l2', C=5, random_state=0))
tag_classifier.fit(X_train_tfidf, y_train)

# Check test accuracy.
y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))
pickle.dump(tag_classifier, open(utils.RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))

starspace_embeddings, embeddings_dim = utils.load_embeddings('data/starspace_embedding.tsv')
posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')
counts_by_tag = posts_df.groupby(['tag']).count()

print(counts_by_tag)

counts_by_tag = posts_df['tag'].value_counts().to_dict()

print(counts_by_tag)

for tag, count in counts_by_tag.items():
    tag_posts = posts_df[posts_df['tag'] == tag]
    
    tag_post_ids = tag_posts['post_id'].values
    
    tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
    for i, title in enumerate(tag_posts['title']):
        tag_vectors[i, :] = utils.question_to_vec(title, starspace_embeddings, embeddings_dim)

    # Dump post ids and vectors to a file.
    filename = os.path.join(utils.RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'],'%s.pkl' % tag).replace("\\","/")
    pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))