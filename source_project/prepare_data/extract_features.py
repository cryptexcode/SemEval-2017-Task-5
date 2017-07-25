import sys
sys.path.append('/raid/data/skar3/semeval/source/ml_semeval17/')
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from features import lexical, syntactic, writing_density, sentiments, embeddings, generic_field_vectorizer
import nltk
import config
import os
import traceback
from sklearn.externals import joblib
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
import json

tokenizer = RegexpTokenizer('\w+')
stemmer = PorterStemmer()


def preprocess(x):
    return x.replace('\n', ' ').replace('\r', '').replace('\x0C', '').lower()


# author :Suraj
features_dict = dict(

    # n-gram
    unigram=lexical.NGramTfidfVectorizer(ngram_range=(1, 1), tokenizer=nltk.word_tokenize,
                                         analyzer="word",
                                         stop_words='english', lowercase=True, min_df=1),
    bigram=lexical.NGramTfidfVectorizer(ngram_range=(2, 2), tokenizer=nltk.word_tokenize,
                                        analyzer="word",
                                        lowercase=True, min_df=1),
    trigram=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), tokenizer=nltk.word_tokenize,
                                         analyzer="word",
                                         lowercase=True, min_df=1),

    binary_unigram=lexical.NGramTfidfVectorizer(ngram_range=(1, 1), tokenizer=nltk.word_tokenize,
                                         analyzer="word", use_idf=False, smooth_idf=False,
                                         stop_words='english', lowercase=True, min_df=1),
    binary_bigram=lexical.NGramTfidfVectorizer(ngram_range=(2, 2), tokenizer=nltk.word_tokenize,
                                        analyzer="word", use_idf=False, smooth_idf=False,
                                        lowercase=True, min_df=1),
    binary_trigram=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), tokenizer=nltk.word_tokenize,
                                         analyzer="word", use_idf=False, smooth_idf=False,
                                         lowercase=True, min_df=1),

    char_tri=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), analyzer="char",
                                          lowercase=True, min_df=5),
    char_4_gram=lexical.NGramTfidfVectorizer(ngram_range=(4, 4), analyzer="char",
                                             lowercase=True, min_df=5),

    char_5_gram=lexical.NGramTfidfVectorizer(ngram_range=(5, 5), analyzer="char",
                                             lowercase=True, min_df=5),

    concepts=sentiments.SenticConceptsTfidfVectorizer(ngram_range=(1, 1), tokenizer=nltk.word_tokenize, analyzer="word",
                                                      lowercase=True, binary=True, use_idf=False),

    concepts_score=sentiments.SenticConceptsScores(),

    google_word_emb=embeddings.Word2VecFeatures(tokenizer=nltk.word_tokenize, analyzer="word",
                                                lowercase=True,
                                                model_name=config.WORD_EMBEDDING_VECTOR_PATH),


    cashtag = generic_field_vectorizer.SimpleVectorizer(field='cashtag', lowercase=False, use_idf=False, binary=True, smooth_idf=False, norm=False),
    source = generic_field_vectorizer.SimpleVectorizer(field='source', lowercase=False, use_idf=False, binary=True, smooth_idf=False, norm=False),
    company = generic_field_vectorizer.SimpleVectorizer(field='company', lowercase=False, use_idf=False, binary=True, smooth_idf=False, norm=False),

    labels=preprocessing.MultiLabelBinarizer()
)


class doc():
    def __init__(self, title, company, concepts, stemmed_concepts, cashtag, source):
        if isinstance(title, float):
            title = ''
        self.content = title
        self.company = company
        self.source = source
        self.cashtag = cashtag

        if isinstance(concepts, str):
            self.concepts = concepts
        else:
            self.concepts = ''

        # if isinstance(stemmed_concepts, str):
        #     self.concepts = stemmed_concepts
        # else:
        #     self.concepts = ''



def load_data():
    """
    Will Change for each project
    :return:
    """
    print("LOADING DATA")
    document_list = []

    # df = pd.read_csv('/raid/data/skar3/semeval/data/preprocessed/headline_train_trial_test_prs.csv')
    df = pd.read_csv('/raid/data/skar3/semeval/data/preprocessed/mb_train_trial_test_new_prs.csv')

    for index, row in df.iterrows():
        document_list.append(
            doc(row['text'], row['cashtag'], row['concepts'], row['stemmed_concepts'], row['cashtag'], row['source']))

            # headline
            # doc(row['text'], row['company'], row['concepts'], row['stemmed_concepts'], '', '') )

    print('===============================')

    return document_list


def extract_and_dump_features():
    """
    This method gets the list of active features in config.features_to_extract.
    Then it extract features with corresponding extractor and dumps to file.

    :return:
    """
    data = load_data()

    for feature_name in config.features_to_extract:
        try:
            print('Extracting Feature {}'.format(feature_name))
            extracted_feature = features_dict[feature_name].fit_transform(data)
            if not isinstance(extracted_feature, np.ndarray ):
                extracted_feature = extracted_feature.toarray()

            print('Extraction Complete of {}'.format(feature_name))
            # file_output_path = os.path.join(config.DUMPED_VECTOR_DIR, 'ner_headline_'+feature_name + '.pkl')
            file_output_path = os.path.join(config.DUMPED_VECTOR_DIR, 'mb_' + feature_name + '.pkl')
            joblib.dump(extracted_feature, file_output_path)
            print('Feature {} vectors are dumped to {}'.format(feature_name,file_output_path))
            print('=========================')
        except:
            print(print('FAILED Extracting Feature {}'.format(feature_name)))
            traceback.print_exc()


def extract_and_dump_numeric_csv_features():
    """
    For concept scores
    :return:
    """
    df = pd.read_csv('/raid/data/skar3/semeval/data/preprocessed/mb_train_trial_test_new_prs.csv')
    # df = pd.read_csv('/raid/data/skar3/semeval/data/preprocessed/microblog_train_trial_test_prs.csv')
    print(config.features_to_extract)
    for feature_name in config.features_to_extract:
        scores = []
        for index, row in df.iterrows():
            scores.append(row[feature_name])
        # joblib.dump(np.array(scores).reshape(-1,1), os.path.join(config.DUMPED_VECTOR_DIR, 'ner_headline_'+feature_name+'.pkl'))
        joblib.dump(np.array(scores).reshape(-1, 1),
                    os.path.join(config.DUMPED_VECTOR_DIR, 'mb_' + feature_name + '.pkl'))
        print(os.path.join(config.DUMPED_VECTOR_DIR, 'mb_'+feature_name+'.pkl'))


def extract_classes():
    # df = pd.read_csv('/raid/data/skar3/semeval/data/preprocessed/headline_train.csv')
    df = pd.read_csv('/raid/data/skar3/semeval/data/preprocessed/mb_train_trial_test_new_prs.csv')
    scores = []
    for index, row in df.iterrows():
        if str(row['sentiment']) == 'nan':
            print(row['sentiment'])
            print(index)
            break
        scores.append( row['sentiment'] )
    #
    joblib.dump(scores, os.path.join(config.DUMPED_VECTOR_DIR, 'mb_scores.pkl'))
    print(len(scores))
    # joblib.dump(scores, os.path.join(config.DUMPED_VECTOR_DIR, 'microblog_scores.pkl'))





if __name__ == '__main__':
    # extract_and_dump_numeric_csv_features()
    extract_and_dump_features()
    # load_data()
    # extract_classes()
    # assign_baseline_tags()