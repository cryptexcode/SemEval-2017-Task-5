import os
import socket

machine = socket.gethostname()

ROOT_DIR = os.path.dirname(__file__)
print(ROOT_DIR)

DUMPED_VECTOR_DIR_HL = '/raid/data/skar3/semeval/vectors_hl_new/'
DUMPED_VECTOR_DIR = '/raid/data/skar3/semeval/vectors_mb_new/'
TRUE_LABELS_PATH = '/raid/data/skar3/semeval/vectors_mb_new/mb_scores.pkl'
# TRUE_LABELS_PATH = '/raid/data/skar3/semeval/vectors/headline_scores.pkl'

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, 'processed_data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'outputs/')
WORD_EMBEDDING_VECTOR_PATH = '/raid/data/skar3/GoogleNews-vectors-negative300.bin.gz'
VOCABULARY_WORD_TO_INDEX_LIST = '/raid/data/skar3/semeval/data/preprocessed/headline_vocabulary_to_index.json'
VOCABULARY_INDEX_TO_WORD_LIST = '/raid/data/skar3/semeval/data/preprocessed/headline_index_to_word.json'

features_to_extract = [
    'unigram',
    'bigram',
    'trigram',
    'binary_unigram',
    'binary_bigram',
    'binary_trigram',
    'char_tri',
    'char_4_gram',
    'char_5_gram',
    'two_skip_3_grams',
    'two_skip_2_grams',
    'concepts',
    # 'stemmed_concepts',
    'google_word_emb',
    'cashtag',
    'source'
    # 'polarity',
    # 'sensitivity',
    # 'attention',
    # 'aptitude',
    # 'pleasantness',
    # 'stemmed_polarity',
    # 'stemmed_sensitivity',
    # 'stemmed_attention',
    # 'stemmed_aptitude',
    # 'stemmed_pleasantness',
    # 'company'

]

features_to_use = [
    'unigram',
    'bigram',
    'trigram',
    'binary_unigram',
    'binary_bigram',
    'binary_trigram',
    'char_tri',
    'char_4_gram',
    'char_5_gram',
    'two_skip_3_grams',
    'two_skip_2_grams',
    'concepts',
    'stemmed_concepts',
    'google_word_emb',
    'polarity',
    'sensitivity',
    'attention',
    'aptitude',
    'pleasantness',
    'stemmed_polarity',
    'stemmed_sensitivity',
    'stemmed_attention',
    'stemmed_aptitude',
    'stemmed_pleasantness',
    'company'
]

features_to_use_mb = [
    'unigram',
    'bigram',
    'trigram',
    'binary_unigram',
    'binary_bigram',
    'binary_trigram',
    'char_tri',
    'char_4_gram',
    'char_5_gram',
    'two_skip_3_grams',
    'two_skip_2_grams',
    'concepts',
    'stemmed_concepts',
    'google_word_emb',
    'cashtag',
    'source',
    'polarity',
    'sensitivity',
    'attention',
    'aptitude',
    'pleasantness',
    'stemmed_polarity',
    'stemmed_sensitivity',
    'stemmed_attention',
    'stemmed_aptitude',
    'stemmed_pleasantness',
]
