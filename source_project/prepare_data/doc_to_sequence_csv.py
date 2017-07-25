import sys

sys.path.append('/raid/data/skar3/semeval/source/ml_semeval17/')
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import config
import codecs
import pandas as pd

MAX_NB_WORDS = 5000


def get_vocabulary_size(data):
    words = set()

    for doc in data:
        tokens = doc.split()
        for t in tokens:
            words.add(t)

    print('\n'.join(sorted(words)))
    print(len(words))


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = codecs.open(gloveFile, 'r', encoding='latin-1').read().split('\n')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        # print(word)
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def convert_into_sequences(df, colname):
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df[colname].values)
    sequences = tokenizer.texts_to_sequences(df[colname].values)
    sl = max(list(map(lambda x: len(x), sequences)))
    print(sl)
    # print(df[colname].values[:10])
    print(sequences[:10])
    print(max(tokenizer.word_index.values()), len(tokenizer.word_index.values()))

    lens = list(map(lambda s: len(s), sequences))
    print(len(lens), max(lens))
    data = pad_sequences(sequences, maxlen=sl)
    print(data[:10])

    #
    #
    #  Write dictionaries to json file
    #  with open(config.VOCABULARY_WORD_TO_INDEX_LIST, 'w') as f:
    #      json.dump(tokenizer.word_index, f)
    #      f.close()
    # #
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    #  with open(config.VOCABULARY_INDEX_TO_WORD_LIST, 'w') as f:
    #      json.dump(index_to_word, f)
    #      f.close()
    #
    #
    # Load word embeddings and create embedding matrix
    print('Loading Embeddings')
    model = Word2Vec.load_word2vec_format(config.WORD_EMBEDDING_VECTOR_PATH, binary=True, encoding='utf-8')
    # model = loadGloveModel(config.WORD_EMBEDDING_VECTOR_PATH)

    print('Loaded')
    print(model['try'])
    #
    nf = set()
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
    counter = 0
    for word, i in tokenizer.word_index.items():

        embedding_vector = np.zeros(300)
        if word in model:
            embedding_matrix[i] = model[word]
            counter += 1
        else:
            nf.add(word)
            embedding_matrix[i] = embedding_vector

    print(len(nf))
    print(len(data))
    print(len(embedding_matrix))
    # joblib.dump(embedding_matrix, config.DUMPED_VECTOR_DIR+'mb_voc_embeddings.pkl')
    # joblib.dump(data, config.DUMPED_VECTOR_DIR+'mb_sequences.pkl')

    print(len(data))
    print(len(embedding_matrix))
    print(data.shape)
    print(embedding_matrix.shape)
    print(counter)

    print(nf)
    return None, None


# def metadata_to_emb_vectors():
#
#     for f in config.features_to_use:
#         transformed_vector = joblib.load(config.DUMPED_VECTOR_DIR + f + '.pkl')


if __name__ == '__main__':
    df = pd.read_csv('/raid/data/skar3/semeval/data/raw/headline_train_trial_test.csv', encoding='utf-8')
    convert_into_sequences(df, 'text')
    # # get_vocabulary_size(data)
    # convert_into_sequences( data )

    # metadata_to_emb_vectors()
