import os
import sys
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu1,floatX=float32"
from keras.engine import Model

sys.path.append('/raid/data/skar3/semeval/source/ml_semeval17/')
from keras.wrappers.scikit_learn import KerasRegressor
import theano
import json
from sklearn import linear_model
from time import gmtime, strftime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import make_scorer
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Embedding, Convolution1D, Dropout, recurrent, LSTM, GRU, Merge
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Convolution1D, Dropout, GlobalMaxPooling1D, recurrent
from keras.layers import MaxPooling1D, Flatten, Bidirectional, GRU, TimeDistributed, Conv1D, Reshape
from keras.engine.topology import Layer, merge, Input
from keras.models import Sequential
from keras import initializations
from keras import backend as K
import config
from sklearn.externals import joblib
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import numpy as np
from evaluation import report
import warnings
warnings.filterwarnings("ignore")

global X_DIM, Y_DIM
max_features = 3306
embedding_dims = 300
max_len = 18
PREDICTION_THRESHOLD = 0.5
NB_EPOCH = 10
BATCH_SIZE = 32
MERGE = False
N_Features = 96179

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.init((input_shape[-1], ), name='{}_W_a'.format(self.name))
        self.trainable_weights += [self.W]
        super(AttLayer, self).build(input_shape)

    # print(len(input_shape), 'input >>>>>>>>>>>>>>>>>>>>>>>')
    #     assert len(input_shape) == 3
    #     # self.W = self.init((input_shape[-1],1))
    #     self.W = self.init((input_shape[-1],))
    #     # self.input_spec = [InputSpec(shape=input_shape)]
    #     self.trainable_weights = [self.W]
    #     super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])



def combine_features( feature_list ):
    return np.concatenate( (feature_list), axis=1 )



#===================================================================================
# Deep Learning
#===================================================================================
def deep_learning_cv(X_train, y_train, k, X_dev, y_dev, X_test, company_train, company_dev, company_test):
    global X_DIM, Y_DIM
    X_DIM = X_train[0].shape[0]
    y_train = np.array(y_train)
    print(X_train.shape, y_train.shape)
    Y_DIM = 1
    y_test = np.array(y_dev).reshape(1,-1)

    kfold_cv = KFold(n_splits=k, shuffle=True, random_state=7)
    fold_counter = 1
    cos_train = []
    cos_test = []
    cos_trial = []

    print('LSTM, Epoch', NB_EPOCH)
    print('Fold\t Training\t \t Test \t\t\t\t Trial')
    print('------------------------------------------------------------------------------------')
    for train_index, test_index in kfold_cv.split(X_train):
        print('Size', len(train_index), len(test_index))
        # print('Running Fold {}'.format(fold_counter))
        # print('------------------------------------------')
        print('Fold {}'.format(fold_counter), end='\t')

        X_tr, X_ts = X_train[train_index], X_train[test_index]
        y_tr, y_ts = y_train[train_index], y_train[test_index]
        C_tr, C_ts = company_train[train_index], company_train[test_index]

        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        ]
        # regressor = KerasRegressor(build_fn=attention_imp_merge_exp_paper, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE, verbose=1)
        # regressor = LinearSVR(C=0.1
        regressor = linear_model.ElasticNet(random_state=7)
        # )
        # print('Training Model')
        if MERGE:
            regressor.fit([X_tr, C_tr, X_tr], y_tr, callbacks=callbacks)
        else:
            # regressor.fit(X_tr, y_tr, callbacks=callbacks)
            regressor.fit(C_tr, y_tr)

        # print('Predicting Tests')
        if MERGE:
            predictions = regressor.predict([X_ts, C_ts, X_ts])
            # test_predictions = regressor.predict(X_dev)
            predictions = np.array(predictions).reshape(1,-1)
            train_predictions = regressor.predict([X_tr, C_tr, X_tr])
            trial_predictions = regressor.predict([X_dev, company_dev, X_dev])
        else:
            predictions = regressor.predict(C_ts)
            # test_predictions = regressor.predict(X_dev)
            predictions = np.array(predictions).reshape(1, -1)
            train_predictions = regressor.predict(C_tr)
            # trial_predictions = regressor.predict(X_dev)

        cos_train.append( cosine_similarity(y_tr, train_predictions)[0][0] )
        cos_test.append(cosine_similarity(y_ts, predictions)[0][0])
        # cos_trial.append(cosine_similarity(y_test, trial_predictions)[0][0])

        print('{}\t{}'.format(cos_train[-1], cos_test[-1]))
        # print("Cosine Similarity for Training fold ", cos_train[-1])
        # print("Cosine Similarity for Test fold ", cos_test[-1])
        # print("Cosine Similarity for Trial Data ", cos_trial[-1])

        fold_counter += 1

    print('---------------')
    print('Final Result')
    print('Training fold cosine mean : ', np.mean(cos_train))
    print('Testing fold cosine mean : ', np.mean(cos_test))
    print('Trial fold cosine mean : ', np.mean(cos_trial))
    print('{},{},{}'.format(np.mean(cos_train), np.mean(cos_test), np.mean(cos_trial)))





#===================================================================================
# FINAL SUBMISSION
#===================================================================================
def final_predict(X_train, y_train, X_test, company_train, company_dev, company_test):
    global X_DIM, Y_DIM
    X_DIM = X_train[0].shape[0]
    y_train = np.array(y_train)
    print(X_train.shape, y_train.shape)
    Y_DIM = 1

    # SVM
    # regressor = LinearSVR(C=0.1, verbose=1)
    regressor = KerasRegressor(build_fn=attention_imp_merge_exp, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE, verbose=1)
    print(regressor)
    regressor.fit([X_train, company_train, X_train], y_train)
    # predictions = regressor.predict(company_test)
    predictions = regressor.predict([X_test, company_test, X_test])
    print(predictions.shape)
    print(predictions[:20])
    joblib.dump(predictions, '/raid/data/skar3/semeval/source/ml_semeval17/outputs/subtask2_hl/dl_predictions2.pkl')

    print('Training result', cosine_similarity(y_train, regressor.predict([X_train, company_train, X_train])))

def pack_data_to_format():
    predictions = joblib.load('/raid/data/skar3/semeval/source/ml_semeval17/outputs/subtask2_hl/dl_predictions2.pkl')
    print(predictions.shape)
    # print(predictions[:5])
    # print(predictions[5:])


    test_json = open('/raid/data/skar3/semeval/data/raw/hl_ids.txt', 'r').read().split('\n')[:491]
    # print(test_json[:5])
    print(len(test_json))
    pred_list = []
    for i in range(len(test_json)):
        data = {'id': test_json[i], 'sentiment score': str(predictions[i])}
        pred_list.append(data)
    # df = pd.read_csv('/raid/data/skar3/semeval/data/raw/microblog_test.csv', dtype='str')
    # print(len(df))
    # print(df.head())


    # for index, row in df.iterrows():
    #     print(index, predictions[index])
    #     data = {'id': row['id'], 'cashtag': row['cashtag'], 'sentiment score': predictions[index]}
    #     pred_list.append(data)
    print(len(pred_list))
    json.dump(pred_list, open('/raid/data/skar3/semeval/source/ml_semeval17/outputs/subtask2_hl/dl_submission2.json', 'w'))





def get_features(mode):
    loaded_feature_list = []

    for feature_name in config.features_to_use:
        if mode != 'mb' and (feature_name == 'cashtag' or feature_name == 'source'):
            continue
        print('\n---------------------------------------------')
        print( 'Loading {} from '.format(feature_name), end=' ' )
        # filename = '/raid/data/skar3/semeval/vectors/' + 'ner_headline_'+feature_name+'.pkl'

        # Microblog
        if mode == 'mb':
            filename = '/raid/data/skar3/semeval/vectors_mb_new/' + 'ner_microblog_' + feature_name + '.pkl'
        else:
            filename = '/raid/data/skar3/semeval/vectors_hl_new/' + 'hl_' + feature_name + '.pkl'
        print(filename)

        loaded_feature = joblib.load( filename )

        if not isinstance(loaded_feature, np.ndarray):
            loaded_feature = loaded_feature.toarray()
        print('Shape =  {}, type = {}'.format(loaded_feature.shape, type(loaded_feature)))
        loaded_feature_list.append( loaded_feature )
        print('---------------------------------------------')

    return combine_features(loaded_feature_list)


def compile_cos_sim_theano(v1, v2):
    v1 = theano.tensor.vector(v1)
    v2 = theano.tensor.vector(v2)
    numerator = theano.tensor.sum(v1*v2)
    denominator = theano.tensor.sqrt(theano.tensor.sum(v1**2)*theano.tensor.sum(v2**2))
    return numerator/denominator

# cos_sim_theano_fn = compile_cos_sim_theano()

def cnn_model1():
    global X_DIM, Y_DIM
    # Load Embeddings matrix
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR+'hl_voc_embeddings_prs.pkl')
    print(embedding_weights.shape)
    model = Sequential()
    model.add( Embedding( max_features,
                            embedding_dims,
                            input_length=max_len,
                            weights=[embedding_weights],
                            trainable=True) )
    model.add( Conv1D(512, 3, activation='relu') )
    # model.add( MaxPooling1D(3) )
    # model.add(Conv1D(512, 4, activation='relu'))
    # model.add(MaxPooling1D(4))
    # model.add(Conv1D(512, 5, activation='relu'))
    model.add(MaxPooling1D())
    model.add( Flatten() )
    model.add( Dense(100, activation='relu') )
    model.add(Dense(1, init='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.compile(loss='cosine_proximity', optimizer='rmsprop', metrics={'output_a': cosine_similarity})
    # model.compile(loss=compile_cos_sim_theano, optimizer='adam', metrics=[compile_cos_sim_theano])
    # print(model.summary())

    return model


def cnn_2():
    global X_DIM, Y_DIM
    # Load Embeddings matrix
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR+'hl_voc_embeddings_prs.pkl')

    model = Sequential()
    model.add( Embedding( max_features,
                            embedding_dims,
                            input_length=max_len,
                            weights=[embedding_weights],
                            trainable=True) )
    model.add( Conv1D(512, 3, activation='relu') )
    # model.add( MaxPooling1D() )
    model.add(Conv1D(512, 4, activation='relu'))
    # model.add(MaxPooling1D())
    model.add(Conv1D(512, 5, activation='relu'))
    model.add(MaxPooling1D())
    # model.add(MaxPooling1D(35))
    # model.add(MaxPooling1D(3))
    model.add( Flatten() )
    model.add( Dense(20, activation='relu') )
    model.add(Dense(1, init='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def cnn_merged():
    global X_DIM, Y_DIM
    # Load Embeddings matrix
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR+'hl_voc_embeddings_prs.pkl')

    model_1 = Sequential()
    model_1.add( Embedding( max_features,
                            embedding_dims,
                            input_length=max_len,
                            weights=[embedding_weights],
                            trainable=True) )
    model_1.add( Conv1D(512, 3, activation='relu') )
    model_1.add( MaxPooling1D() )
    model_1.add(Flatten())

    model_2 = Sequential()
    model_2.add(Embedding(max_features,
                        embedding_dims,
                        input_length=max_len,
                        weights=[embedding_weights],
                        trainable=True))
    model_2.add(Conv1D(512, 4, activation='relu'))
    model_2.add(MaxPooling1D())
    model_2.add(Flatten())

    model_3 = Sequential()
    model_3.add(Embedding(max_features,
                        embedding_dims,
                        input_length=max_len,
                        weights=[embedding_weights],
                        trainable=True))
    model_3.add(Conv1D(512, 5, activation='relu'))
    model_3.add(MaxPooling1D())
    model_3.add(Flatten())

    model_4 = Sequential()
    model_4.add(Embedding(max_features,
                          embedding_dims,
                          input_length=max_len,
                          weights=[embedding_weights],
                          trainable=True))
    model_4.add(GRU(512))
    combined_model = Sequential()
    combined_model.add(Merge([model_1, model_2, model_3, model_4], mode='concat', concat_axis=1))
    # combined_model.add(MaxPooling1D(3))

    combined_model.add( Dense(1024, activation='relu') )

    combined_model.add(Dropout(0.2))
    # combined_model.add(TimeDistributed(Dense(200), name='time_dist'))
    combined_model.add( Dense(256, activation='relu') )

    combined_model.add(Dense(100, activation='relu'))
    combined_model.add(Dense(1, init='normal'))

    # try using different optimizers and different optimizer configs
    combined_model.compile(loss='mean_squared_error', optimizer='adam')
    # print(model.summary())

    return combined_model


def cnn_lstm_keras_imdb():
    global X_DIM, Y_DIM
    filter_length = 5
    nb_filter = 64
    pool_length = 4
    lstm_output_size = 70

    # Load Embeddings matrix
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR + 'hl_voc_embeddings_2k.pkl')
    print(embedding_weights.shape)
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=max_len,
                        weights=[embedding_weights],
                        trainable=True))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1, init='normal'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='mean_squared_error', optimizer='adam')
    # print(model.summary())
    return model


def attention_imp():
    """
    https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/
    :return:
    """
    global X_DIM, Y_DIM
    # Load Embeddings matrix
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR + 'hl_voc_embeddings_prs.pkl')

    # model cnn

    model_atn = Sequential()
    model_atn.add(Embedding(max_features,
                            embedding_dims,
                            input_length=max_len,
                            weights=[embedding_weights],
                            trainable=True))
    model_atn.add(Bidirectional(GRU(100, return_sequences=True), name='bidirectional'))
    model_atn.add(TimeDistributed(Dense(200), name='time_dist'))
    model_atn.add(AttLayer(name='att'))
    model_atn.add(Dense(1, init='normal', name='combined_dense'))

    # # Compile model
    model_atn.compile(loss='mean_squared_error', optimizer='adam')

    print(model_atn.summary())
    return model_atn

def attention_imp_merge():
    """
    Best One so far with 10 epoch
    https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/
    :return:
    """
    global X_DIM, Y_DIM
    # Load Embeddings matrix
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR + 'hl_voc_embeddings_prs.pkl')

    # model cnn

    model_atn = Sequential()
    model_atn.add(Embedding(max_features,
                            embedding_dims,
                            input_length=max_len,
                            weights=[embedding_weights],
                            trainable=True))
    model_atn.add(Bidirectional(GRU(100, return_sequences=True), name='bidirectional'))
    model_atn.add(TimeDistributed(Dense(200), name='time_dist'))
    model_atn.add(AttLayer(name='att'))

    model_feature_vec = Sequential()
    model_feature_vec.add(Dense(200, input_dim=N_Features, init='normal', activation='relu'))
    model_feature_vec.add(Dense(100, init='normal', activation='relu'))
    model_feature_vec.add(Dropout(0.2))
    model_feature_vec.add(Dense(50, init='normal', activation='relu'))
    model_feature_vec.add(Dense(10, init='normal', activation='relu'))

    merged_layer = Sequential()
    merged_layer.add(Merge([model_atn, model_feature_vec], mode='concat',
                    concat_axis=1, name='merge_layer'))
    merged_layer.add(Dense(200, activation='relu'))
    # merged_layer.add(Bidirectional(GRU(100, return_sequences=True), name='bidirectional_2'))
    # merged_layer.add(TimeDistributed(Dense(200), name='time_dist'))
    # merged_layer.add(AttLayer(name='att'))
    merged_layer.add(Dense(1, init='normal', name='combined_dense'))

    # # Compile model
    merged_layer.compile(loss='mean_squared_error', optimizer='adam')

    print(merged_layer.summary())
    return merged_layer


def attention_imp_merge_exp_e():
    """
    https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/
    :return:
    """
    global X_DIM, Y_DIM
    # Load Embeddings matrix
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR + 'hl_voc_embeddings_3k.pkl')

    # model cnn

    model_atn = Sequential()
    model_atn.add(Embedding(max_features,
                            embedding_dims,
                            input_length=max_len,
                            weights=[embedding_weights],
                            trainable=True))
    model_atn.add(Bidirectional(GRU(200, return_sequences=True), name='bidirectional'))
    model_atn.add(TimeDistributed(Dense(200), name='time_dist'))
    model_atn.add(AttLayer(name='att'))


    model_feature_vec = Sequential()
    model_feature_vec.add(Dense(200, input_dim=N_Features, init='normal', activation='relu'))
    model_feature_vec.add(Dense(100, init='normal', activation='relu'))
    model_feature_vec.add(Dropout(0.2))
    model_feature_vec.add(Dense(50, init='normal', activation='relu'))
    model_feature_vec.add(Dense(10, init='normal', activation='relu'))


    model_cnn = Sequential()
    model_cnn.add(Embedding(max_features,
                            embedding_dims,
                            input_length=max_len,
                            weights=[embedding_weights],
                            trainable=True))
    model_cnn.add(Conv1D(512, 3, activation='relu', name='cnn1'))
    model_cnn.add(Conv1D(512, 4, activation='relu', name='cnn2'))
    model_cnn.add(Conv1D(512, 5, activation='relu', name='cnn3'))
    model_cnn.add(MaxPooling1D(2, name='maxpool'))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation='relu'))


    merged_layer = Sequential()
    merged_layer.add(Merge([model_atn, model_feature_vec, model_cnn], mode='concat',
                           concat_axis=1, name='merge_layer'))
    merged_layer.add(Reshape((1, 338)))
    merged_layer.add(Bidirectional(GRU(200, return_sequences=True), name='bidirectional_2'))
    merged_layer.add(TimeDistributed(Dense(50), name='time_dist'))
    merged_layer.add(AttLayer(name='att'))
    merged_layer.add(Dense(1, init='normal', name='combined_dense'))

    # # Compile model
    merged_layer.compile(loss='mae', optimizer='adam')

    # print(merged_layer.summary())
    return merged_layer


def attention_imp_merge_exp():
    """
    https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/
    :return:
    """
    global X_DIM, Y_DIM
    # Load Embeddings matrix
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR_HL + 'hl_voc_embeddings_3k.pkl')

    # model cnn

    model_atn = Sequential()
    model_atn.add(Embedding(max_features,
                            embedding_dims,
                            input_length=max_len,
                            weights=[embedding_weights],
                            trainable=True))
    model_atn.add(Bidirectional(GRU(200, return_sequences=True), name='bidirectional'))
    model_atn.add(TimeDistributed(Dense(200), name='time_dist'))
    model_atn.add(AttLayer(name='att'))


    model_feature_vec = Sequential()
    model_feature_vec.add(Dense(200, input_dim=N_Features, init='normal', activation='relu'))
    model_feature_vec.add(Dense(100, init='normal', activation='relu'))
    model_feature_vec.add(Dropout(0.2))
    model_feature_vec.add(Dense(50, init='normal', activation='relu'))
    model_feature_vec.add(Dense(10, init='normal', activation='relu'))




    #functional API

    sentence = Input(shape=(max_len,), dtype='float32', name='w1')
    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dims,
                            weights=[embedding_weights],
                            )

    sentence = Input(shape=(max_len,), dtype='float32', name='w1')
    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dims,
                                weights=[embedding_weights],
                                )

    sentence_emb = embedding_layer(sentence)
    # dropout_1 = Dropout(0.2, name='emb_dropout')
    # sentence_drop = dropout_1(sentence_emb)
    cnn_layers = [Convolution1D(filter_length=filter_length, nb_filter=512, activation='relu', border_mode='same') for
                  filter_length in [1, 2, 3, 5]]
    merged_cnn = merge([cnn(sentence_emb) for cnn in cnn_layers], mode='concat', concat_axis=-1)
    # pooling_layer = MaxPooling1D(2, name='maxpool')(merged_cnn)
    attention = AttLayer(name='att')(merged_cnn)
    # flatten_layer = Flatten()(attention)
    cnn_model = Dense(128, init='normal', activation='relu')(attention)
    model_cnn = Model(input=[sentence], output=[cnn_model], name='cnn_model')

    # model_cnn = Sequential()
    # model_cnn.add(Embedding(max_features,
    #                         embedding_dims,
    #                         input_length=max_len,
    #                         weights=[embedding_weights],
    #                         trainable=True))
    # model_cnn.add(Conv1D(512, 3, activation='relu', name='cnn1'))
    # model_cnn.add(Conv1D(512, 4, activation='relu', name='cnn2'))
    # model_cnn.add(Conv1D(512, 5, activation='relu', name='cnn3'))
    # model_cnn.add(MaxPooling1D(2, name='maxpool'))
    # model_cnn.add(Flatten())
    # model_cnn.add(Dense(128, activation='relu'))


    merged_layer = Sequential()
    merged_layer.add(Merge([model_atn, model_feature_vec, model_cnn], mode='concat',
                           concat_axis=1, name='merge_layer'))
    merged_layer.add(Reshape((1, 338)))
    merged_layer.add(Bidirectional(GRU(200, return_sequences=True), name='bidirectional_2'))
    merged_layer.add(TimeDistributed(Dense(50), name='time_dist'))
    merged_layer.add(AttLayer(name='att'))
    merged_layer.add(Dense(1, init='normal', name='combined_dense', activation='tanh'))

    # # Compile model
    merged_layer.compile(loss='mae', optimizer='adam')

    print(merged_layer.summary())
    return merged_layer




def attention_imp_merge_exp_paper():
    """
    https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/
    :return:
    """
    global X_DIM, Y_DIM
    # Load Embeddings matrix
    embedding_weights = joblib.load(config.DUMPED_VECTOR_DIR_HL + 'hl_voc_embeddings_3k.pkl')

    # model cnn

    model_atn = Sequential()
    model_atn.add(Embedding(max_features,
                            embedding_dims,
                            input_length=max_len,
                            weights=[embedding_weights],
                            trainable=True))
    model_atn.add(Bidirectional(GRU(200, return_sequences=True), name='bidirectional'))
    model_atn.add(TimeDistributed(Dense(200), name='time_dist'))
    model_atn.add(AttLayer(name='att'))


    model_feature_vec = Sequential()
    model_feature_vec.add(Dense(200, input_dim=N_Features, init='normal', activation='relu'))
    model_feature_vec.add(Dense(100, init='normal', activation='relu'))
    model_feature_vec.add(Dropout(0.2))
    model_feature_vec.add(Dense(50, init='normal', activation='relu'))
    model_feature_vec.add(Dense(10, init='normal', activation='relu'))




    #functional API

    sentence = Input(shape=(max_len,), dtype='float32', name='w1')
    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dims,
                            weights=[embedding_weights],
                            )

    sentence = Input(shape=(max_len,), dtype='float32', name='w1')
    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dims,
                                weights=[embedding_weights],
                                )

    sentence_emb = embedding_layer(sentence)
    # dropout_1 = Dropout(0.2, name='emb_dropout')
    # sentence_drop = dropout_1(sentence_emb)
    cnn_layers = [Convolution1D(filter_length=filter_length, nb_filter=512, activation='relu', border_mode='same') for
                  filter_length in [1, 2, 3, 5]]
    merged_cnn = merge([cnn(sentence_emb) for cnn in cnn_layers], mode='concat', concat_axis=-1)
    # pooling_layer = MaxPooling1D(2, name='maxpool')(merged_cnn)
    attention = AttLayer(name='att')(merged_cnn)
    # flatten_layer = Flatten()(attention)
    cnn_model = Dense(128, init='normal', activation='relu')(attention)
    model_cnn = Model(input=[sentence], output=[cnn_model], name='cnn_model')


    merged_layer = Sequential()
    merged_layer.add(Merge([model_atn, model_feature_vec, model_cnn], mode='concat',
                           concat_axis=1, name='merge_layer'))
    # merged_layer.add(Reshape((1, 338)))
    # # merged_layer.add(Bidirectional(GRU(200, return_sequences=True), name='bidirectional_2'))
    # merged_layer.add(TimeDistributed(Dense(50), name='time_dist'))
    # merged_layer.add(AttLayer(name='att'))
    merged_layer.add(Dense(128, init='normal', name='combined_d', activation='tanh'))
    merged_layer.add(Dense(1, init='normal', name='combined_dense', activation='tanh'))

    # # Compile model
    merged_layer.compile(loss='mae', optimizer='adam')

    print(merged_layer.summary())
    return merged_layer


def main():

    #----------------------------------------------------------------
    # Headline
    MOOD = 'hl'

    print('Loading X')
    X = joblib.load(config.DUMPED_VECTOR_DIR_HL+'hl_sequences_3k.pkl')
    features = get_features(MOOD)
    print(X.shape, features.shape)
    print('Loading Y')

    if MOOD == 'hl':
        y = joblib.load( '/raid/data/skar3/semeval/vectors_hl_new/headline_scores.pkl' )
        # company_names = joblib.load(config.DUMPED_VECTOR_DIR+'hl_company.pkl')
        # print(company_names.shape)
    else:
        y = joblib.load('/raid/data/skar3/semeval/vectors_mb_new/microblog_scores.pkl')
    print(type(X), type(y))

    # Headline Split
    X_train, X_dev, X_test, Y_train, Y_dev = X[:1156], X[1143:1156], X[1156:], y[:1156], y[1143:1156]
    features_train, features_dev, features_test = features[:1156], features[1143:1156], features[1156:]
    # ----------------------------------------------------------------
    print(len(X_train), len(X_test), len(X_dev), len(Y_train), len(Y_dev))
    print(len(y))
    deep_learning_cv(X_train, Y_train, 10, X_dev, Y_dev, X_test, features_train, features_dev, features_test)
    # final_predict(X_train, Y_train, X_test, features_train, features_dev, features_test)
    print("HEADLINE")

if __name__ == '__main__':
    main()
    pack_data_to_format()
    # test_cl_exp()


