import sys
sys.path.append('/raid/data/skar3/semeval/source/ml_semeval17')
import nltk
from sklearn.externals import joblib
import pandas as pd
import config
import os
import traceback
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from concurrent import futures
from shutil import copyfile
import senticnet
import re

tokenizer = RegexpTokenizer('\w+')
stemmer = PorterStemmer()


def copy_files():
    """
        Will Change for each project
        :return:
        """

    print("COPYING DATA")
    file_list = open(config.DATA_FILES_LIST, 'r').read().split('\n')
    data_list = []
    counter = 0

    for file in file_list:
        counter += 1
        copyfile(config.RAW_DATA_PATH+file+'.txt', config.ORIGINAL_DATA_DIR+file+'.txt')

    print("Copy Finished {} files".format(counter))
    return data_list


def replace_url(content):
    return re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'URL', content)

# ======================================================================================================================
# TOKENIZE DATA with/out Named Entity and numbers Removal
# ======================================================================================================================

def remove_named_entities( content ):
    if isinstance(content, float):
        return ''

    sentences = nltk.sent_tokenize(content)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

    def extract_entity_names(t):
        entity_names = []

        if hasattr(t, 'label') and t.label:
            if t.label() == 'NE':
                entity_names.append(' '.join([child[0] for child in t]))
            else:
                for child in t:
                    entity_names.extend(extract_entity_names(child))

        return entity_names

    entity_names = []
    for tree in chunked_sentences:
        entity_names.extend(extract_entity_names(tree))
    entity_names = list(reversed(sorted(entity_names, key=len)))

    for ne in entity_names:
        content = content.replace(ne, 'NAMEDENTITY')

    return content


def replace_numbers(tokens):
    for i in range(len(tokens)):
        if tokens[i].isdigit():
            tokens[i] = 'CC'

    return tokens


def tokenize_csv_file( source_file_path, dest_file_path, column_names, should_remove_NE, should_remove_numbers ):
    """
    Tokenizes contents of all the columns mentioned in the params


    :param source_dir: (str) Directory of reviews
    :return:
    """
    source_df = pd.read_csv(source_file_path, encoding='ISO-8859-1')

    for index, row in source_df.iterrows():

        for target_column in column_names:
            target_text = row[target_column]
            target_text = replace_url(target_text)
            if should_remove_NE:
                ne_removed_content = remove_named_entities(target_text)
            else:
                ne_removed_content = target_text

            tokens = tokenizer.tokenize(ne_removed_content)
            if should_remove_numbers:
                tokens = replace_numbers(tokens)
            source_df.set_value(index, target_column, ' '.join(tokens))
    source_df.to_csv(dest_file_path, index=False)


def pos_tag_single_file(fileName, source_dir, dest_dir):
    try:
        with open(source_dir + fileName, 'r') as f:
            content = f.read()
            f.close()

        sentences = sent_tokenize(content)
        all_tags = ''
        all_tagged_tokens = ''

        for s in sentences:
            tagged = nltk.pos_tag(nltk.word_tokenize(s))
            tags = ' '.join(list( map(lambda tup: tup[1], tagged) ))
            word_tags = ' '.join(list(map(lambda tup: tup[0]+'/'+tup[1], tagged)))
            all_tags += tags
            all_tagged_tokens += word_tags

        with open(config.POS_TAGGED_DATA_DIR + fileName, 'w') as f:
            f.write(all_tags)
            f.close()

        with open(config.POS_WORD_DATA_DIR + fileName, 'w') as f:
            f.write(all_tagged_tokens)
            f.close()

    except:
        traceback.print_exc()
        print(fileName)


def pos_tag_directory():
    source_dir = config.ORIGINAL_DATA_DIR
    destination_dir = config.POS_TAGGED_DATA_DIR
    file_list = os.listdir(source_dir)
    pool = futures.ProcessPoolExecutor(32)
    x = [pool.submit(pos_tag_single_file, file_name, source_dir, destination_dir) for file_name in file_list]
    futures.wait(x)

# ======================================================================================================================
# Writes sentic scores and concepts as column in csv file
# ======================================================================================================================

def extract_sentic_concepts_and_scores_csv():
    source_dir = '/raid/data/skar3/semeval/data/preprocessed/mb_train_trial_test_new_prs.csv'
    source_df = pd.read_csv(source_dir)
    source_df['concepts'] = ''
    source_df['polarity'] = ''
    source_df['attention'] = ''
    source_df['pleasantness'] = ''
    source_df['aptitude'] = ''
    source_df['sensitivity'] = ''
    sn = senticnet.Senticnet()
    counter = 0

    for index, row in source_df.iterrows():
        concepts = []
        concept_scores = {'polarity' : 0, 'attention': 0, 'pleasantness': 0, 'aptitude': 0, 'sensitivity': 0}
        stemmed_concepts = []
        stemmed_concept_scores = {'polarity': 0, 'attention': 0, 'pleasantness': 0, 'aptitude': 0, 'sensitivity': 0}

        if isinstance(row['text'], float):
            content = ''
            stemmed_title = ''
        else:
            content = '_' + row['text'].replace(' ', '_').lower() + '_'
            stemmed_title = '_' + '_'.join([stemmer.stem(t) for t in row['text'].lower().split()]) + '_'
        # print(stemmed_title)
        for concept_key in sn.data.keys():
            if '_' + concept_key +'_' in content:
                concepts.append(concept_key)
                concept_data = sn.concept(concept_key)
                concept_scores['polarity'] += concept_data['polarity']
                concept_scores['attention'] += concept_data['sentics']['attention']
                concept_scores['pleasantness'] += concept_data['sentics']['pleasantness']
                concept_scores['aptitude'] += concept_data['sentics']['aptitude']
                concept_scores['sensitivity'] += concept_data['sentics']['sensitivity']
            if '_' + concept_key +'_' in stemmed_title:
                stemmed_concepts.append(concept_key)
                concept_data = sn.concept(concept_key)
                stemmed_concept_scores['polarity'] += concept_data['polarity']
                stemmed_concept_scores['attention'] += concept_data['sentics']['attention']
                stemmed_concept_scores['pleasantness'] += concept_data['sentics']['pleasantness']
                stemmed_concept_scores['aptitude'] += concept_data['sentics']['aptitude']
                stemmed_concept_scores['sensitivity'] += concept_data['sentics']['sensitivity']

        # print(concept_scores)
        if len(concepts) > 0:
            for k, v in concept_scores.items():
                concept_scores[k] /= len(concepts)
        else:
            print('>> 0  >>', row['text'])
        if len(stemmed_concepts) > 0:
            for k, v in stemmed_concept_scores.items():
                stemmed_concept_scores[k] /= len(stemmed_concepts)
            else:
                print('>> 1  >>', stemmed_title)

        # print(concept_scores)
        source_df.set_value(index, 'polarity', concept_scores['polarity'])
        source_df.set_value(index, 'attention', concept_scores['attention'])
        source_df.set_value(index, 'pleasantness', concept_scores['pleasantness'])
        source_df.set_value(index, 'aptitude', concept_scores['aptitude'])
        source_df.set_value(index, 'sensitivity', concept_scores['sensitivity'])
        source_df.set_value(index, 'concepts', ' '.join(concepts))
        source_df.set_value(index, 'stemmed_polarity', stemmed_concept_scores['polarity'])
        source_df.set_value(index, 'stemmed_attention', stemmed_concept_scores['attention'])
        source_df.set_value(index, 'stemmed_pleasantness', stemmed_concept_scores['pleasantness'])
        source_df.set_value(index, 'stemmed_aptitude', stemmed_concept_scores['aptitude'])
        source_df.set_value(index, 'stemmed_sensitivity', stemmed_concept_scores['sensitivity'])
        source_df.set_value(index, 'stemmed_concepts', ' '.join(stemmed_concepts))

        # joblib.dump(concept_scores, os.path.join(config.SENTIC_SCORES_DATA_DIR + file.replace('.txt', '.pkl')))
        print(counter)
        counter += 1
        # break
    source_df.to_csv(source_dir, index=False)


def extract_data_ids_from_csv( csvpath, output_path, field):
    df = pd.read_csv(csvpath)
    ids = df[field].values

    with open(output_path, 'w') as f:
        for id in ids:
            f.write(str(id))
            f.write('\n')
        f.close()



# copy_files()
# pos_tag_directory()
tokenize_csv_file('/raid/data/skar3/semeval/data/raw/mb_train_trial_test_new_raw.csv',
                  '/raid/data/skar3/semeval/data/preprocessed/mb_train_trial_test_new_prs.csv',
                  ['text'],
                  True,
                  True
                  )
extract_sentic_concepts_and_scores_csv()

# print(replace_url('@lcc007: $ISR bullish pennant/symmetrical triangle b/o in progress. Long from $2.66 http://stks.co/p0T1O" // BOOOM'))
# extract_data_ids_from_csv('/raid/data/skar3/semeval/data/raw/headline_test_raw.csv', '/raid/data/skar3/semeval/data/raw/hl_ids.txt', 'id')

