import os
import numpy as np
import glob
import time
import gzip
import json
import datetime
import pandas as pd
from typing import NamedTuple
import warnings

warnings.filterwarnings('ignore')
import torch
# from keras.preprocessing.sequence import pad_sequences
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from tqdm import tqdm
from pytorch_transformers import *
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict


class TweetUser(NamedTuple):
    user_id: str
    description: str
    label: str


def get_bert_embeddings(corpus):
    # init pretrained model
    # pre_trained: bert-base-uncased
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    bert_embed_features = {}
    for tweet in corpus:
        sent = tweet.description
        input_ids = torch.tensor(tokenizer.encode(sent, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        # NOTE: vector for last layer [cls] token. Since it's used for classification task
        bert_embed_features[tweet.user_id] = outputs[0][0][0].detach().numpy().tolist()
    return bert_embed_features


def vectorize(feature_dict):
    vecs = []
    for k in list(feature_dict.keys()):
        vec = []
        f = feature_dict.get(k)
        vec.extend(f)
        vecs.append(vec)
    return vecs


def construct_dataset(fn):
    dataset = []
    with open(fn, 'r') as inf:
        for line in inf:
            data = json.loads(line.strip())
            dataset.append(TweetUser(data['user_id'], data['description'], data['label']))
    return dataset


def preprocess_text():
    pass


def write_dict_to_json(dic, fn):
    """
    input: dic -> a dictionary to be dumped, fn -> filename
    fn: {feature_name}.json.gz
    """
    print('writing to', fn)
    new_dic = {}
    for tweet_id, ele in dic.items():
        if type(ele) == np.ndarray:
            new_dic[tweet_id] = ele.tolist()
        elif type(ele) == dict:
            ele_dic = {}
            for k, v in ele.items():
                if type(v) == np.ndarray:
                    ele_dic[k] = v.tolist()
                else:
                    ele_dic[k] = v
            new_dic[tweet_id] = ele_dic
        else:
            new_dic[tweet_id] = ele

    with gzip.open(fn, 'w') as outf:
        outf.write("{}".format(json.dumps(new_dic)).encode('utf8'))


def read_dict_from_json(fn):
    """
    input: fn -> filename
    output: a dict
    """
    print('reading from', fn)
    data = {}
    with gzip.open(fn, 'r') as inf:
        for line in inf:
            data.update(json.loads(line.strip().decode('utf8')))

    new_data = {}
    for tweet_id, ele in data.items():
        if type(ele) == list:
            new_data[tweet_id] = np.array(ele)
        elif type(ele) == dict:
            ele_dic = {}
            for k, v in ele.items():
                if type(v) == list:
                    ele_dic[k] = np.array(v)
                else:
                    ele_dic[k] = v
            new_data[tweet_id] = ele_dic
        else:
            new_data[tweet_id] = ele

    return new_data


def get_bert_feature(train_fn, dev_fn, test_fn):
    # pre_process_url = True  # Set to remove URLs
    # pre_process_usr = True

    result = defaultdict(dict)

    for fn in [train_fn, dev_fn, test_fn]:
        data = construct_dataset(fn)
        # test = construct_dataset(dev_fn)
        labels = [t.label for t in data]
        # test_labels = [t.label for t in test]
        name = os.path.basename(fn).split('.')[0]

        outfn = '{}.bert.json.gz'.format(name)
        if os.path.exists(outfn):
            feature = read_dict_from_json(fn=outfn)
        else:
            feature = get_bert_embeddings(data)
            write_dict_to_json(feature, fn=outfn)

        feats_data = vectorize(feature)
        print(len(feats_data), len(feats_data[1]))

        result[name.split('_')[1]]['data'] = feats_data
        result[name.split('_')[1]]['label'] = labels

    return result


def fit_test_model(train, train_label, test, test_label, model, c):
    model.fit(train, train_label)
    # Predict
    # p_pred = model.predict_proba(feats_tst_A)
    # Metrics
    y_pred = model.predict(test)
    score_ = model.score(test, test_label)
    conf_m = confusion_matrix(test_label, y_pred)
    report = classification_report(test_label, y_pred, output_dict=True)

    # print('score_:', score_, end='\n\n')
    # print('conf_m:', conf_m, sep='\n', end='\n\n')
    # print('report:', str(report), sep='\n')

    print(
        f"{report['accuracy']:.4},{report['macro avg']['precision']:.4},{report['macro avg']['recall']:.4},{report['macro avg']['f1-score']:.4}",
        sep='\t')

    return (c, report['macro avg']['f1-score'])


if __name__ == '__main__':
    for task in ['baseline', 'balanced']:
        dataset = get_bert_feature(train_fn=os.path.join('../zach_dataset', 'baseline_train.jsonl'),
                                   dev_fn=os.path.join('../zach_dataset', '{}_dev.jsonl'.format(task)),
                                   test_fn=os.path.join('../zach_dataset', '{}_test.jsonl'.format(task)))

        C_param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        result_list = []
        for c in C_param_range:
            print(c)
            model = LogisticRegression(solver='liblinear', penalty='l2', C=c, random_state=0)
            result_list.append(
                fit_test_model(dataset['train']['data'], dataset['train']['label'], dataset['dev']['data'],
                               dataset['dev']['label'], model, c))

        best_c = sorted(result_list, key=lambda x: x[1], reverse=True)[0][0]
        best_model = LogisticRegression(solver='liblinear', penalty='l2', C=best_c, random_state=0)
        print('best model performance on test set:', best_c)
        fit_test_model(dataset['train']['data'], dataset['train']['label'], dataset['test']['data'],
                       dataset['test']['label'], best_model, best_c)
