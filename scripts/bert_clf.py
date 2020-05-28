import os
import sys
import numpy as np
import glob
import time
import gzip
import json
from typing import NamedTuple, List
import warnings

warnings.filterwarnings('ignore')
import torch
# from keras.preprocessing.sequence import pad_sequences
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from tqdm import tqdm
from pytorch_transformers import *
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

ZACH_MAP = {
    "Asian": 3,
    "A": 3,
    "asian": 3,
    "Black": 1,
    "B": 1,
    "black": 1,
    "Hispanic": 2,
    "H/L": 2,
    "latin": 2,
    "White": 4,
    "W": 4,
    "white": 4,
}


class TweetUser(NamedTuple):
    user_id: str
    description: str
    label: str


class TweetTimeline(NamedTuple):
    user_id: str
    texts: List[str]
    label: str


### featurization

def get_bert_embeddings(corpus, model=None, tokenizer=None, device=None, input='description'):
    # init pretrained model
    # pre_trained: bert-base-uncased
    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = init_models()

    bert_embed_features = {}
    for tweet in corpus:
        if input == 'description':
            sent = preprocess_text(tweet.description)
            input_ids = torch.tensor(tokenizer.encode(sent, add_special_tokens=True), device=device).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids)
            # NOTE: vector for last layer [cls] token. Since it's used for classification task
            bert_embed_features[tweet.user_id] = outputs[0][0][0].cpu().detach().numpy()
        else:
            # start = time.time()
            tmp = []
            count = 0
            if len(tweet.texts) == 0:
                continue
            for tweet_text in tweet.texts:
                count += 1
                if count > 200:
                    break
                sent = preprocess_text(tweet_text)
                input_ids = torch.tensor(tokenizer.encode(sent, add_special_tokens=True), device=device).unsqueeze(0)  # Batch size 1
                outputs = model(input_ids)
                tmp.append(outputs[0][0][0].cpu().detach().numpy())
            bert_embed_features[tweet.user_id] = average_embedding(tmp)
            # print("user {}: {}".format(tweet.user_id, time.time()-start))
    return bert_embed_features


def read_bert_embeddings(dataset):
    features = {}
    embed_dir = '/export/c10/pxu/data/race/distil_bert_embed'

    for user in dataset:
        embed_fn = os.path.join(embed_dir, '{}_embed.json.gz'.format(user))
        with gzip.open(embed_fn, 'r') as inf:
            for line in inf:
                data = json.loads(line.strip().decode('utf8'))
                features.update(data)

    return features


def average_embedding(embed_list):
    return np.mean(embed_list, axis=0)


def vectorize(feature_dict, labels):
    vecs = []
    label_vecs = []
    for k, v in feature_dict.items():
        if type(v) == np.ndarray and len(v) == 768:
            label_vecs.append(labels[k])
            vecs.append(v)
    return vecs, label_vecs


def construct_dataset(fn, input='description'):
    dataset = []
    if fn.endswith('.json.gz'):
        inf = gzip.open(fn, 'r')
    else:
        inf = open(fn, 'r')
    for line in inf:
        if fn.endswith('.json.gz'):
            data = json.loads(line.strip().decode('utf8'))
        else:
            data = json.loads(line.strip())
        if input == 'description':
            dataset.append(TweetUser(data['user_id'], data['description'], data['label']))
        else:
            dataset.append(TweetTimeline(data['id_str'], data['texts'], ZACH_MAP[data['label']]))

    return dataset


def preprocess_text(text):
    # TODO: url? username?
    return text.lower()


### utils

def write_dict_to_json(dic, fn, verbose=True):
    """
    input: dic -> a dictionary to be dumped, fn -> filename
    fn: {feature_name}.json.gz
    """
    if verbose:
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


def read_user_ids(fn):
    user_ids =[]
    with open(fn, 'r') as inf:
        for line in inf:
            user_ids.append(line.strip())
    return user_ids


def init_models():
    model_name = "distilbert-base-uncased"
    # 'bert-base-uncased'
    if model_name == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertModel.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    if torch.cuda.is_available():
        print("using GPU")
        model.cuda()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return model, tokenizer, device


### training setting

def get_bert_feature(train_fn, dev_fn, test_fn, input='description'):
    # pre_process_url = True  # Set to remove URLs
    # pre_process_usr = True

    result = defaultdict(dict)

    fn_mapping = {
        'train': train_fn,
        'dev': dev_fn,
        'test': test_fn
    }

    for fn_key in fn_mapping:
        fn = fn_mapping[fn_key]
        if not len(fn):
            continue
        data = construct_dataset(fn, input)
        labels = {}
        for t in data:
            labels[t.user_id] = t.label

        name = os.path.basename(fn).split('.')[0]

        # outfn = '{}_{}.bert.json.gz'.format(name, fn_key)
        outfn = '{}.bert.json.gz'.format(name)
        # print('reading/writing to', outfn)
        if os.path.exists(outfn):
            feature = read_dict_from_json(fn=outfn)
        else:
            if input == 'description':
                feature = get_bert_embeddings(data, input)
            else:
                feature = read_bert_embeddings(data)
            write_dict_to_json(feature, fn=outfn)

        feats_data, feats_label = vectorize(feature, labels)
        print(len(feats_data), len(feats_data[1]))
        result[fn_key]['data'] = feats_data
        result[fn_key]['label'] = feats_label

    return result


def fit_test_model(train, train_label, test, test_label, model, c):
    print("Size of training set:", len(train_label))
    try:
        model.fit(train, train_label)
    except ValueError as err:
        print(err)
        # print(len(train))
        print(type(train[0]))
        print(train_label[0])
        # model.fit([train[0]], [train_label[0]])
        os._exit(1)
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

    return c, report['macro avg']['f1-score']


def param_search(dataset):
    """
    only c in l2 rugularization for now
    """
    C_param_range = [0.001, 0.01, 0.1, 1, 10, 100]
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


### experiments

def run_description_model():
    for task in ['baseline', 'balanced']:
        dataset = get_bert_feature(train_fn=os.path.join('../zach_dataset', 'baseline_train.jsonl'),
                                   dev_fn=os.path.join('../zach_dataset', '{}_dev.jsonl'.format(task)),
                                   test_fn=os.path.join('../zach_dataset', '{}_test.jsonl'.format(task)))

        param_search(dataset)


def run_unigram_model():
    data_dir = '/export/c10/zach/demographics/models/datasets/noisy'
    baseline_dir = '/export/c10/zach/demographics/models/datasets/baseline'
    # datasets = ['baseline', 'balanced.7756', 'group_person.indorg', 'exact_group.thre035']
    datasets = ['baseline']
    for task in ['baseline', 'balanced']:
        for dataset_name in datasets:
            for crowd in [True, False]:
                if dataset_name == 'baseline' and crowd:
                    continue
                start = time.time()
                print("task: {}, dataset: {}, use_crowd: {}".format(task, dataset_name, crowd))
                train_dir = data_dir if task != 'baseline' else baseline_dir
                dataset = get_bert_feature(
                    train_fn=os.path.join(train_dir, '{}_train.bert_tweets.json.gz'.format(dataset_name)),
                    dev_fn=os.path.join(baseline_dir, '{}_dev.bert_tweets.json.gz'.format(task)),
                    test_fn=os.path.join(baseline_dir, '{}_test.bert_tweets.json.gz'.format(task)),
                    input='timeline')

                if crowd:
                    tmp_dataset = get_bert_feature(
                        train_fn=os.path.join(baseline_dir, 'baseline_train.bert_tweets.json.gz'),
                        dev_fn='',
                        test_fn='',
                        input='timeline')

                    dataset['train']['data'] += tmp_dataset['train']['data']
                    dataset['train']['label'] += tmp_dataset['train']['label']

                param_search(dataset)

                print(time.time() - start)


def get_all_user_ids():
    fn_list = glob.glob('/export/c10/zach/demographics/models/datasets/baseline/' + '*bert_tweets.json.gz')
    fn_list += glob.glob('/export/c10/zach/demographics/models/datasets/noisy/' + '*bert_tweets.json.gz')
    print(len(fn_list))
    tweet_ids = set()
    for fn in fn_list:
        print(fn)
        with gzip.open(fn, 'r') as inf:
            for line in inf:
                data = json.loads(line.strip().decode('utf8'))
                tweet_ids.add(data['id_str'])

    with open('all_user_ids.txt', 'w') as outf:
        for ele in tweet_ids:
            outf.write("{}\n".format(ele))


def get_all_user_embeddings_parallel():
    job_num = sys.argv[-1]
    assert 0 <= int(job_num) <= 3
    embed_dir = '/export/c10/pxu/data/race/distil_bert_embed'
    user_ids = read_user_ids(os.path.join(embed_dir, 'users0{}'.format(int(job_num))))

    fn_list = glob.glob('/export/c10/zach/demographics/models/datasets/baseline/' + '*bert_tweets.json.gz')
    fn_list += glob.glob('/export/c10/zach/demographics/models/datasets/noisy/' + '*bert_tweets.json.gz')

    # init model
    model, tokenizer, device = init_models()

    # count = 0
    for fn in fn_list:
        print(fn)
        with gzip.open(fn, 'r') as inf:
            for line in inf:
                # count += 1
                # if count > 200:
                #     break
                data = json.loads(line.strip().decode('utf8'))
                _id = data['id_str']
                if _id not in user_ids:
                    continue
                outfn = os.path.join(embed_dir, "{}_embed.json.gz".format(_id))
                if os.path.exists(outfn):
                    continue
                one_user = [TweetTimeline(_id, data['texts'], ZACH_MAP[data['label']])]
                embed_dict = get_bert_embeddings(one_user, model=model, tokenizer=tokenizer, device=device, input='timeline')
                write_dict_to_json(embed_dict, fn=outfn, verbose=False)



if __name__ == '__main__':
    # run_description_model()
    # run_unigram_model()
    # get_all_user_embeddings()
    get_all_user_embeddings_parallel()