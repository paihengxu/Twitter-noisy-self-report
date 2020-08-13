import os
import sys
import numpy as np
import glob
import time
import gzip
import json
from typing import NamedTuple, List
import warnings
import pickle
from argparse import ArgumentParser

warnings.filterwarnings('ignore')
import torch
from pytorch_transformers import *
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

from utils import read_user_ids
from utils import write_dict_to_json
from utils import read_dict_from_json

LABEL_MAP = {
    "Asian": 3,
    "A": 3,
    "asian": 3,
    "Black": 1,
    "B": 1,
    "black": 1,
    "Hispanic": 2,
    "H/L": 2,
    "latin": 2,
    "latinx": 2,
    "White": 4,
    "W": 4,
    "white": 4,
}

LABEL_BACK_MAP = {
    1: "B",
    2: "H/L",
    3: "A",
    4: "W"
}

MODEL_NAME = "distilbert-base-uncased"
# embed_dir = '/path/to/embeddings'

class TweetUser(NamedTuple):
    user_id: str
    description: str
    label: str


class TweetTimeline(NamedTuple):
    user_id: str
    texts: List[str]
    label: str


### featurization
class BertRaceClassifier:
    def __init__(self, user_id_fn=None, dataset_dir=None, embed_dir=None, out_dir=None, feature_dir=None, input='timeline',
                 best_model_fn=None, model_name="distilbert-base-uncased", use_cuda=True):
        self.user_id_fn = user_id_fn
        self.dataset_dir = dataset_dir
        self.embed_dir = embed_dir
        self.out_dir = out_dir
        self.input = input
        self.model_fn = best_model_fn
        self.model_name = model_name
        self.use_cuda = use_cuda
        # dir to store intermediate feature json.gz files (for experiment task)
        self.feature_dir = feature_dir


    def init_models(self, distil=True):
        # 'bert-base-uncased'
        if distil:
            tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            model = DistilBertModel.from_pretrained(self.model_name)
        else:
            tokenizer = BertTokenizer.from_pretrained(self.model_name)
            model = BertModel.from_pretrained(self.model_name)
        if self.use_cuda and torch.cuda.is_available():
            print("using GPU")
            model.cuda()
            device = torch.device('cuda:0')
        else:
            print('using CPU')
            device = torch.device('cpu')
        return model, tokenizer, device


    def get_bert_embeddings(self, corpus, model=None, tokenizer=None, device=None, tweet_limit=200):
        # init pretrained model
        # pre_trained: bert-base-uncased
        if model is None or tokenizer is None or device is None:
            model, tokenizer, device = self.init_models()

        bert_embed_features = {}
        for tweet in corpus:
            if self.input == 'description':
                sent = self.preprocess_text(tweet.description)
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
                    if tweet_limit and count > tweet_limit:
                        break
                    sent = self.preprocess_text(tweet_text)
                    input_ids = torch.tensor(tokenizer.encode(sent, add_special_tokens=True), device=device).unsqueeze(0)  # Batch size 1
                    outputs = model(input_ids)
                    tmp.append(outputs[0][0][0].cpu().detach().numpy())
                bert_embed_features[tweet.user_id] = self.average_embedding(tmp)
                # print("user {}: {}".format(tweet.user_id, time.time()-start))
        return bert_embed_features


    def read_bert_embeddings(self, dataset):
        features = {}

        for user in dataset:
            embed_fn = os.path.join(self.embed_dir, '{}_embed.json.gz'.format(user.user_id))
            with gzip.open(embed_fn, 'r') as inf:
                for line in inf:
                    data = json.loads(line.strip().decode('utf8'))
                    features.update(data)

        return features


    @staticmethod
    def average_embedding(embed_list):
        return np.mean(embed_list, axis=0)

    @staticmethod
    def vectorize(feature_dict, labels=None):
        vecs = []
        label_vecs = []
        user_ids = []
        for k, v in feature_dict.items():
            if len(v) == 768:
                if labels:
                    label_vecs.append(labels[k])
                vecs.append(v)
                user_ids.append(k)
        return vecs, label_vecs, user_ids

    @staticmethod
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
                dataset.append(TweetTimeline(data['id_str'], data['texts'], LABEL_MAP[data['label']]))

        return dataset

    @staticmethod
    def preprocess_text(text):
        return text.lower()


    ### training setting

    def get_bert_feature(self, train_fn, dev_fn, test_fn, input='description'):
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
            data = self.construct_dataset(fn, input)
            labels = {}
            for t in data:
                labels[t.user_id] = t.label

            name = os.path.basename(fn).split('.')[0]

            outfn = os.path.join(self.feature_dir, '{}.bert.json.gz'.format(name))
            if os.path.exists(outfn):
                feature = read_dict_from_json(fn=outfn)
            else:
                print(outfn)
                os._exit(1)
                if input == 'description':
                    feature = self.get_bert_embeddings(data, input)
                else:
                    print("reading embedding for {}".format(fn))
                    feature = self.read_bert_embeddings(data)
                write_dict_to_json(feature, fn=outfn)

            feats_data, feats_label, _ = self.vectorize(feature, labels)
            print(len(feats_data))
            print(len(feats_data[1]))
            result[fn_key]['data'] = feats_data
            result[fn_key]['label'] = feats_label

        return result

    @staticmethod
    def fit_test_model(train, train_label, test, test_label, model, c):
        model.fit(train, train_label)
        # Metrics
        y_pred = model.predict(test)
        report = classification_report(test_label, y_pred, output_dict=True)

        print(
            f"{report['accuracy']:.4},{report['macro avg']['precision']:.4},{report['macro avg']['recall']:.4},{report['macro avg']['f1-score']:.4}",
            sep='\t')

        return c, report['macro avg']['f1-score'], report['accuracy'], model


    @staticmethod
    def save_logits(best_model, dataset, split, dataset_setting, task):
        y_pred_logits = best_model.predict_proba(dataset[split]['data'])
        with open("logits/" + "{}_{}_{}.logits.json".format(task, dataset_setting, split), 'w') as outf:
            for probs, true_label in zip(y_pred_logits, dataset[split]['label']):
                obj = {"label": true_label, "logits": probs.tolist()}
                outf.write("{}\n".format(json.dumps(obj)))


    def param_search(self, dataset, dataset_setting, task, class_weight=None):
        """
        only c in l2 regularization
        """
        model_fn = "{}.p".format(dataset_setting)
        if os.path.exists(model_fn):
            print("loading model from:", model_fn)
            best_model = pickle.load(open(model_fn, 'rb'))
            y_pred = best_model.predict(dataset['test']['data'])
            report = classification_report(dataset['test']['label'], y_pred, output_dict=True)
        else:
            C_param_range = [0.001, 0.01, 0.1, 1, 10, 100]
            result_list = []
            for c in C_param_range:
                print(c)
                model = LogisticRegression(solver='liblinear', penalty='l2', C=c, random_state=0, class_weight=class_weight)
                result_list.append(
                    self.fit_test_model(dataset['train']['data'], dataset['train']['label'], dataset['dev']['data'],
                                   dataset['dev']['label'], model, c))

            best_c = sorted(result_list, key=lambda x: x[1], reverse=True)[0][0]
            best_model = sorted(result_list, key=lambda x: x[1], reverse=True)[0][3]
            print('best model performance on test set:', best_c)
            y_pred = best_model.predict(dataset['test']['data'])
            report = classification_report(dataset['test']['label'], y_pred, output_dict=True)

        print(
            f"{report['accuracy']:.4},{report['macro avg']['precision']:.4},{report['macro avg']['recall']:.4},{report['macro avg']['f1-score']:.4}",
            sep='\t')

        ### saving logits
        self.save_logits(best_model, dataset, 'dev', dataset_setting, task)
        self.save_logits(best_model, dataset, 'test', dataset_setting, task)

        ### saving model
        with open("{}.p".format(dataset_setting), 'wb') as file:
            pickle.dump(best_model, file)

        return report['macro avg']['f1-score'], report['accuracy']


    ### experiments

    def run_description_model(self):
        for task in ['baseline', 'balanced']:
            dataset = self.get_bert_feature(train_fn=os.path.join('../zach_dataset', 'baseline_train.jsonl'),
                                       dev_fn=os.path.join('../zach_dataset', '{}_dev.jsonl'.format(task)),
                                       test_fn=os.path.join('../zach_dataset', '{}_test.jsonl'.format(task)))

            self.param_search(dataset, "description", task)


    def run_timeline_model(self, class_weight=None):
        # TODO: update these dirs
        # data_dir = '/path/to/training/data'
        # these dirs doesn't matter
        baseline_dir = '/export/c10/zach/demographics/models/datasets/baseline'
        data_dir = '/export/c10/zach/demographics/models/datasets/noisy'
        # baseline_dir = '/path/to/dev_test/data'
        datasets = ['baseline', 'group_person.indorg', 'exact_group.thre035', 'balanced.7756']
        # datasets = ['baseline']
        outf = open('model_result.csv', 'w')
        for task in ['baseline', 'balanced']:
            outf.write("{}\n".format(task))
            for dataset_name in datasets:
                for crowd in [True, False]:
                    if dataset_name == 'baseline' and crowd:
                        continue
                    start = time.time()
                    print("task: {}, dataset: {}, use_crowd: {}".format(task, dataset_name, crowd))
                    train_dir = data_dir if dataset_name != 'baseline' else baseline_dir
                    dataset = self.get_bert_feature(
                        train_fn=os.path.join(train_dir, '{}_train.bert_tweets.json.gz'.format(dataset_name)),
                        dev_fn=os.path.join(baseline_dir, '{}_dev.bert_tweets.json.gz'.format(task)),
                        test_fn=os.path.join(baseline_dir, '{}_test.bert_tweets.json.gz'.format(task)),
                        input='timeline')

                    if crowd:
                        tmp_dataset = self.get_bert_feature(
                            train_fn=os.path.join(baseline_dir, 'baseline_train.bert_tweets.json.gz'),
                            dev_fn='',
                            test_fn='',
                            input='timeline')

                        dataset['train']['data'] += tmp_dataset['train']['data']
                        dataset['train']['label'] += tmp_dataset['train']['label']

                    dataset_setting = dataset_name
                    dataset_setting += "+crowd" if crowd else ""

                    best_f1, best_acc = self.param_search(dataset, dataset_setting, task, class_weight)
                    outf.write("{}\n".format(",".join([dataset_setting, str(best_f1), str(best_acc)])))

                    print(time.time() - start)
        outf.close()


    def predict(self):
        outfn = os.path.join(self.out_dir, os.path.basename(self.user_id_fn) + '_out.json')
        assert not os.path.exists(outfn), "{} already exists.\n terminated...".format(outfn)

        ## load model
        assert os.path.exists(self.model_fn), "{} not exists".format(self.model_fn)
        loaded_model = pickle.load(open(self.model_fn, 'rb'))
        print("successfully load model from {}".format(self.model_fn))

        ## load user ids
        user_ids = read_user_ids(self.user_id_fn)
        print("get {} user ids".format(len(user_ids)))

        ## load/vectorize dataset
        features = {}
        for _id in user_ids:
            embed_fn = os.path.join(self.embed_dir, '{}_embed.json.gz'.format(_id))
            if not os.path.exists(embed_fn):
                print("embedding for {} not exists".format(_id))
                continue
            with gzip.open(embed_fn, 'r') as inf:
                for line in inf:
                    data = json.loads(line.strip().decode('utf8'))
                    features.update(data)
        feats_data, _, feats_ids = self.vectorize(features)

        y_pred = loaded_model.predict(feats_data)

        print("Made predictions for {} users".format(len(y_pred)))

        ## output prediction result
        with open(outfn, 'w') as outf:
            for _id, demog in zip(feats_ids, y_pred):
                obj = {"id": _id, "label": LABEL_BACK_MAP[demog]}
                outf.write("{}\n".format(json.dumps(obj)))


    def get_all_user_embeddings_parallel(self, user_limit=None):
        # user_ids = read_user_ids(os.path.join('/export/c10/pxu/data/social_distancing_user/bert_tweets/', 'users{}'.format(job_num)))
        # user_ids = read_user_ids(user_id_fn)

        # fn_list = glob.glob('/path/to/dev_test/datasets/' + '*bert_tweets.json.gz')
        # fn_list += glob.glob('/path/to/train/datasets/' + '*bert_tweets.json.gz')
        # fn_list = glob.glob('/export/c10/pxu/data/social_distancing_user/bert_tweets/' + '*.json.gz')
        print("writing embeddings to {}".format(self.embed_dir))

        id_fn = os.path.basename(self.user_id_fn).split('.')[0]
        dataset_fn = os.path.join(self.dataset_dir, "{}_dataset.json.gz".format(id_fn))

        # init model
        model, tokenizer, device = self.init_models()
        print('model initialized')

        count = 0
        with gzip.open(dataset_fn, 'r') as inf:
            for line in inf:
                count += 1
                if user_limit and count > user_limit:
                    break
                data = json.loads(line.strip().decode('utf8'))
                _id = data['id_str']
                # print(_id)
                # if _id not in user_ids:
                #     continue
                outfn = os.path.join(self.embed_dir, "{}_embed.json.gz".format(_id))
                if os.path.exists(outfn):
                    continue
                if 'label' in data:
                    one_user = [TweetTimeline(_id, data['texts'], LABEL_MAP[data['label']])]
                else:
                    one_user = [TweetTimeline(_id, data['texts'], None)]
                embed_dict = self.get_bert_embeddings(one_user, model=model, tokenizer=tokenizer, device=device, input='timeline')
                write_dict_to_json(embed_dict, fn=outfn, verbose=False)


if __name__ == '__main__':
    """
    run get_all_user_embeddings_parallel() first to get the embeddings for all users
    then run_timeline_model() for model performance
    """
    parser = ArgumentParser()
    parser.add_argument('--task', choices=('embedding', 'prediction', 'experiment'),
                        help='Build dataset first, then get embedding, do prediction or experiment at last.'
                             'experiment for training/testing in the paper ')
    # parser.add_argument('--job_num', type=str, default=None,
    #                     help='If splitting the job, input the job_num to match the file names')
    parser.add_argument('--user_id_fn', type=str, default=None, help='a txt file for user ids')
    parser.add_argument('--embed_dir', type=str, help='Directory for user embeddings')
    parser.add_argument('--dataset_dir', nargs='?', type=str, help='Directory for built dataset')
    parser.add_argument('--out_dir', nargs='?', type=str, help='Directory for model output')
    parser.add_argument('--model_fn', type=str, default='/export/c10/pxu/Twitter-noisy-self-report/scripts/distil_bert_model/balanced.7756+crowd.p',
                        help='Path to model')
    parser.add_argument('--cuda', action='store_true', help='Whether use cuda')

    args = parser.parse_args()
    start = time.time()

    clf = BertRaceClassifier(user_id_fn=args.user_id_fn, dataset_dir=args.dataset_dir, embed_dir=args.embed_dir,
                             out_dir=args.out_dir, input='timeline', best_model_fn=args.model_fn, use_cuda=args.cuda,
                             feature_dir='/export/c10/pxu/Twitter-noisy-self-report/scripts/distil_bert_feature')

    if args.task == 'embedding':
        assert args.dataset_dir is not None, 'Please specify out_dir for getting embeddings'
        clf.get_all_user_embeddings_parallel()
    elif args.task == 'prediction':
        assert args.out_dir is not None, 'Please specify out_dir for prediction result'
        clf.predict()
    elif args.task == 'experiment':
        clf.run_timeline_model(class_weight='balanced')
    else:
        raise NotImplementedError

    print("Job done in {} seconds".format(time.time() - start))

    # run_description_model()
    # get_all_user_embeddings_parallel()
    # in_fn = sys.argv[-1]
    # predict(model_fn=best_model_fn, in_fn=os.path.join(out_dir, in_fn))