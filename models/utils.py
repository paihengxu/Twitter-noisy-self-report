import gzip
import json
import numpy as np


def read_user_ids(fn):
    user_ids = []
    with open(fn, 'r') as inf:
        for line in inf:
            user_ids.append(line.strip())
    return user_ids


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
