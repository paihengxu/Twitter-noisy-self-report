#!/usr/bin/env python
# encoding: utf-8
import json, gzip, nltk, re, io, os
import argparse
import glob
import time
from nltk.tokenize import TweetTokenizer
from collections import defaultdict

QUERY = {'i am', 'i\'m', 'ima'}
COLLECT_TAG = 'adj'
OUTDIR = '/tmp/outdir/'
RESULT_OUTDIR = '/result/outdir/'

# nltk.download('averaged_perceptron_tagger')

def collectIamaFrequency_nltk(filename):
    words_des = defaultdict(int)
    swords_des = defaultdict(int)
    words_all = defaultdict(int)
    swords_all = defaultdict(int)
    tknzr = TweetTokenizer()
    i = 0
    with open(filename, 'r') as inf:
        try:
            line = inf.readline()
            while line:  # 176943345 records
                try:
                    if i % 100000 == 0:
                        print('{} record already at {}'.format(i, time.ctime()))
                    line = line.strip().split(":", 1)[1]
                    field, text = line.split(":", 1)
                    field = field.strip('"')
                    text = text.strip('"')
                    text = text.lower().replace('/', ',').replace('|', ',')  # lower case
                    tokens = tknzr.tokenize(text)  # preprocess
                    tags = nltk.tag.pos_tag(tokens)
                    for t in tokens:
                        words_all[t] += 1
                        if field == 'description':
                            words_des[t] += 1
                    # TODO: Get multiple words entity later
                    for q in QUERY:
                        if q in text:
                            q_token = tknzr.tokenize(q)
                            for w in getIamaWords(q_token, tokens, tags):
                                # filter the test without english words
                                if bool(re.search('[a-z0-9]', w)):
                                    if field == 'description':
                                        swords_des[w] += 1
                                    swords_all[w] += 1
                    line = inf.readline()
                    i += 1
                except Exception as er:
                    print(er)
                    line = inf.readline()
                    continue
        except Exception as err:
            print(err)

    print("processed {} lines in this thread".format(str(i)))

    return swords_des, words_des, swords_all, words_all


def getIamaWords(q_token, tokens, tags):
    # potentially multiple I am a in the description.
    swords = []
    lq = len(q_token)
    # (i, e) in the enumerator, only get index for e==first term of q_token
    for ind in (i for i, e in enumerate(tokens) if e == q_token[0]):
        if tokens[ind:ind + lq] == q_token:
            if COLLECT_TAG == 'noun':
                if ind + lq + 1 < len(tokens):
                    # simply matched the following one
                    # swords.append(tokens[ind+lq])
                    if tags[ind + lq][1] == 'NN':
                        swords.append(tokens[ind + lq])
                    if tags[ind + lq][1] == 'JJ' and tags[ind + lq + 1][1] == 'NN':
                        swords.append(tokens[ind + lq + 1])
                    if tags[ind + lq][1] == 'DT' and tags[ind + lq + 1][1] == 'NN':
                        swords.append(tokens[ind + lq + 1])
                if ind + lq + 2 < len(tokens):
                    if tags[ind + lq][1] == 'RB' and tags[ind + lq + 1][1] == 'DT' and tags[ind + lq + 2][1] == 'NN':
                        swords.append(tokens[ind + lq + 2])
                    if tags[ind + lq][1] == 'DT' and tags[ind + lq + 1][1] == 'JJ' and tags[ind + lq + 2][1] == 'NN':
                        swords.append(tokens[ind + lq + 2])
                if ind + lq + 3 < len(tokens):
                    if tags[ind + lq][1] == 'RB' and tags[ind + lq + 1][1] == 'DT' and tags[ind + lq + 2][1] == 'JJ' and \
                        tags[ind + lq + 3][1] == 'NN':
                        swords.append(tokens[ind + lq + 3])
            elif COLLECT_TAG == 'adj':
                if ind + lq + 1 < len(tokens):
                    # simply matched the following one
                    # swords.append(tokens[ind+lq])
                    if tags[ind + lq][1] == 'JJ' and tags[ind + lq + 1][1] == 'NN':
                        swords.append(tokens[ind + lq])
                if ind + lq + 2 < len(tokens):
                    if tags[ind + lq][1] == 'DT' and tags[ind + lq + 1][1] == 'JJ' and tags[ind + lq + 2][1] == 'NN':
                        swords.append(tokens[ind + lq + 1])
                if ind + lq + 3 < len(tokens):
                    if tags[ind + lq][1] == 'RB' and tags[ind + lq + 1][1] == 'DT' and tags[ind + lq + 2][1] == 'JJ' and \
                        tags[ind + lq + 3][1] == 'NN':
                        swords.append(tokens[ind + lq + 2])
    return swords


def save_json_gz(OUT_DIR, dictionary, outname):
    try:
        with gzip.open(os.path.join(OUT_DIR + outname + '.json.gz'), 'w') as outf:
            outf.write("{}\n".format(json.dumps(dictionary)).encode('utf8'))
    except Exception as exp:
        print('write {} failed, {}'.format(outname, exp))


def aggregate(file_pattern):
    """
    aggregate result dict from parallel tmp files
    """
    files = glob.glob(OUTDIR + file_pattern)
    data = defaultdict(int)
    for fn in files:
        print("aggregating {}".format(fn))
        with gzip.open(fn, 'r') as inf:
            for line in inf:
                d = json.loads(line.decode('utf8'))
                for key, value in d.items():
                    data[key] += value
    save_json_gz(RESULT_OUTDIR, data, file_pattern.replace("_*.json.gz", ""))
    return data


def getFrequency(sfwords, words, filename):
    """
    get frequency of selfreporty word, dividing its occurrence as self report by all occurrence
    """
    swords_frequency = defaultdict(float)
    for (ke, va) in sfwords.items():
        swords_frequency[ke] = va / words[ke]

    save_json_gz(RESULT_OUTDIR, swords_frequency, filename)
    return swords_frequency


def sortDictAndWriteToTxt(dic_count, dic_freq, fn):
    swords_list = sorted(dic_count.items(), key=lambda dic_count: dic_count[1], reverse=True)
    print('Writing to txt files.')
    with io.open(fn, 'w', encoding='UTF-8') as outf_des:
        for ele in swords_list:
            try:
                outf_des.write('\t'.join([ele[0], str(ele[1]), str(dic_freq[ele[0]])]))
            except Exception as err:
                print(err)
                continue
            outf_des.write(u'\n')


if __name__ == '__main__':

    # sourcefile = '/export/c10/zach/data/demographics/iama.json.gz'
    parser = argparse.ArgumentParser()
    parser.add_argument("job_name", type=str)
    parser.add_argument("job_num", type=int)
    parser.add_argument("num_jobs", type=int)
    args = parser.parse_args()
    num_jobs = args.num_jobs
    job_num = args.job_num
    job_name = args.job_name

    file_num = -1
    if job_name == 'collect':
        # source files are divided to run the collection parallelly
        files = glob.glob('/path/to/source/files/??')
        files.sort()
        assert len(files) == num_jobs
        for fn in files:
            file_num += 1

            # parallelization purposes
            if file_num % num_jobs != args.job_num:
                continue

            print("processing {}".format(fn))
            swords_des, words_des, swords_all, words_all = collectIamaFrequency_nltk(fn)
            print('swords count in all text is {}'.format(sum(swords_all.values())))
            print('swords count in description only is {}'.format(sum(swords_des.values())))

            print('Saving json.gz files.')
            save_json_gz(OUTDIR, swords_des,
                         'selfreporty_{}_{}_in_description_{}'.format(COLLECT_TAG, 'count', job_num))
            # save_json_gz(OUTDIR, swords_frequency_des, 'selfreporty_{}_{}_in_description_{}'.format(COLLECT_TAG, 'frequency', job_num))
            save_json_gz(OUTDIR, swords_all, 'selfreporty_{}_{}_in_alltext_{}'.format(COLLECT_TAG, 'count', job_num))
            save_json_gz(OUTDIR, words_des, 'all_words_count_in_description_{}'.format(job_num))
            save_json_gz(OUTDIR, words_all, 'all_words_count_in_alltext_{}'.format(job_num))
            # save_json_gz(OUTDIR, swords_frequency_all, 'selfreporty_{}_{}_in_alltext_{}'.format(COLLECT_TAG, 'frequency', job_num))

    elif job_name == 'aggregate':
        sf_words_des = aggregate('selfreporty_{}_{}_in_{}_*.json.gz'.format(COLLECT_TAG, 'count', 'description'))
        words_des = aggregate('all_words_count_in_{}_*.json.gz'.format('description'))
        sf_words_all = aggregate('selfreporty_{}_{}_in_{}_*.json.gz'.format(COLLECT_TAG, 'count', 'alltext'))
        words_all = aggregate('all_words_count_in_{}_*.json.gz'.format('alltext'))

        print("get frequency for words in description")
        sf_words_des_freq = getFrequency(sf_words_des, words_des,
                                         filename='selfreporty_{}_{}_in_{}'.format(COLLECT_TAG, 'frequency',
                                                                                   'description'))
        print("get frequency for words in all text")
        sf_words_all_freq = getFrequency(sf_words_all, words_all,
                                         filename='selfreporty_{}_{}_in_{}'.format(COLLECT_TAG, 'frequency', 'alltext'))

        sortDictAndWriteToTxt(sf_words_des, sf_words_des_freq,
                              fn='selfreporty_{}_count_sorted_in_{}.txt'.format(COLLECT_TAG, 'description'))
        sortDictAndWriteToTxt(sf_words_all, sf_words_all_freq,
                              fn='selfreporty_{}_count_sorted_in_{}.txt'.format(COLLECT_TAG, 'alltext'))

    print('ALL DONE!')
