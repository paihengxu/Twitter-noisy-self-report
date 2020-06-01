#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, nltk, os, gzip, time
import sys
import glob
import string

import demographer
from demographer.indorg import IndividualOrgDemographer
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import MWETokenizer
from collections import defaultdict

MALE = ['male', 'father', 'uncle', 'brother', 'papa', 'mr.', 'mister', 'dad', 'husband', 'boy', 'man', 'he_/_him',
        'he_|_him', 'guy', 'dude']
FEMALE = ['female', 'mother', 'mom', 'lady', 'mama', 'sister', 'ms.', 'mrs.', 'wife', 'girl', 'woman', 'she_/_her',
          'she_|_her', 'gal']

LATINO = ['latino', 'hispanic', 'latina', 'latin', 'latinx']
BLACK = ['black', 'african american']
WHITE = ['white', 'caucasian']
ASIAN = ['asian']


def readUpdateJsongz(filename):
    dic = {}
    with gzip.open(filename, 'r') as inf:
        for line in inf:
            data = json.loads(line.decode('utf-8'))
            dic.update(data)
    return dic


def scoreTFIDF(tf, idf):
    return tf * np.log(idf)


def save_json_gz(dictionary, fn, mode='w'):
    try:
        with gzip.open(fn, mode) as outf:
            outf.write("{}\n".format(json.dumps(dictionary)).encode('utf8'))
    except Exception as exp:
        print('write {} failed, {}'.format(fn, exp))


class noisySelfReport:
    def __init__(self, task, filter_params, win, thre, tfidf, sourcefile, outdir, eval=False,
                 race_dev_set_fn='src/manual_labeled/race_dev_set.json.gz',
                 gender_dev_set_fn='src/manual_labeled/gender_dev_set.json.gz',
                 sword_freq_noun_fn='src/self_reporty_words/refined_selfreporty_noun_freq.json.gz',
                 sword_count_noun_fn='src/self_reporty_words/refined_selfreporty_noun_count.json.gz',
                 sword_freq_adj_fn='src/self_reporty_words/refined_selfreporty_adj_freq.json.gz',
                 sword_count_adj_fn='src/self_reporty_words/refined_selfreporty_adj_count.json.gz',
                 rule_out_bigram='src/manual_labeled/rule_out_bigrams.json.gz',
                 power=1):
        # gender or race
        self.task = task

        # query words for the task
        if self.task == 'race':
            self.keywords = LATINO + BLACK + WHITE + ASIAN
        elif self.task == 'gender':
            self.keywords = MALE + FEMALE

        # hyperparameters, window size, threshold
        assert win % 2 == 1, "window size around the query words has to be odd"
        self.win = win
        self.thre = thre
        self.power = power

        # decide which ad-hoc filters to apply
        self.filter_params = filter_params
        self.sourcefile = sourcefile
        self.outdir = outdir

        # Use tfidf weighting strategy or simple co-occurrence weighting
        self.tfidf = tfidf

        print('Loading selfreporty words collection')
        self.swords_count = readUpdateJsongz(sword_count_noun_fn)
        self.swords_frequency = readUpdateJsongz(sword_freq_noun_fn)
        swords_count_adj = readUpdateJsongz(sword_count_adj_fn)
        swords_frequency_adj = readUpdateJsongz(sword_freq_adj_fn)
        self.swords_count.update(swords_count_adj)
        self.swords_frequency.update(swords_frequency_adj)

        if self.task == 'race':
            print('Loading rule-out bigrams for race task')
            self.rule_out_bigram = readUpdateJsongz(rule_out_bigram)

        self.eval = eval
        if self.eval:
            print("Loading dev set")
            if self.task == 'race':
                self.dev_set = readUpdateJsongz(race_dev_set_fn)
            elif self.task == 'gender':
                self.dev_set = readUpdateJsongz(gender_dev_set_fn)
        # recording filter stats
        self.filter_stats = defaultdict(int)
        self.filter_stats_user = defaultdict(int)

        # generate file names based on filters applied
        all_filter_keys = sorted(list(self.filter_params.keys()))
        self.filter_applied = [key for key in all_filter_keys if self.filter_params[key]]

        # value for calculating tfidf scores
        self.sum_count = sum(self.swords_count.values())

    def __preProcessedTweet__(self, user):
        """
        Determine whether a user should be filtered out.
        """
        filtered = False

        tknzr = TweetTokenizer()
        if self.task == 'race':
            MWEtknzr = MWETokenizer([('african', 'american')])
        else:
            MWEtknzr = MWETokenizer([('he', '/', 'him'), ('she', '/', 'her'), ('he', '|', 'him'), ('she', '|', 'her')])
        if user['description'] is None:
            filtered = True
            return filtered, [], []

        des = user['description'].lower().replace('\n', '.')  # lower case
        destoken = MWEtknzr.tokenize(tknzr.tokenize(des))
        tags = nltk.tag.pos_tag(destoken)

        # note: may throw out some good users
        fireQuery = []

        colors = ['white', 'black', 'red', 'green', 'yellow', 'blue', 'brown', 'pink', 'purple', 'orange', 'gray']
        for q in self.keywords:
            if q in destoken:
                if q in ['he_/_him', 'he_|_him', 'she_/_her', 'she_|_her']:
                    continue
                ind = destoken.index(q)
                query_filtered = False
                # Color filter
                if self.filter_params['color']:
                    if q == 'black' or q == 'white':
                        if len(self.__getSameElement__(colors, destoken, q)):
                            filtered = True
                            self.filter_stats['c_color'] += 1

                # Plural filter
                if self.filter_params['plural']:
                    if len(destoken) > ind + 1:
                        if tags[ind][1] == 'JJ' and (tags[ind + 1][1] == 'NNPS' or tags[ind + 1][1] == 'NNS'):
                            query_filtered = True
                            self.filter_stats['c_plural'] += 1

                # bigram filter
                if self.filter_params['bigram']:
                    if len(destoken) > ind + 1:
                        if (destoken[ind], destoken[ind + 1]) in self.rule_out_bigram:
                            query_filtered = True
                            self.filter_stats['c_bigram'] += 1

                # quote filter
                # globally remove quote, considering quotes only around single query would be rare
                if self.filter_params['quote']:
                    if "\"" in destoken:
                        start = destoken.index("\"")
                        if "\"" in destoken[(start + 1):]:
                            end = destoken[(start + 1):].index("\"") + start + 1
                            if end > ind > start and end - start != 2:
                                query_filtered = True
                                self.filter_stats['c_quote'] += 1

                # filter for "dad of daughter"
                if self.filter_params['of']:
                    ignore = True
                    if "of" in destoken:
                        of_idx = destoken.index("of")
                        if of_idx >= 1 and destoken[of_idx - 1] not in self.keywords:
                            continue
                        if ind > of_idx:
                            tag_set = set()
                            # for tag between query and of
                            for i in range(of_idx + 1, ind):
                                if destoken[i] in string.punctuation:
                                    # if punc in between, the query is outside the scope, don't filter it
                                    ignore = False
                                    break
                                tag_set.add(tags[i][1])
                            if ignore and tag_set.issubset({'CC', 'JJ', 'DT', 'CD', 'JJS'}):
                                query_filtered = True
                                self.filter_stats['c_of'] += 1

                # filter for possessive pronouns
                if self.filter_params['possessive_pronouns']:
                    ignore = True
                    only_tag = [tag[1] for tag in tags]
                    if 'PRP$' in only_tag:
                        tag_set = set()
                        prp_idx = only_tag.index('PRP$')
                        if ind > prp_idx:
                            for i in range(prp_idx + 1, ind):
                                if destoken[i] in string.punctuation:
                                    ignore = False
                                    break
                                tag_set.add(tags[i][1])
                            if ignore and tag_set.issubset({'CC', 'JJ', 'DT', 'CD', 'JJS'}):
                                query_filtered = True
                                self.filter_stats['c_possessive_pronouns'] += 1
                # filter out specific query
                if self.task == 'gender' and not query_filtered:
                    fireQuery.append(q)
                elif self.task == 'race':
                    fireQuery.append(q)

                # only color
                if self.task == 'race' and (filtered or query_filtered):
                    self.filter_stats['c_filter'] += 1
                    return True, [], []

        return filtered, fireQuery, destoken

    def __labelExactUsersWithScore__(self, user, fireQuery, destoken):
        # NOTE: process one user at a time
        userWithScore = {}
        userWithScore['demographics'] = []
        user_id = user['id_str']
        userWithScore['id_str'] = user_id
        for q in fireQuery:
            # strong indicators for gender
            if q in ['he_/_him', 'he_|_him', 'she_/_her', 'she_|_her']:
                score = self.thre + 1
            else:
                score, sfwords = self.__selfreporty__(destoken, q)
            if self.task == 'race':
                if q in LATINO:
                    userWithScore['demographics'].append(('latin', score))
                elif q in WHITE:
                    userWithScore['demographics'].append(('white', score))
                elif q in ASIAN:
                    userWithScore['demographics'].append(('asian', score))
                elif q in BLACK:
                    userWithScore['demographics'].append(('black', score))
            else:
                # note: same-group query appears multiple times.
                if q in MALE:
                    if len(userWithScore['demographics']):
                        for group_score in userWithScore['demographics']:
                            if group_score[0] == 'male':
                                score += group_score[1]
                                userWithScore['demographics'].remove(group_score)
                    userWithScore['demographics'].append(('male', score))
                elif q in FEMALE:
                    if len(userWithScore['demographics']):
                        for group_score in userWithScore['demographics']:
                            if group_score[0] == 'female':
                                score += group_score[1]
                                userWithScore['demographics'].remove(group_score)
                    userWithScore['demographics'].append(('female', score))

        if len(userWithScore['demographics']):
            userWithScore['description'] = user['description']
            return userWithScore, user_id
        else:
            return {}, user_id

    def __evalSelfreportyScore__(self, dic):
        users_labeled = {}
        for k, v in dic.items():
            bestScore = self.thre
            bestInd = -1
            for ele in v['demographics']:
                index = v['demographics'].index(ele)
                if ele[1] >= bestScore:
                    bestScore = ele[1]
                    bestInd = index
            if bestInd >= 0:
                users_labeled[k] = v['demographics'][bestInd][0]
        return users_labeled

    def __getSameElement__(self, list1, list2, query):
        # Now globally matching color term
        set1 = set(list1)
        set1.remove(query)
        set2 = set(list2)
        iset = set1.intersection(set2)
        return list(iset)

    def __selfreporty__(self, tokens, query):
        # get the index of query
        score = .0
        ind = tokens.index(query)
        words = []
        for i in range(self.win):
            index = ind - int((self.win - 1) / 2) + i
            if index < 0 or index >= len(tokens):
                continue
            if tokens[index] in self.swords_frequency and tokens[index] != query:
                dist = abs(ind - index)
                if self.tfidf:
                    s = scoreTFIDF(self.swords_frequency[tokens[index]],
                                   self.sum_count / float(self.swords_count[tokens[index]]))
                    score += (1.0 / dist ** self.power) * s
                else:
                    score += (1.0 / (dist ** self.power)) * self.swords_frequency[tokens[index]]
                words.append(tokens[index])
        return score, words

    def calculateSelfReportyScore(self):
        print('Starting calculating self-reporty scores')
        d = IndividualOrgDemographer()

        processed = 0
        org_num = 0
        start = time.time()

        saved_path = os.path.join(self.outdir, 'user_score_window{}_power{}_'.format(self.win, self.power) +
                                  '_'.join(self.filter_applied) + '.json.gz')
        # if file exists, return false
        assert not os.path.exists(saved_path), \
            'Self-reporty score in this hyper-parameter setting already exists in {}, please save it or delete it.'.format(
                saved_path)

        with gzip.open(self.sourcefile, 'r') as inf:
            try:
                line = inf.readline()
            except Exception as err:
                print(err)
            while line:
                processed += 1
                if processed % 100000 == 0:
                    print("{} seconds for {} tweet records".format(time.time() - start, processed))
                tweet = json.loads(line.decode('utf8'))
                try:
                    indorg = d.process_tweet(tweet)
                except Exception as err:
                    print(err)
                    print(tweet)
                    try:
                        line = inf.readline()
                    except Exception as err:
                        print(err)
                    continue
                if indorg['indorg']['value'] == 'org':
                    org_num += 1
                    # print("Org, pass")
                    try:
                        line = inf.readline()
                    except Exception as err:
                        print(err)
                    continue
                users = [tweet['user']]
                for key in ['retweeted_status', 'quoted_status']:
                    if key in tweet.keys():
                        users.append(tweet[key]['user'])
                for user in users:
                    # note: for processed user, if one user is quoted or retweets before, may not get the latest info.
                    filtered, fireQuery, destoken = self.__preProcessedTweet__(user)
                    if filtered:
                        continue

                    userWithScore, userid = self.__labelExactUsersWithScore__(user, fireQuery, destoken)
                    if len(userWithScore) == 0:
                        # print("no valid score")
                        continue

                    save_json_gz(userWithScore, saved_path, mode='a')
                try:
                    line = inf.readline()
                except Exception as err:
                    print(err)

        print("{} seconds for {} tweet records".format(time.time() - start, processed))
        print("{} org records in total {} tweet records".format(org_num, processed))
        print("Number of users that fired for each filter:")
        print(self.filter_stats)

    def __readUserWithScore__(self, fn):
        """
         read in format consistent with output format with userWithScore
        """
        users = {}
        del_num = 0
        with gzip.open(fn, 'r') as inf:
            for line in inf:
                d = json.loads(line.decode('utf8'))
                if d['id_str'] in users.keys():
                    del_num += 1
                users[d['id_str']] = d
        print("{} duplicate users in {}, we keep the latest bio".format(del_num, fn))
        return users

    def labelUsers(self):
        # note: users with different labels will get the latest score
        usersWithScore = self.__readUserWithScore__(os.path.join(self.outdir,
                                                                 'user_score_window{}_power{}_'.format(self.win,
                                                                                                       self.power) + '_'.join(
                                                                     self.filter_applied) + '.json.gz'))
        print("{} unique users".format(len(usersWithScore)))
        usersLabeled = self.__evalSelfreportyScore__(usersWithScore)

        save_json_gz(usersLabeled, os.path.join(self.outdir,
                                                'user_label_window{}_power{}_thre{}_'.format(self.win, self.power,
                                                                                             self.thre) + '_'.join(
                                                    self.filter_applied) + '.json.gz'))
        if self.eval:
            self.outputResult(self.outdir,
                              'user_label_window{}_power{}_thre{}_'.format(self.win, self.power, self.thre) + '_'.join(
                                  self.filter_applied) + '.json.gz',
                              datasetname='{}_dev'.format(self.task), dataset=self.dev_set)

    def __evalDataset__(self, data, test):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for k, v in test.items():
            # yes and selfreport
            if k in data:
                if v == 'non':
                    fp += 1
                else:
                    if data[k] == v:
                        tp += 1
                    else:
                        fp += 1
            else:
                # true negative: test: no, us: not exist
                if v == 'non':
                    tn += 1
                # false negative: test: selfreport, us: not exist
                else:
                    fn += 1
        print('For all race, true positive: {}'.format(tp))
        print('For all race, false positive: {}'.format(fp))
        print('For all race, true negative: {}'.format(tn))
        print('For all race, false negative: {}'.format(fn))
        stats = [tp, tn, fp, fn]
        try:
            precision = float(tp) / (tp + fp)
            recall = float(tp) / (tp + tn)
            f1 = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError as testerr:
            precision = 0
            recall = 0
            f1 = 0
            print(testerr)
        print("pre: {}, recall {}, f1 {}".format(precision, recall, f1))
        return precision, recall, f1, stats

    def outputResult(self, file_dir, file, datasetname, dataset):
        data = readUpdateJsongz(os.path.join(file_dir, file))
        name = file.strip('.json.gz').split('_')
        pre, recall, f1, stats = self.__evalDataset__(data, dataset)
        filename = '_eval_result_tfidf.txt' if self.tfidf else '_eval_result.txt'
        with open(os.path.join(self.outdir, datasetname + filename), 'a') as outf:
            # Write the hyperparameters and their values
            outf.write(' '.join(name[2:5]) + ' ')
            outf.write(' '.join([format(pre, '5.4f'), format(recall, '5.4f'), format(f1, '5.4f')]) + ' ')
            outf.write(' '.join([str(ele) for ele in stats]))
            outf.write(' ' + str(len(data)))
            outf.write('\n')
