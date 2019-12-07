import math
import pandas as pd
import sys
import numpy as np
from collections import defaultdict
from scipy.stats import mannwhitneyu


def readList(fn, k):
    l = []
    count = 0
    with open(fn, 'r', encoding='utf8') as inf:
        for line in inf:
            d = line.strip().split('\t')
            l.append(d[0])
            count += 1
            if count == k:
                break
    return l


def readTxt(fn):
    score = []
    with open(fn, 'r') as inf:
        for line in inf:
            score.append(float(line.strip()))
    return np.asarray(score)


def concordant(item1, item2, list1, list2):
    """
    judge item1 and item2 are concordant pair
    """
    assert item1 in list1
    assert item2 in list2
    if item1 not in list2 or item2 not in list1:
        return False
    idx_1_in_1 = list1.index(item1)
    idx_1_in_2 = list2.index(item1)

    idx_2_in_1 = list1.index(item2)
    idx_2_in_2 = list2.index(item2)
    if (idx_1_in_1 - idx_2_in_1) * (idx_1_in_2 - idx_2_in_2) > 0:
        return {item1, item2}
    else:
        return False


def cumulativeSum(n):
    if n == 1 or n == 0:
        return 1
    else:
        return n + cumulativeSum(n - 1)


def iterAllItems(list1, list2):
    concordant_num = 0
    pair_num = 0
    concordant_pairs = []
    all_pairs = []
    t1 = 0
    t2 = 0
    # vocab = set(list1)
    # vocab.union(set(list2))
    # print(len(vocab))
    # print(cumulativeSum(len(vocab)))
    for i in list1:
        if i not in list2:
            t1 += 1
        for j in list2:
            # only do this for list2 one time
            if i == list1[0]:
                if j not in list1:
                    t2 += 1
            # literal set
            if {i, j} not in all_pairs and j != i:
                pair_num += 1
                all_pairs.append({i, j})
                if concordant(i, j, list1, list2):
                    concordant_num += 1
                    concordant_pairs.append({i, j})
                    # print(i, j)
    # diff = [ele for ele in all_pairs if ele not in concordant_pairs]
    # print(diff)
    return concordant_num, pair_num, t1, t2


def selectFromDist(dist, lower, upper):
    smaller_dist = []
    for ele in dist:
        if ele >= lower and ele <= upper:
            smaller_dist.append(ele)
    return np.asarray(smaller_dist), len(smaller_dist) / len(dist)


def getTau(p_c, p_d, t_f, t_s):
    tau = (p_c - p_d) / math.sqrt((p_c + p_d + t_f) * (p_c + p_d + t_s))
    return tau


if __name__ == '__main__':
    group = ['asian', 'black', 'latin', 'white']
    task = sys.argv[1]
    if task == 'list':
        k = 10  # top k items
        feature_list = ['emojis', 'emoticon_use', 'pos_bigrams', 'pos_trigrams', 'hashtags_user_lower']
        with open("kendall_result_k{}.txt".format(k), 'w') as outf:
            for f in feature_list:
                outf.write("\nKendall's tau correlation coefficient top {} items in the {} lists\n".format(k, f))
                processed = []
                for idx1, g1 in enumerate(group):
                    for idx2 in range(idx1 + 1, len(group)):
                        g2 = group[idx2]
                        l1 = readList("{}_{}_list.txt".format(g1, f), k)
                        l2 = readList("{}_{}_list.txt".format(g2, f), k)
                        if len(l1) < k:
                            outf.write('k too large for group: {}, feature: {}\n'.format(g1, f))
                            continue
                        if len(l2) < k:
                            outf.write('k too large for group: {}, feature: {}\n'.format(g2, f))
                            continue
                        assert k == len(l1) and k == len(l2)
                        concordant_num, pair_num, t_s, t_f = iterAllItems(l1, l2)
                        outf.write("{g1} vs. {g2}: {tau}\n".format(g1=g1, g2=g2,
                                                                   tau=getTau(concordant_num, pair_num - concordant_num,
                                                                              t_s, t_f)))
                        # print(concordant_num, pair_num, t_s, t_f)
                        processed.append({g1, g2})
    elif task == 'numerical':
        feature_list = ['lexical_diversity', 'num_contractions_per_tweet', 'num_ht_per_tweet', 'scraped_tweets_num',
                        'type_token_ratio']
        u_data = defaultdict(list)
        for idx1, g1 in enumerate(group):
            df1 = pd.read_csv('{}_numerical_result.csv'.format(g1), sep='\t')
            for idx2 in range(idx1 + 1, len(group)):
                g2 = group[idx2]
                df2 = pd.read_csv('{}_numerical_result.csv'.format(g2), sep='\t')
                item_name = "{} vs. {}".format(g1, g2)
                u_data['comparing'].append(item_name)
                for f in feature_list:
                    u, p = mannwhitneyu(df1[f], df2[f], alternative='two-sided')
                    u_data[f + '_u'].append(u)
                    u_data[f + '_p'].append(p)
        df = pd.DataFrame(u_data)
        df.to_csv('mannwhitney_u_test.csv', sep=',', float_format='%.5f', header=True)

    elif task == "intangible":
        feature_list = ['formality', 'polite']
        u_data = defaultdict(list)
        for f in feature_list:
            for idx1, g1 in enumerate(group):
                array1 = readTxt('{}_{}_score.txt'.format(g1, f))

                for idx2 in range(idx1 + 1, len(group)):
                    g2 = group[idx2]
                    array2 = readTxt('{}_{}_score.txt'.format(g2, f))
                    item_name = "{} vs. {}".format(g1, g2)
                    if f == feature_list[0]:
                        u_data['comparing'].append(item_name)
                    u, p = mannwhitneyu(array1, array2, alternative='two-sided')
                    u_data[f + '_u'].append(u)
                    u_data[f + '_p'].append(p)

        df = pd.DataFrame(u_data)
        df.to_csv('mannwhitney_u_test_intangible.csv', sep=',', float_format='%.5f', header=True)

    elif task == "behavior":
        # numerical + mannwhitney_u_test + plot?
        for g in group:
            df = pd.read_csv("{}_behavior_result.csv".format(g), header=0, sep='\t')
            feature_list = list(df.columns)
            for feature in feature_list:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
            with open('summary_behaviour_result.txt', 'a') as outf:
                outf.write("{}\n".format(g))
            df[feature_list].describe().to_csv('summary_behaviour_result.txt',
                                               float_format='%.5f', mode='a', sep='\t')
        print("done collecting behavioral results")

        u_data = defaultdict(list)
        for idx1, g1 in enumerate(group):
            df1 = pd.read_csv('{}_behavior_result.csv'.format(g1), sep='\t')
            # refined_d, proportion = selectFromDist(df1[visual_f], 0, 1000)
            # print(visual_f, g1, proportion)

            for idx2 in range(idx1 + 1, len(group)):
                g2 = group[idx2]
                df2 = pd.read_csv('{}_behavior_result.csv'.format(g2), sep='\t')
                item_name = "{} vs. {}".format(g1, g2)
                u_data['comparing'].append(item_name)
                for f in feature_list:
                    try:
                        u, p = mannwhitneyu(df1[f], df2[f], alternative='two-sided')
                        u_data[f + '_u'].append(u)
                        u_data[f + '_p'].append(p)
                    except ValueError as err:
                        print(err)
                        print(f)
                        continue
        df = pd.DataFrame(u_data)
        df.to_csv('mannwhitney_u_test_behavior.csv', sep=',', float_format='%.5f', header=True)

    else:
        raise ValueError("unknown task for {}".format(task))
