# encoding = utf8
import gzip
import json
import glob
import nltk
from collections import defaultdict
import os
import re
import time
import pandas as pd
from optparse import OptionParser
import emot
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


def intersectionTokens(set1, set2):
    return set1.intersection(set2)


def collectFeaturesFromTimeline(filename):
    user_feature = defaultdict(dict)
    tknzr = TweetTokenizer()
    tknzr_reduce = TweetTokenizer(strip_handles=True, reduce_len=True)
    stop_words = set(stopwords.words('english'))
    with gzip.open(filename, 'r') as inf:
        line_count = 0
        num_contr = 0
        num_tokens = 0
        # key: emoticon text, value: occurrence
        emoticon_use = defaultdict(int)
        emoji_use = defaultdict(int)
        num_ht = 0
        num_with_ht_tweets = 0
        all_tokens_set = set()
        reduced_all_tokens_set = set()
        tweet_text = []
        pos_bigrams = defaultdict(int)
        pos_trigrams = defaultdict(int)
        hashtags_occur = defaultdict(int)

        for line in inf:
            line_count += 1
            tweet = json.loads(line.decode('utf8'))
            userid_str = tweet['user']['id_str']

            # filtering section
            # English and non-retweet
            if tweet['lang'] != 'en':
                continue
            if 'retweeted_status' in tweet.keys():
                continue

            # note: need to get untruncated tweets
            text = tweet['full_text']
            if text == '':
                continue
            tweet_text.append(text)

            # Tokenize
            # tokenize has preserve_case option default as True
            tokens = tknzr.tokenize(text)
            num_tokens += len(tokens)
            all_tokens_set.update(set(tokens))

            if len(tweet['entities']['hashtags']) is not 0:
                num_with_ht_tweets += 1
                for ht in tweet['entities']['hashtags']:
                    hashtags_occur[ht['text']] += 1
            num_ht += len(tweet['entities']['hashtags'])

            # lexical diversity number of tokens in a tweet without URLs, user mention and stopwords divided by the total number of tokens
            text_without_url = re.sub(r"http\S+", "", text)
            reduced_tokens = tknzr_reduce.tokenize(text_without_url)
            # eliminate :
            if reduced_tokens[0] == ':':
                reduced_tokens.pop(0)
            reduced_tokens_set = set(reduced_tokens)

            for sw in intersectionTokens(reduced_tokens_set, stop_words):
                # note: remove could raise keyerror
                reduced_tokens_set.remove(sw)

            # order doesn't matter here
            reduced_all_tokens_set.update(reduced_tokens_set)

            # lots of :/ with URL in the text
            emoticons_text = emot.emoticons(text_without_url)
            if type(emoticons_text) == dict:
                for ele in emoticons_text['value']:
                    emoticon_use[ele] += 1
            emoji_text = emot.emoji(text)
            for ele in emoji_text['value']:
                emoji_use[ele] += 1

            # features need to loop over all the tokens
            # syntactic features and contractions
            # pos tagging
            tags = nltk.pos_tag(tokens)
            for i in range(len(tokens)):
                if tags[i][0] in contractions:
                    num_contr += 1
                # For noun's-like contractions
                if tags[i][1] == 'NN' or tags[i][1] == 'NNP':
                    if len(tags[i][0]) <= 3:
                        continue
                    if tags[i][0][-2:] == "'s" or tags[i][0][-3:] == "'re":
                        if i + 1 < len(tokens):
                            if tags[i + 1][1] == 'VBN' and tags[i + 1][1] == 'VBG':
                                num_contr += 1

                # not including BOS
                if i < len(tokens) - 1:
                    pos_bigrams[(tags[i][1], tags[i + 1][1])] += 1
                if i < len(tokens) - 2:
                    pos_trigrams[(tags[i][1], tags[i + 1][1], tags[i + 2][1])] += 1

    # line_count is num of Tweets Got
    if line_count == 0 or num_tokens == 0:
        return {}
    user_feature[userid_str]['scraped_tweets_num'] = line_count

    # NOTE: store the user information for behavioral analysis, store the last tweet per user
    with open(os.path.join(outdir, "{}_user_unique.json".format(group)), 'a', encoding='utf8') as uniquef:
        uniquef.write("{}\n".format(json.dumps(tweet)))

    # Collect features here
    # Type-Token Ratio
    user_feature[userid_str]['type_token_ratio'] = len(all_tokens_set) / num_tokens

    # Usage of Contractions
    user_feature[userid_str]['num_contractions_per_tweet'] = num_contr / line_count

    # lexical diversity
    user_feature[userid_str]['lexical_diversity'] = len(reduced_all_tokens_set) / len(all_tokens_set)

    # hashtags: hashtag occurrence and use by num of users
    user_feature[userid_str]['hashtags_occur'] = hashtags_occur
    hashtags_user = {}
    for key in hashtags_occur.keys():
        hashtags_user[key] = 1
    user_feature[userid_str]['hashtags_user'] = hashtags_user
    user_feature[userid_str]['num_ht_per_tweet'] = num_ht / line_count

    # emoticons and emoji
    user_feature[userid_str]['emoticon_use'] = emoticon_use
    user_feature[userid_str]['emojis'] = emoji_use

    # syntactic features
    user_feature[userid_str]['pos_bigrams'] = pos_bigrams
    user_feature[userid_str]['pos_trigrams'] = pos_trigrams

    # store the text for intangible feature use
    if os.path.exists(os.path.join(outdir, 'tmp/{}/{}.txt'.format(group, userid_str))):
        return user_feature
    with open(os.path.join(outdir, 'tmp/{}/{}.txt'.format(group, userid_str)), 'w',
              encoding='utf8') as outf:
        for tt in tweet_text:
            # NOTE: replace \n with space in text
            tt = tt.replace("\n", ".")
            assert "\n" not in tt
            outf.write('{}\n'.format(tt))

    return user_feature


def loadCSV(fn):
    """
    Return the first column of a csv file as a dictionary with values equal to 1
    """
    contraction = defaultdict(int)
    with open(fn, 'r') as inf:
        for line in inf:
            key = line.strip().split(',')[0]
            contraction[key] = 1
    return contraction


def readIdFromTxt(fn):
    dic = {}
    with open(fn, 'r') as inf:
        for line in inf:
            id = line.strip()
            dic[id] = True
    return dic


def groupLevelFeatures(users):
    """
    Feature list:
    Lexical features:
      lexical richness: type-token ratio, lexical diversity, contractions, emoticon/emojis
    Syntactic features:
      most frequent tagging bigrams/trigrams: top 10
    Topic analysis:
      top hashtag analysis: top k
    """
    scraped_tweets_num = []
    type_token_ratio = []
    num_contractions_per_tweet = []
    lexical_diversity = []
    num_ht_per_tweet = []
    consistent_ids = []
    for user_id in users.keys():
        consistent_ids.append(user_id)
        user_features = users[user_id]
        # features in number
        scraped_tweets_num.append(user_features['scraped_tweets_num'])
        type_token_ratio.append(user_features['type_token_ratio'])
        num_contractions_per_tweet.append(user_features['num_contractions_per_tweet'])
        lexical_diversity.append(user_features['lexical_diversity'])
        num_ht_per_tweet.append(user_features['num_ht_per_tweet'])

    # output the results
    numerical_data = {
        "user_id": consistent_ids,
        "type_token_ratio": type_token_ratio,
        "num_contractions_per_tweet": num_contractions_per_tweet,
        "lexical_diversity": lexical_diversity,
        "num_ht_per_tweet": num_ht_per_tweet,
        "scraped_tweets_num": scraped_tweets_num
    }
    # NOTE: writing mode is a due to parallelization
    df_numerical = pd.DataFrame(numerical_data)
    df_numerical = df_numerical.astype(float)
    df_numerical.to_csv(os.path.join(outdir, '{}_numerical_result.csv'.format(group)), index=False, mode='a',
                        float_format='%.5f', sep='\t')

    # # Top list features:
    # for writing_feature in ['hashtags_occur', 'emoticon_use', 'hashtags_user']:
    for writing_feature in ['hashtags_occur', 'emoticon_use', 'emojis', 'pos_trigrams', 'pos_bigrams', 'hashtags_user']:
        with open(os.path.join(outdir, '{}_{}.txt'.format(group, writing_feature)), 'w') as outf:
            for tuple_item in selectTopItems(users, feature_str=writing_feature):
                if type(tuple_item[0]) == str:
                    # hashtags, *_emoticon_use, emojis
                    outf.write('{}\t{}\n'.format(tuple_item[0], str(tuple_item[1])))
                elif type(tuple_item[0]) == tuple:
                    # pos_bigrams and pos_trigrams
                    # NOTE item and count separated by \t, sub item separated by ,
                    outf.write('{}'.format(','.join(tuple_item[0])))
                    outf.write("\t{}\n".format(str(tuple_item[1])))
    return True


def selectTopItems(users, feature_str):
    collected_dict = defaultdict(int)
    for userid, features in users.items():
        for item, item_val in features[feature_str].items():
            collected_dict[item] += item_val
    sorted_list = sorted(collected_dict.items(), key=lambda dic_item: dic_item[1], reverse=True)
    return sorted_list


def loadTweets(fndir):
    """
    Load tweets from json.gz file line by line.
    :return dictionary with key as tweet id_str, value as tweet object
    """
    tweet_files = glob.glob(os.path.join(fndir, 'out.tweets*.gz'))
    print("get {} full text files".format(len(tweet_files)))
    tweets = defaultdict(dict)
    for tf in tweet_files:
        with gzip.open(tf, 'r') as inf:
            for line in inf:
                tweet = json.loads(line.decode('utf-8'))
                assert tweet['id_str'] not in tweets.keys()
                tweets[tweet['id_str']] = tweet['full_text']
    return tweets


parser = OptionParser()
parser.add_option('--group', dest='Group', help='gender or race', metavar='TASK')
parser.add_option('--sourcedir', type='str', help='source directory for group analysis, timeline files',
                  dest='SourceDir', metavar='SOURCEDIR')
parser.add_option('--outdir', type='str', help='output directory for group analysis, txt for list-based features, '
                                               'csv for numerical features', dest='OutDir', metavar='OUTDIR')
(options, args) = parser.parse_args()

print('loading contractions')
contractions = loadCSV('src/group_analysis/contractions.csv')

group = options.Group
sourcedir = options.SourceDir
outdir = options.OutDir
try:
    # os.makedirs(os.path.join(outdir, group))
    os.makedirs(os.path.join(outdir, 'tmp', group))
except OSError:
    print("Output directory {} already exists".format(os.path.join(outdir, group)))
    pass

files = glob.glob(sourcedir + '{}/*.json.gz'.format(group))
files.sort()
print("{} files in source directory".format(len(files)))
# TODO: if statuses files are stored properly, it doesn't need to check the ids.

users = defaultdict(dict)
start = time.time()
print('Start processing {} group at {}'.format(group, time.strftime('%X %x %Z')))
processed = 0
for f in files:
    users.update(collectFeaturesFromTimeline(f))
    processed += 1
    if processed % 200 == 0:
        print("Done {} {} users in {} seconds".format(processed, group, time.time() - start))
        break

# Collect stats for each group
print('Collecting group level features for {}'.format(group))
df = groupLevelFeatures(users)
print("all done")
