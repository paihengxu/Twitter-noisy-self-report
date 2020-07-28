import gzip
import json
import os
import sys
from argparse import ArgumentParser


def read_user_ids(fn):
    user_set = set()
    with open(fn, 'r') as inf:
        for line in inf:
            user_set.add(line.strip())
    return user_set


def build(user_id_fn, dataset_dir, tweet_dir, ignore_rt=True, tweet_limit=200, user_limit=None):
    # user_set = read_user_ids(fn=os.path.join(dataset_dir, "users{}".format(job_num)))
    user_set = read_user_ids(user_id_fn)
    id_fn = os.path.basename(user_id_fn).split('.')[0]
    print("{} users to process".format(len(user_set)))
    outf_fn = os.path.join(dataset_dir, "{}_dataset.json.gz".format(id_fn))
    print("writing to {}".format(outf_fn))
    outf = gzip.open(outf_fn, 'w')
    count = 0
    for _id in user_set:
        count += 1
        if user_limit and count > user_limit:
            break
        timeline_fn = os.path.join(tweet_dir, "{}_statuses.json.gz".format(_id))
        if not os.path.exists(timeline_fn):
            continue

        count = 0
        texts = []
        with gzip.open(timeline_fn, 'r') as inf:
            for line in inf:
                data = json.loads(line.strip().decode('utf8'))
                text = data['full_text'] if 'full_text' in data else data['text']
                text = text.replace('\n', '. ')
                if ignore_rt:
                    if text.lower().startswith('rt') or data.get("retweeted_status") is not None:
                        continue

                count += 1
                if count > tweet_limit:
                    break
                texts.append(text)

            obj = {
                'texts': texts,
                'id_str': data['user']['id_str']
            }
            outf.write("{}\n".format(json.dumps(obj)).encode('utf8'))
    outf.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--user_id_fn', type=str, help='a txt file for user ids')
    parser.add_argument('--dataset_dir', type=str, help='Directory for built dataset')
    parser.add_argument('--tweet_dir', type=str, help='Directory where you stored scraped tweet timelines')
    ## optional arguments
    parser.add_argument('--ignore_rt', type=bool, default=True, help='Whether include retweets in the dataset')
    parser.add_argument('--tweet_limit', type=int, default=200, help='Max number of tweet to include')
    parser.add_argument('--user_limit', type=int, default=None, help='Max number of users to include, used for testing')


    args = parser.parse_args()
    build(user_id_fn=args.user_id_fn, dataset_dir=args.dataset_dir, tweet_dir=args.tweet_dir,
          ignore_rt=args.ignore_rt, tweet_limit=args.tweet_limit, user_limit=args.user_limit)