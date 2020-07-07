import gzip
import json
import os
import sys

dataset_dir = '/export/c10/pxu/data/social_distancing_user/bert_tweets/'
tweet_dir = '/export/c10/pxu/data/location_0330/scraped'

def read_user_ids(fn):
    user_set = set()
    # if fn.endswith('.txt') :
    with open(fn, 'r') as inf:
        for line in inf:
            user_set.add(line.strip())
    # elif fn.endswith('.json.gz'):
    #     with gzip.open(fn, 'r') as inf:
    #         for line in inf:
    #             data = json.loads(line.strip().decode('utf8'))
    #             user_set.add(data['user']['id_str'])
    return user_set


def build(job_num, ignore_rt=True, tweet_limit=200):
    user_set = read_user_ids(fn=os.path.join(dataset_dir, "users{}".format(job_num)))

    print("{} users to process".format(len(user_set)))
    outf = gzip.open(os.path.join(dataset_dir, "{}.json.gz".format(job_num)), 'w')
    # count = 0
    for _id in user_set:
        # count += 1
        # if count > 1000:
        #     break
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
    job_num = sys.argv[-1]
    build(job_num)