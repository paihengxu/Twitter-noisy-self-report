#!/usr/bin/env bash
WORK_DIR=/export/c10/pxu/data/social_distancing_user/bert_tweets
USER_ID=$WORK_DIR/sample.id.txt
DATASET_DIR=$WORK_DIR
TWEET_DIR=/export/c10/pxu/data/location_0330/scraped
EMBED_DIR=$WORK_DIR/distil_bert_embed
OUT_DIR=$WORK_DIR

source /export/c10/pxu/venv3.7/bin/activate

python build_dataset.py --user_id_fn $USER_ID --dataset_dir $DATASET_DIR --tweet_dir $TWEET_DIR

python bert_clf.py --task embedding --user_id_fn $USER_ID --dataset_dir $DATASET_DIR --embed_dir $EMBED_DIR

python bert_clf.py --task prediction --user_id_fn $USER_ID --out_dir $OUT_DIR  --embed_dir $EMBED_DIR

deactivate