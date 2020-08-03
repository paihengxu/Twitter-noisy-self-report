### Build dataset
First extract up to 200 tweets per user from scraped twitter timelines, run
 ```shell script
python build_dataset.py --user_id_fn $USER_ID --dataset_dir $DATASET_DIR --tweet_dir $TWEET_DIR
```

### Get bert embeddings 
Get BERT embeddings from 200 most recent tweets via average pooling for each user.

If you already got embeddings for all the users, skip this step. Get embeddings for each user. Recommended using GPU.
```shell script
python bert_clf_test.py --task embedding --user_id_fn $USER_ID --dataset_dir $DATASET_DIR --embed_dir $EMBED_DIR --cuda
```

## Prediction
```shell script
python bert_clf_test.py --task prediction --user_id_fn $USER_ID --out_dir $OUT_DIR  --embed_dir $EMBED_DIR
```

Please refer [run_pipeline.sh](run_pipeline.sh) for a working shell script.