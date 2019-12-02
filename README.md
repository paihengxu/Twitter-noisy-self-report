# Twitter-noisy-self-report
Code for "Using Noisy Self-Reports to Predict Twitter User Demographics" 

<!--- sourcefile defined later. maybe present some sample users without violating term of services -->
To get noisy labeled dataset, run:
```
python scrpts/label_eval.py --task race --window 5 --threshold 0.35 --sourcefile ${src} --outdir ../noisy_labeled --simple
```
* ```task``` supports race and gender.
* ```window``` and ```threshold``` are the hyper-parameters we choose based on manual labeled development set in ```/src/manual_labeled```.
* ```--simple``` and ```--tfidf``` flags indicate simple co-occurrence weighting and tfidf weighting strategies respectively.

## Experiments in the paper
### Noisy self report
<!--- we have to at least provide users in dev set to reproduce the results-->
To test the effect of our weighted group filters (Table 2), run:
```
python scripts/filter_effect.py
```

To reproduce the performance on dev set, run:
```
python scripts/label_eval.py --task race --window 5 --threshold 0.35 --sourcefile ${src} --outdir ../noisy_labeled --simple --eval_dev
```
Modify the hyper-parameters, to get different results.

### Demographics prediction
Collect 200 tweets for each user.

### Group analysis
To get results for list-based features and quantitative linguistic features, run
```
python group_analysis.py --group latin --sourcedir /export/fs03/a10/pxu/groups-analysis/src/scraped/merged/ --outdir ../result/
```
* change the input for ```--group``` to collect the features for different group.
* the format for source timelines for each group is ```/source/directory/{group}/{userid}.json.gz```.
* intermediate results for behavioural analysis, pre-trained linguistic models, and statistical analysis are stored in the ```--outdir```.


