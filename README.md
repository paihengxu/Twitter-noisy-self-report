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

## Dataset
We provide our collected datasets in ```src/```. 
Note due to ethical issue, we only provide user ids and corresponding demographics labels.
* ```src/dataset_race_simple_win5_thre0.35.json.gz``` and ```src/dataset_race_tfidf_win11_thre2.5.json.gz``` are the large dataset
collected using noisy self report from users' descriptions, with simple co-occurrence and tfidf weighting strategy respectively.
* files in ```src/manual_labeled``` are manually labeled users. 
```rule_out_bigrams.json.gz``` contains bigrams with race-related query words that cause false positive.
* files in ```src/self_reporty_words/``` are candidate self-report words collected from 177M user descriptions.

  * To collect the candidate words, please refer to scripts in ```/scripts/iama/```.
  * We also refine the collections by only keeping the words that are majorly tagged as noun or adjective by [GoogleNgrams](https://books.google.com/ngrams/info).
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
* before running the script, the timeline for each users should be collected first. 
The format for source timelines should be stored by group as ```/source/directory/{group}/{userid}.json.gz```.
* intermediate results for behavioural analysis, pre-trained linguistic models, and statistical analysis are stored in the ```--outdir```.

To get results for previously-trained model, i.e., [formality](https://github.com/YahooArchive/formality-classifier), 
[politeness](https://github.com/sudhof/politeness) 
and [lexical variation](https://github.com/jacobeisenstein/SAGE/tree/master/py-sage),
run the scripts in provided links on intermediate results collected above which contains all the text from each user in ```outdir/tmp/```.

To get statistical difference for various measures, run ```scrpts/correlation.py```.
* pass the argument ```list``` to the script to get the Kendall's Tau correlation for top items in the lists.
* pass the argument ```numerical```, ```intangible```, ```behavior``` to the script to get the Mann Whitney U test results for corresponding features
after obtaining the group analysis results.
