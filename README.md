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

