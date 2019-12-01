# Twitter-noisy-self-report
Code for "Using Noisy Self-Reports to Predict Twitter User Demographics" 

<!--- sourcefile defined later. maybe present some sample users without violating term of services -->
To get noisy labeled dataset, run:
```
python label_eval.py --task race --window 5 --threshold 0.35 --sourcefile ${src} --outdir ../noisy_labeled 
```
* ```task``` supports race and gender.
* ```window``` and ```threshold``` are the hyper-parameters we choose based on manual labeled development set in ```/src/manual_labeled```.

## Experiments in the paper
<!--- we have to at least provide users in dev set to reproduce the results-->
To run:
```
python label_eval.py --task race --window 5 --threshold 0.35 --sourcefile ${src} --outdir ../noisy_labeled --eval_dev
```
