source /export/fs03/a10/pxu/venv3.7_new/bin/activate
src=/export/c10/zach/data/demographics/descriptions/exact_group.2018.json.gz

python label_eval.py --task race --window 5 --threshold 0.35 --sourcefile ${src} --outdir ../noisy_labeled --simple --eval_dev
