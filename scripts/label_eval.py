from optparse import OptionParser
import os
from noisy_self_report import *

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--task', dest='Task',
                      help='gender or race', metavar='TASK')
    # parser.add_option('--mode', dest='Mode',
    #                   help='score for calculating the score with different window size, '
    #                        'label for generate label for different threshold', metavar='MODE')
    parser.add_option('--window', type='int', dest='Win', default=5, help='hyperparameter for window size', metavar='WIN')
    parser.add_option('--threshold', type='float', dest='Thre', default=0.35, help='hyperparameter for self reporty score threshold',
                      metavar='THRE')
    parser.add_option('--sourcefile', type='str', help='source file for labeling group, from 1% API feed',
                      dest='SourceFile', metavar='SOURCEFILE')
    parser.add_option('--outdir', type='str', help='output directory',
                      dest='OutDir', metavar='OUTDIR')
    parser.add_option('--eval_dev', help='whether to evaluate on manual labeled dev set',
                      action='store_true', dest='Eval', metavar='EVAL')
    parser.add_option('--tfidf', help='tfidf weighting', action='store_true', dest='Tfidf', metavar='TFIDF')
    parser.add_option('--simple', help='simple weighting', action='store_false', dest='Tfidf', metavar='TFIDF')
    (options, args) = parser.parse_args()

    task = options.Task
    sourcefile = options.SourceFile
    outdir = options.OutDir
    tfidf = options.Tfidf
    win = options.Win
    thre = options.Thre
    eval_dev = options.Eval


    if task == 'race':
        filter_params = {
            "bigram": True,
            "color": True,
            "plural": True,
            "quote": True,
            "of": False,
            "possessive_pronouns": False}
    elif task == 'gender':
        filter_params = {
            "bigram": False,
            "color": False,
            "plural": True,
            "quote": True,
            "of": True,
            "possessive_pronouns": True}
    else:
        raise ValueError("unknown task {}".format(task))

    # os.makedir
    if tfidf:
        outdir += '/tfidf/'
    else:
        outdir += '/simple/'

    try:
        os.makedirs(outdir)
    except OSError:
        pass

    print("Writing to {}".format(outdir))

    all_filter_keys = sorted(list(filter_params.keys()))
    filter_applied = [key for key in all_filter_keys if filter_params[key]]
    saved_path = os.path.join(outdir, 'user_score_window{}_power{}_'.format(win, 1) +
                              '_'.join(filter_applied) + '.json.gz')

    nsr = noisySelfReport(task, filter_params, win, thre, tfidf, sourcefile, outdir, eval=eval_dev)
    if not os.path.exists(saved_path):
        nsr.calculateSelfReportyScore()
    else:
        print('Self-reporty score in this hyper-parameter setting already exists in {}, running labelUsers only. '
              'If you want to recalculate the scores, please rename the existed score file first'.format(
            saved_path))
    nsr.labelUsers()

