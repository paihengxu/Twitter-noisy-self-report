from noisy_self_report import *

class filterEffect(noisySelfReport):
    def effectivenessFilters(self):
        filter_params = {
            "bigram": False,
            "color": False,
            "plural": False,
            "quote": False,
            "of": False,
            "possessive_pronouns": False}
        if self.task == 'race':
            apply_filters = ["", "quote", "plural", "color", "bigram"]
            # note: selected params
            win = 5
            thre = 0.35
            processed_user = {}
            with gzip.open(self.sourcefile, 'r') as inf:
                line = inf.readline()
                line_count = 0
                while line:
                    tweet = json.loads(line.decode('utf8'))
                    user = tweet['user']

                    line = inf.readline()
                    line_count += 1

                    if line_count % 100000 == 0:
                        print("{} lines scanned, found {} users in dev set".format(line_count, len(processed_user)))
                    if tweet['user']['id_str'] in self.dev_set.keys():
                        filter_applied = []
                        for fi in apply_filters:
                            if fi in filter_params.keys():
                                filter_params[fi] = True
                                filter_applied.append(fi)

                            filtered, fireQuery, destoken = self.__preProcessedTweet__(user)
                            userWithScore, userid = self.__labelExactUsersWithScore__(user, fireQuery, destoken)
                            processed_user[userid] = True

                            if len(userWithScore) == 0:
                                # print("no valid score")
                                continue
                            assert type(userid) == str

                            save_json_gz(userWithScore, os.path.join(self.outdir,
                                                                     'dev_user_score_window{}_power{}_'.format(win,
                                                                                                               1) + '_'.join(
                                                                         filter_applied) + '.json.gz'),
                                         mode='a')
                            if processed_user.keys() == self.dev_set.keys():
                                line = False
                                break
                        # reset
                        filter_params = {
                            "bigram": False,
                            "color": False,
                            "plural": False,
                            "quote": False,
                            "of": False,
                            "possessive_pronouns": False}

            print("{} lines scanned, found {} users in dev set".format(line_count, len(processed_user)))
            filters = []
            for fi in apply_filters:
                filters.append(fi)
                usersWithScore = self.__readUserWithScore__(os.path.join(self.outdir, 'dev_user_score_window{}_power{}_'
                                                                         .format(self.win, self.power)
                                                                         + '_'.join(filters) + '.json.gz'))
                usersLabeled = self.__evalSelfreportyScore__(usersWithScore)
                print("{} unique users".format(len(usersWithScore)))
                save_json_gz(usersLabeled, os.path.join(self.outdir,
                                                        'dev_user_label_window{}_power{}_thre{}_'.format(win, 1,
                                                                                                         thre) + '_'.join(
                                                            filters) + '.json.gz'))
                print('dev_user_label_window{}_power{}_thre{}_'.format(win, 1, thre) + '_'.join(filters) + '.json.gz')
                self.outputResult(self.outdir,
                                  'dev_user_label_window{}_power{}_thre{}_'.format(win, 1, thre) + '_'.join(
                                      filters) + '.json.gz',
                                  datasetname='dev_{}'.format(self.task), dataset=self.dev_set)
                if "" in filters:
                    filters.remove("")

if __name__ == '__main__':
    task = 'race'
    win = 5
    thre = 0.35
    tfidf = False
    outdir = '../noisy_labeled/'
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    filter_params = {
            "bigram": True,
            "color": True,
            "plural": True,
            "quote": True,
            "of": False,
            "possessive_pronouns": False}

    # source file must contain all users in dev set.
    sourcefile = '/path/to/sourcefile'
    nsr_fe = filterEffect(task, filter_params, win, thre, tfidf, sourcefile, outdir, eval=True)
    nsr_fe.effectivenessFilters()