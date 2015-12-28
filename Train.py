import xgboost as xgb

train_path = 'raw/train.part2.ellfeature1'
test_path = 'raw/train.part1.ellfeature1'

dtrain = xgb.DMatrix(train_path)
dtest = xgb.DMatrix(test_path)

param = {'silent': 1, 'objective': 'binary:logitraw', 'booster': 'gblinear', 'alpha': 0.0001, 'lambda': 1,
         # 'eval_metric': 'error'
         }

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 500
bst = xgb.train(param, dtrain, num_round, watchlist)
preds = bst.predict(dtest)
labels = dtest.get_label()

bst.dump_model(train_path + '.param')
bst.save_model(train_path + '.model')

import numpy as np

np.savetxt(test_path + '.pred', preds)
