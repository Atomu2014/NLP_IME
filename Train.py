import xgboost as xgb

train_path = 'raw/train.part1.vfeature2.part2'
test_path = 'raw/train.part1.vfeature2.part2'

dtrain = xgb.DMatrix(train_path)
dtest = xgb.DMatrix(test_path)

param = {'silent': 1, 'objective': 'binary:logistic', 'booster': 'gblinear', 'alpha': 0.0001, 'lambda': 1,
         'eval_metric': 'error'}

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 100
bst = xgb.train(param, dtrain, num_round, watchlist)
preds = bst.predict(dtest)
labels = dtest.get_label()

bst.dump_model(train_path + '.param')
bst.save_model(train_path + '.model')

# x = 0
# y = 0
# for i in range(len(preds)):
#     if labels[i]:
#         y += 1
#     if int(preds[i] > 0.5) and labels[i]:
#         x += 1
#
# print ('error=%f' % (x * 1.0 / y))
