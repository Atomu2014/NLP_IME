import xgboost as xgb

dtrain = xgb.DMatrix('raw/train.part1.feature.part1')
dtest = xgb.DMatrix('raw/train.part1.feature.part2')

param = {'silent': 1, 'objective': 'binary:logistic', 'booster': 'gblinear', 'alpha': 0.0001, 'lambda': 1}

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 100
bst = xgb.train(param, dtrain, num_round, watchlist)
preds = bst.predict(dtest)
labels = dtest.get_label()

bst.dump_model('raw/dump.txt')

x = 0
y = 0
for i in range(len(preds)):
    if labels[i]:
        y += 1
    if int(preds[i] > 0.5) and labels[i]:
        x += 1

print ('error=%f' % (x * 1.0 / y))
