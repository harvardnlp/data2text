local train = {}

train.Checkpoint = require('onmt.train.Checkpoint')
train.EpochState = require('onmt.train.EpochState')
train.Optim = require('onmt.train.Optim')
train.Greedy = require('onmt.train.Greedy')

return train
