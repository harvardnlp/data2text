require('onmt.init')

local path = require('pl.path')
tds = require('tds')
local cmd = torch.CmdLine()

cmd:text("")
cmd:text("**train.lua**")
cmd:text("")

cmd:option('-config', '', [[Read options from this file]])

cmd:text("")
cmd:text("**Data options**")
cmd:text("")

cmd:option('-data', '', [[Path to the training *-train.t7 file from preprocess.lua]])
cmd:option('-save_model', '', [[Model filename (the model will be saved as
                              <save_model>_epochN_PPL.t7 where PPL is the validation perplexity]])
cmd:option('-train_from', '', [[If training from a checkpoint then this is the path to the pretrained model.]])
cmd:option('-continue', false, [[If training from a checkpoint, whether to continue the training in the same configuration or not.]])
cmd:option('-just_eval', false, [[Evaluate ppl of a saved model]])
cmd:option('-just_gen', false, [[Generate from a saved model]])
cmd:option('-test', false, [[Use test (not validation) data]])
cmd:option('-beam_size', 5, [[Beam size used for generation]])
cmd:option('-gen_file', 'preds.txt', [[File to write generations to]])

cmd:text("")
cmd:text("**Model options**")
cmd:text("")

cmd:option('-layers', 2, [[Number of layers in the LSTM encoder/decoder]])
cmd:option('-rnn_size', 200, [[Size of LSTM hidden states]])
cmd:option('-word_vec_size', 200, [[Word embedding sizes]])
cmd:option('-feat_merge', 'concat', [[Merge action for the features embeddings: concat or sum]])
cmd:option('-input_feed', 1, [[Feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder.]])
cmd:option('-residual', false, [[Add residual connections between RNN layers.]])
cmd:option('-just_lm', false, [[No conditioning]])
cmd:option('-copy_generate', false, [[Use copy attn]])
cmd:option('-tanh_query', false, [[Apply tanh to attn query vector]])
cmd:option('-recdist', 0, [[Distance to use if doin continuous reconstruction]])
cmd:option('-discrec', false, [[Do discrete reconstruction]])
cmd:option('-discdist', 0, [[1 for total dev; 2 for hellinger]])
cmd:option('-recembsize', 300, [[Embedding size of entries to reconstruct]])
cmd:option('-partition_feats', false, [[Partition feats used in discrete reconstruction]])
cmd:option('-nfilters', 200, [[Convolutional filters for reconstruction]])
cmd:option('-nrecpreds', 3, [[Number of entries to reconstruct]])
cmd:option('-rho', 0.5, [[Reconstruction loss coefficient]])

cmd:option('-switch', false, [[Use switching/conditional copying]])
cmd:option('-multilabel', false, [[Marginalize over possibly correct pointer locations]])
cmd:option('-map', false, [[Select MAP word (under switching model)]])

cmd:option('-pool', 'mean', [[Table embedding pooling: mean or max]])
cmd:option('-enc_layers', 1, [[Number of encoder layers]])
cmd:option('-enc_emb_size', 200, [[Table encoder embedding size]])
cmd:option('-enc_dropout', 0, [[Rate]])
cmd:option('-enc_relu', false, [[Use ReLU nonlinearity in encoder]])

cmd:text("")
cmd:text("**Optimization options**")
cmd:text("")

cmd:option('-max_batch_size', 64, [[Maximum batch size]])
cmd:option('-epochs', 13, [[Number of training epochs]])
cmd:option('-start_epoch', 1, [[If loading from a checkpoint, the epoch from which to start]])
cmd:option('-start_iteration', 1, [[If loading from a checkpoint, the iteration from which to start]])
cmd:option('-param_init', 0.1, [[Parameters are initialized over uniform distribution with support (-param_init, param_init)]])
cmd:option('-optim', 'sgd', [[Optimization method. Possible options are: sgd, adagrad, adadelta, adam, mom]])
cmd:option('-learning_rate', 1, [[Starting learning rate. If adagrad/adadelta/adam is used,
                                then this is the global learning rate. Recommended settings are: sgd = 1,
                                adagrad = 0.1, adadelta = 1, adam = 0.0002]])
cmd:option('-mom', 0.9, [[momentum]])
cmd:option('-max_grad_norm', 5, [[If the norm of the gradient vector exceeds this renormalize it to have the norm equal to max_grad_norm]])
cmd:option('-dropout', 0.3, [[Dropout probability. Dropout is applied between vertical LSTM stacks.]])
cmd:option('-learning_rate_decay', 0.5, [[Decay learning rate by this much if (i) perplexity does not decrease
                                        on the validation set or (ii) epoch has gone past the start_decay_at_limit]])
cmd:option('-start_decay_at', 10000, [[Start decay after this epoch]])
cmd:option('-decay_update2', false, [[Decay only when validation doesn't improve]])
cmd:option('-curriculum', 0, [[For this many epochs, order the minibatches based on source
                             sequence length. Sometimes setting this to 1 will increase convergence speed.]])
cmd:option('-pre_word_vecs_enc', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the encoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-pre_word_vecs_dec', '', [[If a valid path is specified, then this will load
                                     pretrained word embeddings on the decoder side.
                                     See README for specific formatting instructions.]])
cmd:option('-fix_word_vecs_enc', false, [[Fix word embeddings on the encoder side]])
cmd:option('-fix_word_vecs_dec', false, [[Fix word embeddings on the decoder side]])
cmd:option('-max_bptt', 500, [[Maximum BPTT window size]])

cmd:text("")
cmd:text("**Other options**")
cmd:text("")

-- GPU
cmd:option('-gpuid', 0, [[1-based identifier of the GPU to use. CPU is used when the option is < 1]])
cmd:option('-nparallel', 1, [[When using GPUs, how many batches to execute in parallel.
                            Note: this will technically change the final batch size to max_batch_size*nparallel.]])
cmd:option('-disable_mem_optimization', false, [[Disable sharing internal of internal buffers between clones - which is in general safe,
                                                except if you want to look inside clones for visualization purpose for instance.]])

-- bookkeeping
cmd:option('-save_every', 0, [[Save intermediate models every this many iterations within an epoch.
                             If = 0, will not save models within an epoch. ]])
cmd:option('-report_every', 50, [[Print stats every this many iterations within an epoch.]])
cmd:option('-seed', 3435, [[Seed for random initialization]])
cmd:option('-json_log', false, [[Outputs logs in JSON format.]])

local opt = cmd:parse(arg)

if not opt.just_gen then
    print(opt)
end

local function reseed()
  torch.manualSeed(opt.seed)
  if opt.gpuid > 0 then
    cutorch.manualSeed(opt.seed)
  end
end

local function initParams(model, verbose)
    local numParams = 0
    local params = {}
    local gradParams = {}

    if verbose then
        print('Initializing parameters...')
    end

    -- we assume all the sharing has already been done,
    -- so we just make a big container to flatten everything
    local everything = nn.Sequential()
    for k, mod in pairs(model) do
        everything:add(mod)
    end

    local p, gp = everything:getParameters()

    if opt.train_from:len() == 0 then
        p:uniform(-opt.param_init, opt.param_init)
        -- do module specific init; wordembeddings will happen multiple times,
        -- but who cares
        for k, mod in pairs(model) do
            mod:apply(function (m)
                if m.postParametersInitialization then
                    m:postParametersInitialization()
                end
            end)
        end
    else
        print("copying loaded params...")
        local checkpoint = torch.load(opt.train_from)
        p:copy(checkpoint.flatParams[1])
    end

    numParams = numParams + p:size(1)
    table.insert(params, p)
    table.insert(gradParams, gp)

    if verbose then
        print(" * number of parameters: " .. numParams)
    end
    return params, gradParams
end

local function buildCriterion(vocabSize, features)
  local criterion = nn.ParallelCriterion(false)

  local function addNllCriterion(size)
    -- Ignores padding value.
    local w = torch.ones(size)
    w[onmt.Constants.PAD] = 0

    local nll = nn.ClassNLLCriterion(w)

    -- Let the training code manage loss normalization.
    nll.sizeAverage = false
    criterion:add(nll)
  end

  addNllCriterion(vocabSize)

  for j = 1, #features do
    addNllCriterion(features[j]:size())
  end

  return criterion
end

function allTraining(model)
    for _, mod in pairs(model) do
        if mod.training then
            mod:training()
        end
    end
end

function allEvaluate(model)
    for _, mod in pairs(model) do
        if mod.evaluate then
            mod:evaluate()
        end
    end
end

-- gets encodings for all rows
function allEncForward(model, batch)
    local aggEncStates, catCtx = model.encoder:forward(batch)
    if opt.just_lm then
        for i = 1, #aggEncStates do
            aggEncStates[i]:zero()
        end
        catCtx:zero()
    end
    return aggEncStates, catCtx
end

-- goes backward over all encoders
function allEncBackward(model, batch, encGradStatesOut, gradContext)
    model.encoder:backward(batch, encGradStatesOut, gradContext)
end

local function eval(model, criterion, data)
  local loss = 0
  local total = 0

  -- model.encoder:evaluate()
  -- model.decoder:evaluate()
  allEvaluate(model)
  for i = 1, data:batchCount() do
    model.decoder:resetLastStates()
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local aggEncStates, catCtx = allEncForward(model, batch)
    --loss = loss + model.decoder:computeLoss(batch, encoderStates, context, criterion)
    loss = loss + model.decoder:computeLoss(batch, aggEncStates, catCtx, criterion)
    total = total + batch.targetNonZeros
  end

  -- model.encoder:training()
  -- model.decoder:training()
  allTraining(model)

  return math.exp(loss / total)
end

local function convert_and_shorten_string(ts, max_len, dict)
   local strtbl = {}
   for i = 1, max_len do
       if ts[i] == onmt.Constants.EOS then
           break
       end
       table.insert(strtbl, dict.idxToLabel[ts[i]])
   end
   return stringx.join(' ', strtbl)
end

local function beamGen(model, data, tgtDict)
  -- adapted from Translator:translateBatch()
  local max_sent_length = 1500
  print("using max len:", 1500)
  allEvaluate(model)
  local outFile = io.open(opt.gen_file, 'w')
  for i = 1, data:batchCount() do
    model.decoder:resetLastStates()
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local aggEncStates, catCtx = allEncForward(model, batch)
    local advancer
    if opt.switch then
        advancer = onmt.translate.SwitchingDecoderAdvancer.new(model.decoder,
           batch, catCtx, max_sent_length, nil, aggEncStates, nil, opt.map, opt.multilabel)
    else
        advancer = onmt.translate.Decoder2Advancer.new(model.decoder,
           batch, catCtx, max_sent_length, nil, aggEncStates, nil)
    end
    local beamSearcher = onmt.translate.BeamSearcher.new(advancer)
    local results = beamSearcher:search(opt.beam_size, 1, 1, false)
    for b = 1, batch.size do
        local top1 = results[b][1].tokens
        local top1tostr = convert_and_shorten_string(top1, #top1, tgtDict)
        print(top1tostr)
        outFile:write(top1tostr, '\n')
    end
  end
  outFile:close()
end


local function trainModel(model, trainData, validData, dataset, info)
    local criterion
    local verbose = true
    local params, gradParams = initParams(model, verbose)
    allTraining(model)
    -- for _, mod in pairs(model) do
    --     mod:training()
    -- end

    -- define criterion of each GPU
    criterion = onmt.utils.Cuda.convert(buildCriterion(dataset.dicts.tgt.words:size(),
                                                          dataset.dicts.tgt.features))
    local recCrit
    if opt.discrec then
        recCrit = onmt.utils.Cuda.convert(nn.KMinXent())
        recCrit.sizeAverage = false
    elseif opt.recdist > 0 then
        recCrit = onmt.utils.Cuda.convert(nn.KMinDist(opt.recdist))
        recCrit.sizeAverage = false
    end

    local switchCrit, ptrCrit
    if opt.switch then
        switchCrit = onmt.utils.Cuda.convert(nn.BCECriterion())
        switchCrit.sizeAverage = false
        if opt.multilabel then
            ptrCrit = onmt.utils.Cuda.convert(nn.MarginalNLLCriterion())
            ptrCrit.sizeAverage = false
        else
            ptrCrit = onmt.utils.Cuda.convert(nn.ClassNLLCriterion())
            ptrCrit.sizeAverage = false
        end
    end

    -- optimize memory of the first clone
    if not opt.disable_mem_optimization then
        local batch = onmt.utils.Cuda.convert(trainData:getBatch(1))
        batch.totalSize = batch.size
        onmt.utils.Memory.boxOptimize2(model, criterion, batch, verbose, switchCrit, ptrCrit)
    end

    local optim = onmt.train.Optim.new({
        method = opt.optim,
        numModels = 1, -- we flattened everything
        learningRate = opt.learning_rate,
        learningRateDecay = opt.learning_rate_decay,
        startDecayAt = opt.start_decay_at,
        optimStates = opt.optim_states,
        mom = opt.mom
    })

    local checkpoint = onmt.train.Checkpoint.new(opt, model, params, optim, dataset)

    local function trainEpoch(epoch, lastValidPpl)
        local epochState
        local batchOrder
        local startI = opt.start_iteration

        local numIterations = trainData:batchCount()

        if startI > 1 and info ~= nil then
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl, info.epochStatus)
            batchOrder = info.batchOrder
        else
            epochState = onmt.train.EpochState.new(epoch, numIterations, optim:getLearningRate(), lastValidPpl)
            -- Shuffle mini batch order.
            batchOrder = torch.randperm(trainData:batchCount())
        end

        --opt.start_iteration = 1

        local iter = 1
        local totalLoss2, totalLoss3 = 0, 0
        model.decoder:remember()
        for i = startI, trainData:batchCount() do
            local batchIdx = epoch <= opt.curriculum and i or batchOrder[i]
            local batch =  trainData:getBatch(batchIdx)
            batch.totalSize = batch.size -- fuck off
            onmt.utils.Cuda.convert(batch)

            local batchPieces = batch:splitIntoPieces(opt.max_bptt)
            model.decoder:resetLastStates() -- don't use saved last state for new batch
            for j = 1, batchPieces do
                optim:zeroGrad(gradParams)
                local aggEncStates, catCtx = allEncForward(model, batch)
                local ctxLen = catCtx:size(2)

                local decOutputs = model.decoder:forward(batch, aggEncStates, catCtx)
                local encGradStatesOut, gradContext, loss, loss2, loss3 = model.decoder:backward(batch, decOutputs,
                                                                           criterion, ctxLen, recCrit,
                                                                            switchCrit, ptrCrit)
                allEncBackward(model, batch, encGradStatesOut, gradContext)

                -- Update the parameters.
                optim:prepareGrad(gradParams, opt.max_grad_norm)
                optim:updateParams(params, gradParams)
                --epochState:update(batch, loss, recloss)
                epochState:update(batch, loss, nil)
                if loss2 then
                    totalLoss2 = totalLoss2 + loss2
                end
                if loss3 then
                    totalLoss3 = totalLoss3 + loss3
                end
                batch:nextPiece()
            end

            if iter % opt.report_every == 0 then
                epochState:log(iter, opt.json_log)
                if opt.switch then
                    print("switchLoss", totalLoss2/epochState.status.trainNonzeros)
                    print("ptrLoss", totalLoss3/epochState.status.trainNonzeros)
                end
                collectgarbage()
            end
            if opt.save_every > 0 and iter % opt.save_every == 0 then
                checkpoint:saveIteration(iter, epochState, batchOrder, not opt.json_log)
            end
            iter = iter + 1
        end
        return epochState
    end -- end local function trainEpoch

    reseed()
    local validPpl = 0
    local bestPpl = math.huge
    local bestEpoch = -1

    if opt.just_gen then
        beamGen(model, validData, g_tgtDict)
        return
    elseif opt.just_eval then
        validPpl = eval(model, criterion, validData)
            print('Validation perplexity: ' .. validPpl)
        return
    end

    if not opt.json_log then
        print('Start training...')
    end

    for epoch = opt.start_epoch, opt.epochs do
        if not opt.json_log then
            print('')
        end

        local epochState = trainEpoch(epoch, validPpl)
        validPpl = eval(model, criterion, validData)
        if not opt.json_log then
            print('Validation perplexity: ' .. validPpl)
        end

        if opt.optim == 'sgd' or opt.optim == 'mom' then
            if opt.decay_update2 then
                optim:updateLearningRate2(validPpl, epoch)
            else
                optim:updateLearningRate(validPpl, epoch)
            end
        end

        if validPpl < bestPpl then
            checkpoint:deleteEpoch(bestPpl, bestEpoch)
            checkpoint:saveEpoch(validPpl, epochState, not opt.json_log)
            bestPpl = validPpl
            bestEpoch = epoch
        end
        collectgarbage()
        collectgarbage()
    end

end -- end local function trainModel

local function main()
  local requiredOptions = {
    "data",
    "save_model"
  }

  onmt.utils.Opt.init(opt, requiredOptions)
  onmt.utils.Cuda.init(opt)
  onmt.utils.Parallel.init(opt)

  reseed()

  -- Create the data loader class.
  if not opt.json_log then
      if not opt.just_gen then
          print('Loading data from \'' .. opt.data .. '\'...')
      end
  end

  local dataset = torch.load(opt.data, 'binary', false)

  assert(dataset.dicts.src.words:size() == dataset.dicts.tgt.words:size())
  -- add extra for all the column features
  --Hacky Constants
  g_nRegRows = #dataset.train.src.words/2 - 1 -- two teams and nRegRows players
  assert(g_nRegRows == 13)
  g_nCols = dataset.train.src.words[1][1]:size(1) - 1 -- leave off first b/c it's the row name
  assert(g_nCols == 22)
  g_specPadding = g_nCols -- assume last real column is the row name for special (i.e., team) rows
  g_nFeatures = 4

  if not opt.just_gen then
      print("USING HACKY GLOBALS!!!",
        string.format("regRows: %d; specPadding: %d; nCols: %d; nFeats: %d",
        g_nRegRows, g_specPadding, g_nCols, g_nFeatures))
      print("")
  end

  local colStartIdx = dataset.dicts.src.words:size()+1 -- N.B. order is crucial
  for i = 1, g_nCols*2+2 do -- column names for reg and spec, plus home/away features
      dataset.dicts.src.words:add("DOPEEXTRALABEL" .. i)
      dataset.dicts.tgt.words:add("DOPEEXTRALABEL" .. i)
  end
  assert(dataset.dicts.src.words:size() == dataset.dicts.tgt.words:size())

  g_tgtDict = dataset.dicts.tgt.words

  local tripV   -- vocabulary for each element in a triple (for rec)
  if opt.discrec then
      tripV = {dataset.dicts.src.rows:size(), dataset.dicts.src.cols:size(), dataset.dicts.src.cells:size()}
      print("tripV:", tripV)
  end

  local trainData = onmt.data.BoxDataset2.new(dataset.train.src, dataset.train.tgt,
    colStartIdx, g_nFeatures, opt.copy_generate, nil, tripV, opt.switch, opt.multilabel)
  local validData
  if opt.test then
    print("using test data...")
    validData = onmt.data.BoxDataset2.new(dataset.test.src, dataset.test.tgt,
      colStartIdx, g_nFeatures, opt.copy_generate, nil, tripV)
  else
    validData = onmt.data.BoxDataset2.new(dataset.valid.src, dataset.valid.tgt,
      colStartIdx, g_nFeatures, opt.copy_generate, nil, tripV) -- no switching on valid
  end

  trainData:setBatchSize(opt.max_batch_size)
  validData:setBatchSize(opt.max_batch_size)

  if not opt.json_log then
      if not opt.just_gen then
        print(string.format(' * vocabulary size: source = %d; target = %d',
                            dataset.dicts.src.words:size(), dataset.dicts.tgt.words:size()))
        print(string.format(' * additional features: source = %d; target = %d',
                            #dataset.dicts.src.features, #dataset.dicts.tgt.features))
        print(string.format(' * maximum sequence length: source = %d; target = %d',
                            trainData.maxSourceLength, trainData.maxTargetLength))
        print("nSourceRows", trainData.nSourceRows)
        print(string.format(' * number of training instances: %d', #trainData.tgt))
        print(string.format(' * maximum batch size: %d', opt.max_batch_size))
      end
  else
    local metadata = {
      options = opt,
      vocabSize = {
        source = dataset.dicts.src.words:size(),
        target = dataset.dicts.tgt.words:size()
      },
      additionalFeatures = {
        source = #dataset.dicts.src.features,
        target = #dataset.dicts.tgt.features
      },
      sequenceLength = {
        source = trainData.maxSourceLength,
        target = trainData.maxTargetLength
      },
      trainingSentences = #trainData.tgt
    }

    onmt.utils.Log.logJson(metadata)
  end

  if not opt.json_log then
      if not opt.just_gen then
          print('Building model...')
      end
  end

    local model = {}

    local verbose = true
    -- make decoder first
    model.decoder = onmt.Models.buildDecoder(opt, dataset.dicts.tgt, verbose, tripV)
    -- send to gpu immediately to make cloning things simpler
    onmt.utils.Cuda.convert(model.decoder)
    model.encoder = onmt.BoxTableEncoder({
        vocabSize = dataset.dicts.src.words:size(),
        encDim = opt.enc_emb_size,
        decDim = opt.rnn_size,
        feat_merge = opt.feat_merge,
        nFeatures = g_nFeatures,
        nLayers = opt.enc_layers,
        nRows = trainData.nSourceRows,
        nCols = g_nCols,
        pool = opt.pool or "mean",
        effectiveDecLayers = opt.layers*2,
        dropout = opt.enc_dropout,
        relu = opt.enc_relu,
        wordVecSize = opt.word_vec_size,
        input_feed = opt.input_feed
    })
    onmt.utils.Cuda.convert(model.encoder)
    -- share all the things
    assert(model.encoder.lut.weight:size(1) == model.decoder.inputNet.net.weight:size(1))
    model.encoder.lut:share(model.decoder.inputNet.net, 'weight', 'gradWeight')
    model.encoder:shareTranforms()

    trainModel(model, trainData, validData, dataset, nil)
end

main()
