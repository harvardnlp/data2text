require 'nn'
require 'hdf5'
require 'cutorch'
require 'cunn'
require 'cudnn'
--require 'nngraph'
require 'onmt.modules.MarginalNLLCriterion'

local stringx = require('pl.stringx')

local cmd = torch.CmdLine()
cmd:option('-datafile', 'roto-ie.h5', [[path to hdf5 file containing train/val data]])
cmd:option('-batchsize', 32, [[batch size]])
cmd:option('-embed_size', 200, [[size of embeddings]])
cmd:option('-num_filters', 200, [[number of convolutional filters]])
cmd:option('-conv_fc_layer_size', 500, [[size of fully connected layer in convolutional model]])
cmd:option('-blstm_fc_layer_size', 700, [[size of fully connected layer in BLSTM model]])
cmd:option('-dropout', 0.5, [[dropout rate]])
cmd:option('-uniform_init', 0.1, [[init in params in this range]])
cmd:option('-lr', 0.7, [[learning rate]])
cmd:option('-lr_decay', 0.5, [[decay factor]])
cmd:option('-clip', 5, [[clip grads so they do not exceed this]])
cmd:option('-seed', 3435, [[Random seed]])
cmd:option('-epochs', 10, [[training epochs]])
cmd:option('-gpuid', 1, [[gpu idx]])
cmd:option('-savefile', '', [[path to save model to]])
cmd:option('-preddata', '', [[path to hdf5 file containing candidate relations from generated data]])
cmd:option('-dict_pfx', '', [[prefix of .dict and .labels files]])
cmd:option('-ignore_idx', 11, [[idx of NONE class in *.labels file]])
cmd:option('-just_eval', false, [[just eval generations]])
cmd:option('-lstm', false, [[use a BLSTM rather than a convolutional model]])
cmd:option('-geom', false, [[average models geometrically]])
cmd:option('-test', false, [[use test data]])

local opt = cmd:parse(arg)

function prep_data(batchsize)
    local f = hdf5.open(opt.datafile)
    local trlabels = f:read("trlabels"):all()
    local perm = torch.randperm(trlabels:size(1)):long()
    trlabels = trlabels:index(1, perm)
    local trsents = f:read("trsents"):all():index(1, perm)
    local trlens = f:read("trlens"):all():index(1, perm)
    local trentdists = f:read("trentdists"):all():index(1, perm)
    local trnumdists = f:read("trnumdists"):all():index(1, perm)

    local valsents, vallens, valentdists, valnumdists, vallabels, vallabelnums
    if opt.test then
      valsents = f:read("testsents"):all()
      vallens = f:read("testlens"):all()
      valentdists = f:read("testentdists"):all()
      valnumdists = f:read("testnumdists"):all()
      vallabels = f:read("testlabels"):all() -- these are 2d
      vallabelnums = vallabels:select(2, vallabels:size(2))
      vallabels = vallabels:narrow(2, 1, vallabels:size(2)-1):contiguous()
    else
      valsents = f:read("valsents"):all()
      vallens = f:read("vallens"):all()
      valentdists = f:read("valentdists"):all()
      valnumdists = f:read("valnumdists"):all()
      vallabels = f:read("vallabels"):all() -- these are 2d
      vallabelnums = vallabels:select(2, vallabels:size(2))
      vallabels = vallabels:narrow(2, 1, vallabels:size(2)-1):contiguous()
    end
    f:close()

    local psents, plens, pentdists, pnumdists, plabels, pboxrestartidxs
    if opt.just_eval and opt.preddata:len() > 0 then
        local f = hdf5.open(opt.preddata)
        psents = f:read("valsents"):all()
        plens = f:read("vallens"):all()
        pentdists = f:read("valentdists"):all()
        pnumdists = f:read("valnumdists"):all()
        plabels = f:read("vallabels"):all() -- these are 2d
        pboxrestartidxs = f:read("boxrestartidxs"):all() -- already 1-indexed
        plabelnums = plabels:select(2, plabels:size(2))
        plabels = plabels:narrow(2, 1, plabels:size(2)-1):contiguous()
        f:close()
    end

    -- need to shift negative distances...
    min_entdist = math.min(trentdists:min(), valentdists:min())
    if pentdists then pentdists:clamp(min_entdist, trentdists:max()) end
    min_numdist = math.min(trnumdists:min(), valnumdists:min())
    if pentdists then pnumdists:clamp(min_numdist, trnumdists:max()) end
    trentdists:add(-min_entdist+1)
    valentdists:add(-min_entdist+1)
    if pentdists then
        pentdists:add(-min_entdist+1)
    end
    trnumdists:add(-min_numdist+1)
    valnumdists:add(-min_numdist+1)
    if pnumdists then
        pnumdists:add(-min_numdist+1)
    end

    local nlabels = trlabels:max()

    word_pad = trsents:max()+1
    ent_dist_pad = trentdists:max()+1
    num_dist_pad = trnumdists:max()+1


    local function make_batches(sents, lens, entdists, numdists, labels, labelnums)
        local batches = {}
        for i = 1, sents:size(1), batchsize do
            local ub = math.min(i+batchsize-1, sents:size(1))
            local max_len = lens:sub(i, ub):max()
            for j = i, ub do
                if lens[j] < max_len then
                    sents[j]:sub(lens[j]+1, max_len):fill(word_pad)
                    entdists[j]:sub(lens[j]+1, max_len):fill(ent_dist_pad)
                    numdists[j]:sub(lens[j]+1, max_len):fill(num_dist_pad)
                end
            end
            table.insert(batches, {
                sent = sents:sub(i, ub, 1, max_len),
                ent_dists = entdists:sub(i, ub, 1, max_len),
                num_dists = numdists:sub(i, ub, 1, max_len),
                labels = labels:sub(i, ub),
                labelnums = labelnums and labelnums:sub(i, ub)
            })
        end
        return  batches
    end

    local tr_batches = make_batches(trsents, trlens, trentdists, trnumdists, trlabels)
    --print("num training batches:", #tr_batches)
    local val_batches = make_batches(valsents, vallens, valentdists, valnumdists, vallabels, vallabelnums)
    --print("num val batches:", #val_batches)
    local pred_batches
    if psents then
        pred_batches = make_batches(psents, plens, pentdists, pnumdists, plabels, plabelnums)
    end

    collectgarbage()
    return tr_batches, val_batches, {word_pad, ent_dist_pad, num_dist_pad}, nlabels, pred_batches, pboxrestartidxs
end

function get_dict(finame, invert)
    local dict = {}
    local dict_size = 0
    local fi = assert(io.open(finame, "r"))
    while true do
        local line = fi:read()
        if line == nil then
            break
        end
        local pieces = stringx.split(line)
        if invert then
            dict[tonumber(pieces[2])] = pieces[1]
        else
            dict[pieces[1]] = tonumber(pieces[2])
        end
        dict_size = dict_size + 1
    end
    return dict, dict_size
end

function make_conv_model(vocab_sizes, emb_sizes, nlabels, opt)
    local par = nn.ParallelTable()
    local first_layer_size = 0
    local kWs = {2, 3, 5} -- kernel widths

    for j = 1, #vocab_sizes do
        if emb_sizes then
            par:add(nn.LookupTable(vocab_sizes[j], emb_sizes[j]))
            first_layer_size = first_layer_size + emb_sizes[j]
        else
            par:add(nn.LookupTable(vocab_sizes[j], opt.embed_size))
        end
    end

    if not emb_sizes then
        first_layer_size = opt.embed_size
    end

    local mod = nn.Sequential():add(par)
    mod:add(nn.JoinTable(3))

    -- simple 1 layer conv
    local cat = nn.ConcatTable()
    for j = 1, #kWs do
        cat:add(nn.Sequential()
                  :add(cudnn.TemporalConvolution(first_layer_size, opt.num_filters, kWs[j], 1, kWs[j]-1))
                  :add(nn.ReLU())
                  :add(nn.Max(2)))
    end
    mod:add(cat)
    mod:add(nn.JoinTable(2))

    if opt.dropout > 0 then
       mod:add(nn.Dropout(opt.dropout))
    end

    mod:add(nn.Linear(#kWs*opt.num_filters, opt.conv_fc_layer_size))
    mod:add(nn.ReLU())

    if opt.dropout > 0 then
        mod:add(nn.Dropout(opt.dropout))
    end

    mod:add(nn.Linear(opt.conv_fc_layer_size, nlabels))
    mod:add(nn.SoftMax())
    return mod
end


function make_blstm_model(vocab_sizes, emb_sizes, nlabels, opt)
    local par = nn.ParallelTable()
    local first_layer_size = 0

    for j = 1, #vocab_sizes do
        par:add(nn.LookupTable(vocab_sizes[j], emb_sizes[j]))
        first_layer_size = first_layer_size + emb_sizes[j]
    end

    local mod = nn.Sequential():add(par)
    mod:add(nn.JoinTable(3)) -- bsz x seqlen x dim

    mod:add(nn.Transpose({1,2}))
    mod:add(cudnn.BLSTM(first_layer_size, first_layer_size, 1)) -- seqlen x bsz x 2dim
    mod:add(nn.Max(1))

    mod:add(nn.Linear(2*first_layer_size, opt.blstm_fc_layer_size))
    mod:add(nn.ReLU())

    if opt.dropout > 0 then
        mod:add(nn.Dropout(opt.dropout))
    end

    mod:add(nn.Linear(opt.blstm_fc_layer_size, nlabels))
    mod:add(nn.SoftMax())

    return mod
end

function get_acc(model, valbatches)
    if not g_maxes then
        g_maxes = torch.CudaTensor()
        g_argmaxes = torch.CudaLongTensor()
        g_ycopy = torch.CudaLongTensor()
    end
    model:evaluate()
    local correct, total = 0, 0
    for j = 1, #valbatches do
        local sent = valbatches[j].sent:cudaLong()
        local ent_dists = valbatches[j].ent_dists:cudaLong()
        local num_dists = valbatches[j].num_dists:cudaLong()
        local labels = valbatches[j].labels:cuda()
        local preds = model:forward({sent, ent_dists, num_dists})
        g_maxes:resize(sent:size(1), 1)
        g_argmaxes:resize(sent:size(1), 1)
        g_ycopy:resize(sent:size(1))
        torch.max(g_maxes, g_argmaxes, preds, 2)
        g_ycopy:copy(labels)
        correct = correct + g_ycopy:eq(g_argmaxes:view(-1)):sum()
        total = total + sent:size(1)
    end
    local acc = correct/total
    model:training()
    return acc
end


function get_multilabel_acc(model, valbatches, ignoreIdx, convens, lstmens)
    if not g_maxes then
        g_maxes = torch.CudaTensor()
        g_argmaxes = torch.CudaLongTensor()
        g_one_hot = torch.CudaTensor()
        g_correct_buf = torch.CudaTensor()
        g_ens_scores = torch.CudaTensor()
    end
    model:evaluate()
    if convens then
        for j = 1, #convens do
            convens[j]:evaluate()
        end
    end
    if lstmens then
        for j = 1, #lstmens do
            lstmens[j]:evaluate()
        end
    end
    local correct, total, ignored = 0, 0, 0
    local pred5s, true5s = 0, 0
    local nonnolabel = 0
    for j = 1, #valbatches do
        local sent = valbatches[j].sent:cudaLong()
        local ent_dists = valbatches[j].ent_dists:cudaLong()
        local num_dists = valbatches[j].num_dists:cudaLong()
        local labels = valbatches[j].labels:cudaLong()
        local labelnums = valbatches[j].labelnums
        local preds

        if convens then
            local enpreds1 = convens[1]:forward({sent, ent_dists, num_dists})
            if opt.geom then
                enpreds1:log()
            end
            for j = 2, #convens do
                local enpredsj = convens[j]:forward({sent, ent_dists, num_dists})
                if opt.geom then
                    enpredsj:log()
                end
                enpreds1:add(enpredsj)
            end
            preds = enpreds1
        end

        if lstmens then
            local enpreds1 = lstmens[1]:forward({sent, ent_dists, num_dists})
            if opt.geom then
                enpreds1:log()
            end
            for j = 2, #lstmens do
                local enpredsj = lstmens[j]:forward({sent, ent_dists, num_dists})
                if opt.geom then
                    enpredsj:log()
                end
                enpreds1:add(enpredsj)
            end
            if preds then
                preds:add(enpreds1)
            else
                preds = enpreds1
            end
        end

        if not convens and not lstmens then
            preds = model:forward({sent, ent_dists, num_dists})
        end

        g_maxes:resize(sent:size(1), 1)
        g_argmaxes:resize(sent:size(1), 1)
        torch.max(g_maxes, g_argmaxes, preds, 2)
        --pred5s = pred5s + g_argmaxes:eq(5):sum()
	--true5s = true5s + labels:eq(5):sum()
	nonnolabel = nonnolabel + labels:select(2,1):ne(ignoreIdx):sum()
        --g_one_hot:resize(sent:size(1), labels:size(2)):zero()
	g_one_hot:resize(sent:size(1), preds:size(2)):zero()
        local numpreds = 0
        local in_denominator = g_argmaxes
        for k = 1, sent:size(1) do
            if not ignoreIdx or in_denominator[k][1] ~= ignoreIdx then
                g_one_hot[k]:indexFill(1, labels[k]:sub(1, labelnums[k]), 1)
                numpreds = numpreds + 1
            end
        end
        g_correct_buf:resize(sent:size(1), 1):zero()
        g_correct_buf:gather(g_one_hot, 2, g_argmaxes)
        correct = correct + g_correct_buf:sum()
        total = total + numpreds
        ignored = ignored + sent:size(1) - numpreds
    end
    local acc = correct/total
    local rec = correct/nonnolabel
    print("rec", rec)
    print("ignored", ignored/(ignored+total))
    model:training()
    return acc, rec
end

function idxstostring(t, dict)
    local strtbl = {}
    local forlimit = t.size and t:size(1) or #t
    for i = 1, forlimit do
        --print(t[i], dict[t[i]])
        table.insert(strtbl, dict[t[i]])
    end
    --assert(false)
    return stringx.join(' ', strtbl)
end

function get_args(sent, ent_dists, num_dists, dict)
    --local min_entdist = ent_dists:min()
    --local min_numdist = num_dists:min()
    local entwrds, numwrds = {}, {}
    for i = 1, sent:size(1) do
        if ent_dists[i]+min_entdist-1 == 0 then
            table.insert(entwrds, sent[i])
        end
        if num_dists[i]+min_numdist-1 == 0 then
            table.insert(numwrds, sent[i])
        end
    end
    return idxstostring(entwrds, dict), idxstostring(numwrds, dict)
end


function eval_gens(predbatches, ignoreIdx, boxrestartidxs, convens, lstmens)
  local ivocab = get_dict(opt.dict_pfx .. ".dict", true)
  local ilabels = get_dict(opt.dict_pfx .. ".labels", true)
  local tupfile = assert(io.open(opt.preddata .. "-tuples.txt", 'w'))

  if ignoreIdx then
    assert(ilabels[ignoreIdx] == "NONE")
  end

  local boxRestarts
  if boxrestartidxs then
    boxRestarts = {}
    assert(boxrestartidxs:dim() == 1)
    for i = 1, boxrestartidxs:size(1) do
      boxRestarts[boxrestartidxs[i]] = true
    end
  end

  if not g_maxes then
    g_maxes = torch.CudaTensor()
    g_argmaxes = torch.CudaLongTensor()
    g_one_hot = torch.CudaTensor()
    g_correct_buf = torch.CudaTensor()
  end

  if convens then
    for j = 1, #convens do
      convens[j]:evaluate()
    end
  end

  if lstmens then
    for j = 1, #lstmens do
      lstmens[j]:evaluate()
    end
  end

  local correct, total = 0, 0
  local candNum = 0 -- numberth candidate, so we can keep track of when tables change
  local seen = {}
  local ndupcorrects = 0
  local nduptotal = 0
  for j = 1, #predbatches do
    local sent = predbatches[j].sent:cudaLong()
    local ent_dists = predbatches[j].ent_dists:cudaLong()
    local num_dists = predbatches[j].num_dists:cudaLong()
    local labels = predbatches[j].labels:cudaLong()
    local labelnums = predbatches[j].labelnums
    local preds

    if convens then
      local enpreds1 = convens[1]:forward({sent, ent_dists, num_dists})
      if opt.geom then
        enpreds1:log()
      end
      for j = 2, #convens do
        local enpredsj = convens[j]:forward({sent, ent_dists, num_dists})
        if opt.geom then
          enpredsj:log()
        end
        enpreds1:add(enpredsj)
      end
      preds = enpreds1
    end

    if lstmens then
      local enpreds1 = lstmens[1]:forward({sent, ent_dists, num_dists})
      if opt.geom then
        enpreds1:log()
      end
      for j = 2, #lstmens do
        local enpredsj = lstmens[j]:forward({sent, ent_dists, num_dists})
        if opt.geom then
          enpredsj:log()
        end
        enpreds1:add(enpredsj)
      end

      if preds then
        preds:add(enpreds1)
      else
        preds = enpreds1
      end
    end

    g_maxes:resize(sent:size(1), 1)
    g_argmaxes:resize(sent:size(1), 1)
    torch.max(g_maxes, g_argmaxes, preds, 2)
    g_one_hot:resize(sent:size(1), preds:size(2)):zero()
    local numpreds = 0
    local in_denominator = g_argmaxes
    for k = 1, sent:size(1) do
      if not ignoreIdx or in_denominator[k][1] ~= ignoreIdx then
        g_one_hot[k]:indexFill(1, labels[k]:sub(1, labelnums[k]), 1)
        numpreds = numpreds + 1
      end
    end

    g_correct_buf:resize(sent:size(1), 1):zero()
    g_correct_buf:gather(g_one_hot, 2, g_argmaxes)

    for k = 1, sent:size(1) do
      candNum = candNum + 1
      if boxRestarts and boxRestarts[candNum] then
          tupfile:write('\n')
          seen = {}
      end
      if not ignoreIdx or in_denominator[k][1] ~= ignoreIdx then
        local sentstr = idxstostring(sent[k], ivocab)
        local entarg, numarg = get_args(sent[k], ent_dists[k], num_dists[k], ivocab)
        local predkey = entarg .. numarg .. ilabels[g_argmaxes[k][1]]
        tupfile:write(entarg, '|', numarg, '|', ilabels[g_argmaxes[k][1]], '\n')
        if g_correct_buf[k][1] > 0 then
          if seen[predkey] then
              ndupcorrects = ndupcorrects + 1
          end
        end
        if seen[predkey] then
            nduptotal = nduptotal + 1
        end
        seen[predkey] = true
      end
    end

    correct = correct + g_correct_buf:sum()
    total = total + numpreds
  end
  local acc = correct/total
  print("prec", acc)
  print("nodup prec", ( correct - ndupcorrects ) / ( total - nduptotal ))
  print("total correct", correct) -- total number of possible correct is fixed and constant, so just reporting this /is/ recall
  print("nodup correct", correct - ndupcorrects)
  tupfile:close()
  return acc
end


function set_up_saved_models()
 --[[
  local convens_paths = {"convie-ep9-1.t7",
                         "convie-ep9-2.t7",
			 "convie-ep8-3.t7"}

  local lstmens_paths = {"blstmie-ep7-1.t7",
	                 "blstmie-ep7-2.t7",
			 "blstmie-ep10-3.t7"}
  --]]
  local convens_paths = {"conv1ie-ep6-94-74.t7",
                         "conv2ie-ep3-94-60.t7",
			 "conv3ie-ep8-95-72.t7"}

  local lstmens_paths = {"blstm1ie-ep4-93-75.t7",
	                 "blstm2ie-ep3-93-71.t7",
			 "blstm3ie-ep2-94-72.t7"}  
  opt.embed_size = 200
  opt.num_filters = 200
  opt.conv_fc_layer_size = 500
  opt.blstm_fc_layer_size = 700
  return convens_paths, lstmens_paths
end

function main()
    torch.manualSeed(opt.seed)
    cutorch.manualSeed(opt.seed)
    cutorch.setDevice(opt.gpuid)

    local trbatches, valbatches, V_sizes, nlabels, pred_batches, pboxrestartidxs = prep_data(opt.batchsize)
    local emb_sizes = {opt.embed_size, opt.embed_size/2, opt.embed_size/2}

    if opt.just_eval then
      local convens_paths, lstmens_paths = set_up_saved_models()

      local convens, lstmens

      if convens_paths then
          convens = {}
          for j = 1, #convens_paths do
              local mod = make_conv_model(V_sizes, emb_sizes, nlabels, opt):cuda()
              local p,g = mod:getParameters()
              local saved_p = torch.load(convens_paths[j])
              p:copy(saved_p)
              table.insert(convens, mod)
          end
      end

      if lstmens_paths then
          lstmens = {}
          for j = 1, #lstmens_paths do
              local mod = make_blstm_model(V_sizes, emb_sizes, nlabels, opt):cuda()
              local p,g = mod:getParameters()
              local saved_p = torch.load(lstmens_paths[j])
              p:copy(saved_p)
              table.insert(lstmens, mod)
          end
      end

    	eval_gens(pred_batches, opt.ignore_idx, pboxrestartidxs, convens, lstmens)
      return
    end

    local model
    if opt.lstm then
        model = make_blstm_model(V_sizes, emb_sizes, nlabels, opt):cuda()
    else
        model = make_conv_model(V_sizes, emb_sizes, nlabels, opt):cuda()
    end
    local crit = nn.MarginalNLLCriterion():cuda()
    local params, grads = model:getParameters()

    if opt.uniform_init > 0 then
        params:uniform(-opt.uniform_init, opt.uniform_init)
    end

    local prev_loss = math.huge
    local best_acc = 0

    for i = 1, opt.epochs do
        print("epoch", i, "lr:", opt.lr)
        local loss = 0
        model:training()
	      model:get(1):get(1).weight[word_pad]:zero()
	      model:get(1):get(2).weight[ent_dist_pad]:zero()
	      model:get(1):get(3).weight[num_dist_pad]:zero()
        for j = 1, #trbatches do
            grads:zero()
            local sent = trbatches[j].sent:cudaLong()
            local ent_dists = trbatches[j].ent_dists:cudaLong()
            local num_dists = trbatches[j].num_dists:cudaLong()
            local labels = trbatches[j].labels:cuda()
            local preds = model:forward({sent, ent_dists, num_dists})
            loss = loss + crit:forward(preds, labels)
            local dLdpreds = crit:backward(preds, labels)
            model:backward({sent, ent_dists, num_dists}, dLdpreds)

	          if opt.lstm then
               model:get(1):get(1).gradWeight[word_pad]:zero()
	             model:get(1):get(2).gradWeight[ent_dist_pad]:zero()
	             model:get(1):get(3).gradWeight[num_dist_pad]:zero()
	             local shrinkage = 5/grads:norm(2)
	             if shrinkage < 1 then
	                grads:mul(shrinkage)
	             end
	          end

            params:add(-opt.lr, grads)

	    model:get(1):get(1).weight[word_pad]:zero()
	    model:get(1):get(2).weight[ent_dist_pad]:zero()
	    model:get(1):get(3).weight[num_dist_pad]:zero()
        end
        print("train loss:", loss/#trbatches)

        local acc, rec = get_multilabel_acc(model, valbatches, opt.ignore_idx)
        print("acc:", acc)

	local savefi = string.format("%s-ep%d-%d-%d", opt.savefile, i, math.floor(100*acc), math.floor(100*rec))
        print("saving to", savefi)
        torch.save(savefi, params)
        print("")
        valloss = -acc
        if valloss >= prev_loss then
            opt.lr = opt.lr*opt.lr_decay
        end
        prev_loss = valloss
    end
end

main()
