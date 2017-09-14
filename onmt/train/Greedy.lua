local stringx = require('pl.stringx')

local function get_ngrams(s, n, count)
   local ngrams = {}
   count = count or 0
   for i = 1, #s do
      for j = i, math.min(i+n-1, #s) do
    local ngram = table.concat(s, ' ', i, j)
    local l = j-i+1 -- keep track of ngram length
    if count == 0 then
       table.insert(ngrams, ngram)
    else
       if ngrams[ngram] == nil then
          ngrams[ngram] = {1, l}
       else
          ngrams[ngram][1] = ngrams[ngram][1] + 1
       end
    end
      end
   end
   return ngrams
end

local function get_ngram_prec(cand, ref, n)
   -- n = number of ngrams to consider
   local results = {}
   for i = 1, n do
      results[i] = {0, 0} -- total, correct
   end
   local cand_ngrams = get_ngrams(cand, n, 1)
   local ref_ngrams = get_ngrams(ref, n, 1)
   for ngram, d in pairs(cand_ngrams) do
      local count = d[1]
      local l = d[2]
      results[l][1] = results[l][1] + count
      local actual
      if ref_ngrams[ngram] == nil then
    actual = 0
      else
    actual = ref_ngrams[ngram][1]
      end
      results[l][2] = results[l][2] + math.min(actual, count)
   end
   return results
end

local function convert_tostring(ts, size, dict)
   --assert(ts:dim() == 1)
   local strtbl = {}
   for i = 1, size do
       table.insert(strtbl, dict.idxToLabel[ts[i]])
   end
   return stringx.join(' ', strtbl)
end

local function nestedstringshit(tbl)
    local strbl = {}
    for i = 1, #tbl do
        table.insert(strbl, stringx.join(",", tbl[i]))
    end
    return stringx.join(" ", strbl)
end

local function convert_predtostring(ts, size, dict, probs, n)
   --assert(ts:dim() == 1)
   local strtbl = {}
   for i = 1, size do
       table.insert(strtbl, dict.idxToLabel[ts[i]])
       if probs and i > 1 then
           table.insert(strtbl, "[")
           --table.insert(strtbl, stringx.join(",", probs[i-1][n]))
           table.insert(strbl, nestedstringshit(probs[i-1][n]))
           table.insert(strtbl, "]")
       end
   end
   return stringx.join(' ', strtbl)
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

local function greedy_eval(model, data, src_dict, targ_dict,
    start_print_batch, end_print_batch, verbose)

  local start_print_batch = start_print_batch or 0
  local ngram_crct = torch.zeros(4)
  local ngram_total = torch.zeros(4)
  --local probs = torch.CudaTensor()

  allEvaluate(model)

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local aggEncStates, catCtx = allEncForward(model, batch)
    model.decoder:resetLastStates()
    --probs:resize(batch.targetLength, batch.size)
    local preds, probs
    if verbose and i >= start_print_batch and i <= end_print_batch then
        --preds, probs = model.decoder:greedyFixedFwd2(batch, aggEncStates, catCtx)
        preds, probs = model.decoder:greedyFixedFwd3(batch, aggEncStates, catCtx)
    else
        preds, probs = model.decoder:greedyFixedFwd(batch, aggEncStates, catCtx, probs)
    end
    -- if i >= start_print_batch and i <= end_print_batch then
    --     local brows = math.min(50, batch.targetLength)
    --     print(probs:sub(1, brows))
    -- end
    for n = 1, batch.size do
        -- will just go up to true gold_length
        local trulen = batch.targetSize[n]
        local pred_sent = preds:select(2, n):sub(2, trulen+1):totable()
        local gold_sent
        if batch.targetOutput:dim() == 3 then
            gold_sent = batch.targetOutput:select(2, n)
                 :sub(1, trulen):select(2, 1):totable()
        else
           gold_sent = batch.targetOutput:select(2, n)
             :sub(1, trulen):totable()
        end
        local prec = get_ngram_prec(pred_sent, gold_sent, 4)
        for ii = 1, 4 do
            ngram_crct[ii] = ngram_crct[ii] + prec[ii][2]
            ngram_total[ii] = ngram_total[ii] + prec[ii][1]
        end
        if i >= start_print_batch and i <= end_print_batch then
            -- local left_string = convert_tostring(batch.source_input:select(2, n),
            --     batch.source_length, src_dict)
            local targ_string = convert_tostring(batch.targetInput:select(2, n),
                batch.targetLength, targ_dict)
            local gen_targ_string = convert_predtostring(preds:select(2, n),
                batch.targetLength+1, targ_dict, probs, n)
            --print( "Left  :", left_string)
            print( "True  :", targ_string)
            print( "Gen   :", gen_targ_string)
            -- for kk = 1, batch.targetLength do
            --     print(stringx.join(",", probs[kk][n]))
            -- end
            print(" ")
        end
    end
  end

  ngram_crct:cdiv(ngram_total)
  print("Accs", ngram_crct[1], ngram_crct[2], ngram_crct[3], ngram_crct[4])
  ngram_crct:log()
  local bleu = math.exp(ngram_crct:sum()/4) -- no length penalty b/c we know the length
  print("bleu", bleu)

  allTraining(model)
end


local function greedy_gen(model, data, src_dict, targ_dict, max_len)

  allEvaluate(model)

  for i = 1, data:batchCount() do
    local batch = onmt.utils.Cuda.convert(data:getBatch(i))
    local aggEncStates, catCtx = allEncForward(model, batch)
    model.decoder:resetLastStates()
    batch.targetLength = max_len
    local preds, probs = model.decoder:greedyFixedFwd(batch, aggEncStates, catCtx, probs)

    for n = 1, batch.size do
        local gen_targ_string = convert_and_shorten_string(preds:select(2, n)
            :sub(2, max_len+1), max_len, targ_dict)
        print(gen_targ_string)
    end
  end

  allTraining(model)
end


return {
    greedy_eval = greedy_eval,
    greedy_gen = greedy_gen
}
