-- I THINK this is for ignore datasets
--[[ Data management and batch creation. Handles data created by `preprocess.lua`. ]]
local BoxDataset2 = torch.class("BoxDataset2")

--[[ Initialize a data object given aligned tables of IntTensors `srcData`
  and `tgtData`.
--]]
function BoxDataset2:__init(srcData, tgtData, colStartIdx, nFeatures,
      copyGenerate, version, tripV, switch, multilabel)

  self.srcs = srcData.words
  self.srcFeatures = srcData.features
  self.srcTriples = srcData.triples
  self.tripV = tripV
  self.switch = switch
  self.multilabel = multilabel

  if tgtData ~= nil then
    self.tgt = tgtData.words
    self.tgtFeatures = tgtData.features
    self.pointers = switch and tgtData.pointers
  end
  -- source length(s) don't change (and we'll pad line scores...)
  self.maxSourceLength = self.srcs[1][1]:size(1)
  self.nSourceRows = #self.srcs

  self.colStartIdx = colStartIdx -- idx after vocab where stuff starts
  self.nFeatures  = nFeatures
  self.copyGenerate = copyGenerate
  self.version = version
end

--[[ Setup up the training data to respect `maxBatchSize`. ]]
function BoxDataset2:setBatchSize(maxBatchSize)

  self.batchRange = {}
  self.maxTargetLength = 0

  -- Prepares batches in terms of range within self.src and self.tgt.
  local offset = 0
  local batchSize = 1
  local sourceLength = 0
  local targetLength = 0

  for i = 1, #self.tgt do
    -- All sources are the same size; there are rarely enough targets of same length
    -- to really batch, so will have padding on targets, as usual
    if batchSize == maxBatchSize or i == 1 then --or self.tgt[i]:size(1) ~= targetLength then
      if i > 1 then
        table.insert(self.batchRange, { ["begin"] = offset, ["end"] = i - 1 })
      end

      offset = i
      batchSize = 1
      --targetLength = self.tgt[i]:size(1)
    else
      batchSize = batchSize + 1
    end

    --self.maxTargetLength = math.max(self.maxTargetLength, self.tgt[i]:size(1))

    -- Target contains <s> and </s>.
    local targetSeqLength = self.tgt[i]:size(1) - 1
    --targetLength = math.max(targetLength, targetSeqLength)
    self.maxTargetLength = math.max(self.maxTargetLength, targetSeqLength)
  end
  -- catch last thing
  table.insert(self.batchRange, { ["begin"] = offset, ["end"] = #self.tgt })
end

--[[ Return number of batches. ]]
function BoxDataset2:batchCount()
  if self.batchRange == nil then
    return 1
  end
  return #self.batchRange
end

--[[ Get `Batch` number `idx`. If nil make a batch of all the data. ]]
function BoxDataset2:getBatch(idx)
  if idx == nil or self.batchRange == nil then
      assert(false)
    return onmt.data.BoxBatch.new(self.srcs, self.srcFeatures, self.tgt,
      self.tgtFeatures, self.maxSourceLength)
  end

  local rangeStart = self.batchRange[idx]["begin"]
  local rangeEnd = self.batchRange[idx]["end"]

  local srcs = {}
  for j = 1, #self.srcs do srcs[j] = {} end
  local tgt = {}
  local triples = {}
  local pointers = {}

  local srcFeatures = {}
  local tgtFeatures = {}

  for i = rangeStart, rangeEnd do
    for j = 1, #self.srcs do
        table.insert(srcs[j], self.srcs[j][i])
    end
    table.insert(tgt, self.tgt[i])

    if self.srcTriples then
        table.insert(triples, self.srcTriples[i]:long())
    end

    if self.switch then
        table.insert(pointers, self.pointers[i])
    end

    if self.srcFeatures[i] then
      table.insert(srcFeatures, self.srcFeatures[i])
    end

    if self.tgtFeatures[i] then
      table.insert(tgtFeatures, self.tgtFeatures[i])
    end
  end

  local bb
  if self.switch then
      bb = onmt.data.BoxSwitchBatch.new(srcs, srcFeatures, tgt, tgtFeatures,
            self.maxSourceLength, self.colStartIdx, self.nFeatures,
            pointers, self.multilabel)
  else
      bb = onmt.data.BoxBatch3.new(srcs, srcFeatures, tgt, tgtFeatures,
        self.maxSourceLength, self.colStartIdx, self.nFeatures,
        triples, self.tripV)
  end
  return bb
end

return BoxDataset2
