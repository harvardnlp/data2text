--[[ Data management and batch creation. Handles data created by `preprocess.lua`. ]]
local BoxDataset = torch.class("BoxDataset")

--[[ Initialize a data object given aligned tables of IntTensors `srcData`
  and `tgtData`.
--]]
function BoxDataset:__init(srcData, tgtData, usePosnFeats)

  self.srcs = srcData.words
  self.srcFeatures = srcData.features
  self.usePosnFeats = usePosnFeats

  if tgtData ~= nil then
    self.tgt = tgtData.words
    self.tgtFeatures = tgtData.features
  end
  -- source length(s) don't change (and we'll pad line scores...)
  self.maxSourceLength = self.srcs[1][1]:size(1)
  self.nSourceRows = #self.srcs
  self.cache = {} -- stores batches
  if usePosnFeats then
      self.rowFeats = torch.range(1, self.nSourceRows):long():view(-1, 1) -- need nRows*batchsize tensor for this
      self.colFeats = torch.range(1, self.maxSourceLength):long():view(-1, 1) -- need srcLen*batchSize tensor for this
  end
end

--[[ Setup up the training data to respect `maxBatchSize`. ]]
function BoxDataset:setBatchSize(maxBatchSize)

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
function BoxDataset:batchCount()
  if self.batchRange == nil then
    return 1
  end
  return #self.batchRange
end

--[[ Get `Batch` number `idx`. If nil make a batch of all the data. ]]
function BoxDataset:getBatch(idx, cache)
  if idx == nil or self.batchRange == nil then
    return onmt.data.BoxBatch.new(self.srcs, self.srcFeatures, self.tgt,
      self.tgtFeatures, self.maxSourceLength)
  end

  local bb = self.cache[idx]

  if not bb or not cache then
      local rangeStart = self.batchRange[idx]["begin"]
      local rangeEnd = self.batchRange[idx]["end"]

      local srcs = {}
      for j = 1, #self.srcs do srcs[j] = {} end
      local tgt = {}

      local srcFeatures = {}
      local tgtFeatures = {}

      for i = rangeStart, rangeEnd do
        for j = 1, #self.srcs do
            table.insert(srcs[j], self.srcs[j][i])
        end
        table.insert(tgt, self.tgt[i])

        if self.srcFeatures[i] then
          table.insert(srcFeatures, self.srcFeatures[i])
        end

        if self.tgtFeatures[i] then
          table.insert(tgtFeatures, self.tgtFeatures[i])
        end
      end

      local batchRowFeats, batchColFeats
      if self.usePosnFeats then
          local size = #tgt
          batchRowFeats = self.rowFeats:expand(self.rowFeats:size(1), size)
          batchColFeats = self.colFeats:expand(self.colFeats:size(1), size)
      end

      bb = onmt.data.BoxBatch.new(srcs, srcFeatures, tgt, tgtFeatures,
        self.maxSourceLength, batchRowFeats, batchColFeats)

      if cache then
          self.cache[idx] = bb
      end
  end

  return bb
end

return BoxDataset
