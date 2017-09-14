--[[ Return the maxLength, sizes, and non-zero count
  of a baBoxBatch`seq`s ignoring `ignore` words.
--]]
local function getLength(seq, ignore)
  local sizes = torch.IntTensor(#seq):zero()
  local max = 0
  local sum = 0

  for i = 1, #seq do
    local len = seq[i]:size(1)
    if ignore ~= nil then
      len = len - ignore
    end
    max = math.max(max, len)
    sum = sum + len
    sizes[i] = len
  end
  return max, sizes, sum
end

--[[ Data management and batch creation.

Batch interface reference [size]:

  * size: number of sentences in the batch [1]
  * sourceLength: max length in source batch [1]
  * sourceSize:  lengths of each source [batch x 1]
  * sourceInput:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * sourceInputFeatures: table of source features sequences
  * sourceInputRev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * sourceInputRevFeatures: table of reversed source features sequences
  * targetLength: max length in source batch [1]
  * targetSize: lengths of each source [batch x 1]
  * targetNonZeros: number of non-ignored words in batch [1]
  * targetInput: input idx's of target (SABCDEPPPPPP) [batch x max]
  * targetInputFeatures: table of target input features sequences
  * targetOutput: expected output idx's of target (ABCDESPPPPPP) [batch x max]
  * targetOutputFeatures: table of target output features sequences

 TODO: change name of size => maxlen
--]]

--[[ A batch of sentences to translate and targets. Manages padding,
  features, and batch alignment (for efficiency).

  Used by the decoder and encoder objects.
--]]
local BoxSwitchBatch = torch.class('BoxSwitchBatch')

--[[ Create a batch object.

Parameters:

  * `src` - 2D table of source batch indices
  * `srcFeatures` - 2D table of source batch features (opt)
  * `tgt` - 2D table of target batch indices
  * `tgtFeatures` - 2D table of target batch features (opt)
  * `pointers`   - table of numPtrLocs x (1 + maxNumPtrs + 1)
--]]
function BoxSwitchBatch:__init(srcs, srcFeatures, tgt, tgtFeatures, bsLen,
    colStartIdx, nFeatures, pointers, multilabel)
  local srcs = srcs or {}

  if tgt ~= nil then
    assert(#srcs[1] == #tgt, "source and target must have the same batch size")
  end

  self.size = #tgt

  self.sourceLength = bsLen-1 -- skipping first col...
  self.totalSourceLength = #srcs*self.sourceLength -- all rows
  assert(srcs[1][1]:size(1) == bsLen)
  local srcLen = self.sourceLength
  local vocabSize = colStartIdx+2*srcLen+1
  --self.sourceLength, self.sourceSize = getLength(src)

  --local sourceSeq = torch.IntTensor(#srcs, self.sourceLength, self.size):fill(onmt.Constants.PAD)
  -- source concatenates all rows in the table into a single column (and concatenates everything in the batch too)
  self.sourceInput = torch.LongTensor(self.size*self.totalSourceLength, nFeatures)
  --self.sourceInput = sourceSeq:clone()

  if tgt ~= nil then
    -- N.B. targetSize is now wrongish....
    self.rulTargetLength, self.rulTargetSize, self.rulTargetNonZeros = getLength(tgt, 1)
    self.targetLength = self.rulTargetLength -- will change this, since this is what decoder looks at
    self.targetNonZeros = self.rulTargetNonZeros
    self.targetSize = self.rulTargetSize

    local targetSeq = torch.LongTensor(self.rulTargetLength, self.size):fill(onmt.Constants.PAD)
    self.targetInput = targetSeq:clone()
    self.targetOutput = targetSeq:clone()
    self.zs = torch.zeros(self.rulTargetLength, self.size)
    if multilabel then
        -- find max labels
        local maxLabels = 1
        for b = 1, self.size do
            if pointers[b]:dim() > 0 then
                maxLabels = math.max(maxLabels, pointers[b]:size(2)-2)
            end
        end
        self.pointerTargets = torch.ones(self.rulTargetLength, self.size, maxLabels+1)
    else
        self.pointerTargets = torch.ones(self.rulTargetLength, self.size)
    end
  end

  if tripIdxs ~= nil and #tripIdxs > 0 and tripV then
      self.triples = torch.zeros(self.size, tripIdxs[1]:size(1), tripV[1]+tripV[2]+tripV[3])
  end

  local currRow = 1

  for b = 1, self.size do
    for j = 1, #srcs do
        local sourceInput = srcs[j][b]:sub(2, srcs[j][b]:size(1)) -- skip first (ok for linescore since padded)
        self.sourceInput:sub(currRow, currRow+srcLen-1, 1, 1):copy(sourceInput)

        -- -- Source input is left padded [PPPPPPABCDE] .
        -- self.sourceInput[j][{{sourceOffset, self.sourceLength}, b}]:copy(sourceInput)
        -- self.sourceInputPadLeft = true

        if j <= 2*g_nRegRows then
            -- second feature is row name; conceivable we would want a different vocab for these but
            -- since they don't appear in the rows it's probably fine
            self.sourceInput:sub(currRow, currRow+srcLen-1, 2, 2):fill(srcs[j][b][1])
            -- third feature is col name
            self.sourceInput:sub(currRow, currRow+srcLen-1, 3, 3)
              :range(colStartIdx, colStartIdx+srcLen-1)
            -- fourth feature is home or away
            local lastFeat = j <= g_nRegRows and colStartIdx+2*srcLen or colStartIdx+2*srcLen+1
            self.sourceInput:sub(currRow, currRow+srcLen-1, 4, 4):fill(lastFeat)
        else
            self.sourceInput:sub(currRow, currRow+srcLen-1, 2, 2):fill(srcs[j][b][g_specPadding+1])
            self.sourceInput:sub(currRow, currRow+srcLen-1, 3, 3)
              :range(colStartIdx+srcLen, colStartIdx+2*srcLen-1)
            local lastFeat = j < #srcs and colStartIdx+2*srcLen or colStartIdx+2*srcLen+1
            self.sourceInput:sub(currRow, currRow+srcLen-1, 4, 4):fill(lastFeat)
        end
        currRow = currRow + srcLen
    end

    if tgt ~= nil then
      -- Input: [<s>ABCDE]
      -- Output: [ABCDE</s>]
      local targetLength = tgt[b]:size(1) - 1
      local targetInput = tgt[b]:narrow(1, 1, targetLength)
      local targetOutput = tgt[b]:narrow(1, 2, targetLength)

      -- Target is right padded [<S>ABCDEPPPPPP] .
      self.targetInput[{{1, targetLength}, b}]:copy(targetInput)
      self.targetOutput[{{1, targetLength}, b}]:copy(targetOutput)

      -- pointers[b] is numPtrs x (1+maxLabels+1)
      -- note that below is correct b/c preproc appends a BOS token, and
      -- the pointers have the (unpadded) tgtIndices that point to srcIndices.
      -- so we get that the zs indicate the word preceding a copied word
      if pointers[b]:dim() > 0 then -- sometimes we have no pointer info i guess
          local maxLabels_b = pointers[b]:size(2)-2
          for jj = 1, pointers[b]:size(1) do
              local t = pointers[b][jj][1]
              self.zs[t][b] = 1 -- a pointer
              if multilabel then
                  -- copy label indices
                  self.pointerTargets[t][b]:sub(1, maxLabels_b)
                      :copy(pointers[b][jj]:sub(2, maxLabels_b+1))
                  -- put number of nz labels in last spot
                  self.pointerTargets[t][b][self.pointerTargets:size(3)] = pointers[b][jj][maxLabels_b+2]
              else -- just take first
                  self.pointerTargets[t][b] = pointers[b][jj][2]
              end
          end
      end
    end

    -- make one hot (concatenated) triple representation
    if tripIdxs ~= nil and #tripIdxs > 0 and tripV then
        self.triples[b]:narrow(2, 1, tripV[1]):scatter(2, tripIdxs[b]:narrow(2, 1, 1), 1)
        self.triples[b]:narrow(2, tripV[1]+1, tripV[2]):scatter(2, tripIdxs[b]:narrow(2, 2, 1), 1)
        self.triples[b]:narrow(2, tripV[1]+tripV[2]+1, tripV[3]):scatter(2, tripIdxs[b]:narrow(2, 3, 1), 1)
    end

  end -- end for b
  --print(currRow, self.sourceInput:size(1))
  assert(currRow == self.sourceInput:size(1)+1)

  self.targetOffset = 0 -- used for long target stuff
end

function BoxSwitchBatch:splitIntoPieces(maxBptt)
    self.maxBptt = maxBptt
    self.targetLength = math.min(self.rulTargetLength, maxBptt)
    return math.ceil(self.rulTargetLength/maxBptt)
end

function BoxSwitchBatch:nextPiece()
    self.targetOffset = self.targetOffset + self.maxBptt
    self.targetLength = math.min(self.rulTargetLength-self.targetOffset, self.maxBptt)
    self.targetNonZeros = 0 -- so we only count this once...
end

-- -- would be faster to precompute everything for each minibatch, but might be tricky....
-- function getBatchLocations(srcLocs, tgt)
--     for b = 1, #tgt do
--         local targetLength = tgt[b]:size(1) - 1
--         local targetOutput = tgt[b]:narrow(1, 2, targetLength)
--         for t = 1, targetLength do
--             if srcLocs[b][targetOutput[t]] then


local function addInputFeatures(inputs, featuresSeq, t)
  local features = {}
  for j = 1, #featuresSeq do
    table.insert(features, featuresSeq[j][t])
  end
  if #features > 1 then
    table.insert(inputs, features)
  else
    onmt.utils.Table.append(inputs, features)
  end
end

--[[ Get source batch at timestep `t`. --]]
function BoxSwitchBatch:getSourceInput(t)
    assert(false)
  -- If a regular input, return word id, otherwise a table with features.
  local inputs = self.sourceInput[self.inputRow][t]

  if self.batchRowFeats then
      inputs = {inputs, self.batchRowFeats[self.inputRow], self.batchColFeats[t]}
  end

  -- if #self.sourceInputFeatures > 0 then
  --   inputs = { inputs }
  --   addInputFeatures(inputs, self.sourceInputFeatures, t)
  -- end

  return inputs
end

-- returns a nRows*srcLen x batchSize tensor
function BoxSwitchBatch:getSource()
    return self.sourceInput
end

function BoxSwitchBatch:getSourceWords()
    return self.sourceInput:select(2,1):reshape(self.size, self.totalSourceLength)
end

function BoxSwitchBatch:getCellsForExample(b)
    return self.sourceInput
      :sub((b-1)*self.totalSourceLength+1, b*self.totalSourceLength):select(2,1)
end

function BoxSwitchBatch:getSourceTriples()
    return self.triples
end

--[[ Get target input batch at timestep `t`. --]]
function BoxSwitchBatch:getTargetInput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local inputs = self.targetInput[self.targetOffset + t]

  return inputs
end

--[[ Get target output batch at timestep `t` (values t+1). --]]
function BoxSwitchBatch:getTargetOutput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local outputs = { self.targetOutput[self.targetOffset + t] }

  return outputs
end

function BoxSwitchBatch:getZs(t)
    return self.zs[self.targetOffset + t]
end

function BoxSwitchBatch:getPointerTargets(t)
    return self.pointerTargets[self.targetOffset + t]
end

return BoxSwitchBatch
