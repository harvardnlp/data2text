local Aggregator, parent = torch.class('onmt.Aggregator', 'nn.Container')

function Aggregator:__init(nRows, encDim, decDim)
    parent.__init(self)

    self.nRows = nRows
    -- have stuff for both cells and hiddens
    self.cellNet = self:_buildModel(nRows, encDim, decDim)
    self.hidNet = self:_buildModel(nRows, encDim, decDim)
    self:add(self.cellNet)
    self:add(self.hidNet)
    self.layerClones = {} -- use same transformation for every layer
    self.catCtx = torch.Tensor()
end

function Aggregator:_buildModel(nRows, encDim, decDim)
    return nn.Sequential()
            :add(nn.JoinTable(2))
            :add(nn.Linear(nRows*encDim, decDim))
end

-- allEncStates is an nRows-length table containing nLayers-length tables;
-- allCtxs is an nRows-length table containing batchSize x srcLen x dim tensors
function Aggregator:forward(allEncStates, allCtxs)
    -- do aggregation
    if self.train then
        self.layInputs = {}
    end
    local aggEncStates = {}
    for i = 1, #allEncStates[1] do
        if not self.layerClones[i] then
            if i % 2 == 1 then
                self.layerClones[i] = self.cellNet:clone('weight', 'gradWeight', 'bias', 'gradBias')
            else
                self.layerClones[i] = self.hidNet:clone('weight', 'gradWeight', 'bias', 'gradBias')
            end
        end
        -- get all the stuff we're concatenating
        local layInput = {}
        for j = 1, self.nRows do
            table.insert(layInput, allEncStates[j][i])
        end
        if self.train then
            table.insert(self.layInputs, layInput)
        end
        table.insert(aggEncStates, self.layerClones[i]:forward(layInput))
    end

    -- now concatenate all the contexts
    local firstCtx = allCtxs[1]
    local rowLen = firstCtx:size(2) -- assumed constant for all rows
    self.catCtx:resize(firstCtx:size(1), self.nRows*rowLen, firstCtx:size(3))
    -- just copy
    for j = 1, self.nRows do
        self.catCtx:narrow(2, (j-1)*rowLen + 1, rowLen):copy(allCtxs[j])
    end

    return aggEncStates, self.catCtx
end

-- encGradStatesOut is an nLayers-length table;
-- gradContext sho
function Aggregator:backward(encGradStatesOut, gradContext, inputFeed)
    local allEncGradOuts = {}
    for j = 1, self.nRows do
        allEncGradOuts[j] = {}
    end
    local ifOffset = inputFeed == 1 and 1 or 0
    for i = 1, #encGradStatesOut - ifOffset do
        local gradIns = self.layerClones[i]:backward(self.layInputs[i], encGradStatesOut[i])
        for j = 1, self.nRows do
            table.insert(allEncGradOuts[j], gradIns[j])
        end
    end

    -- unconcatenate catCtx
    local gradCtxs = {}
    local rowLen = gradContext:size(2)/self.nRows
    for j = 1, self.nRows do
        table.insert(gradCtxs, gradContext:narrow(2, (j-1)*rowLen + 1, rowLen))
    end

    return allEncGradOuts, gradCtxs
end

-- function Aggregator:postParametersInitialization()
--   self:reset() -- should reset Linears
-- end


function Aggregator:serialize()
  return {
    modules = self.modules,
    args = {self.nRows}
  }
end
