local BoxTableEncoder, parent = torch.class('onmt.BoxTableEncoder', 'nn.Container')

function BoxTableEncoder:__init(args)
    parent.__init(self)
    self.args = args
    self.network = self:_buildModel()
    self:add(self.network)
end

-- --[[ Return a new Encoder using the serialized data `pretrained`. ]]
-- function BoxTableEncoder.load(pretrained)
--     assert(false)
--   local self = torch.factory('onmt.Encoder')()
--
--   self.args = pretrained.args
--   parent.__init(self, pretrained.modules[1])
--
--   self:resetPreallocation()
--
--   return self
-- end

--[[ Return data to serialize. ]]
function BoxTableEncoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end

-- function Encoder:resetPreallocation()
--   -- Prototype for preallocated hidden and cell states.
--   self.stateProto = torch.Tensor()
--
--   -- Prototype for preallocated output gradients.
--   self.gradOutputProto = torch.Tensor()
--
--   -- Prototype for preallocated context vector.
--   self.contextProto = torch.Tensor()
-- end

function BoxTableEncoder:_buildModel()
    local args = self.args
    local x = nn.Identity()() -- batcSize*nRows*srcLen x nFeatures
    local lut = nn.LookupTable(args.vocabSize, args.wordVecSize)
    self.lut = lut
    local featEmbs
    if args.feat_merge == "concat" then
        -- concatenates embeddings of all features and applies MLP
        featEmbs = nn.Linear(args.nFeatures*args.wordVecSize, args.encDim)(
            nn.View(-1, args.nFeatures*args.wordVecSize)(
             lut(x)))
    else
        assert(args.wordVecSize == args.encDim)
        -- adds embeddings of all features and applies bias and nonlinearity
        -- (i.e., embeds sparse features)
        featEmbs = nn.Add(args.wordVecSize)(
                     nn.Sum(2)(
                        lut(x)))
    end
    featEmbs = args.relu and nn.ReLU()(featEmbs) or nn.Tanh()(featEmbs)
    -- featEmbs are batchSize*nRows*nCols x encDim

    for i = 2, args.nLayers do
        if args.dropout and args.dropout > 0 then
            featEmbs = nn.Dropout(args.dropout)(featEmbs) -- maybe don't want?
        end
        featEmbs = nn.Linear(args.encDim, args.encDim)(featEmbs) -- wrong for summing, but that seems worse anyway
        featEmbs = args.relu and nn.ReLU()(featEmbs) or nn.Tanh()(featEmbs)
    end

    -- if args.dropout and args.dropout > 0 then
    --     featEmbs = nn.Dropout(args.dropout)(featEmbs) -- maybe don't want?
    -- end

    -- attn ctx should be batchSize x nRows*nCols x dim
    local ctx
    if args.encDim ~= args.decDim then
        ctx = nn.View(-1, args.nRows*args.nCols, args.decDim)(nn.Linear(args.encDim, args.decDim)(featEmbs))
    else
        ctx = nn.View(-1, args.nRows*args.nCols, args.encDim)(featEmbs)
    end

    -- for now let's assume we also want row-wise summaries
    local byRows = nn.View(-1, args.nCols, args.encDim)(featEmbs) -- batchSize*nRows x nCols x dim
    if args.pool == "mean" then
        byRows = nn.Mean(2)(byRows)
    else
        byRows = nn.Max(2)(byRows)
    end
    -- byRows is now batchSize*nRows x dim
    local flattenedByRows = nn.View(-1, args.nRows*args.encDim)(byRows) -- batchSize x nRows*dim

    -- finally need to make something that can be copied into an lstm
    self.transforms = {}
    local outputs = {}
    for i = 1, args.effectiveDecLayers do
        local lin = nn.Linear(args.nRows*args.encDim, args.decDim)
        table.insert(self.transforms, lin)
        table.insert(outputs, lin(flattenedByRows))
    end

    table.insert(outputs, ctx)
    local mod = nn.gModule({x}, outputs)
    -- output is a table with an encoding for each layer of the dec, followed by the ctx
    return mod
end

function BoxTableEncoder:shareTranforms()
    for i = 3, #self.transforms do
        if i % 2 == 1 then
            self.transforms[i]:share(self.transforms[1], 'weight', 'gradWeight', 'bias', 'gradBias')
        else
            self.transforms[i]:share(self.transforms[2], 'weight', 'gradWeight', 'bias', 'gradBias')
        end
    end
end

--[[Compute the context representation of an input.

Parameters:

  * `batch` - as defined in batch.lua.

Returns:

  1. - final hidden states: layer-length table with batchSize x decDim tensors
  2. - context matrix H: batchSize x nRows*nCols x encDim
--]]
function BoxTableEncoder:forward(batch)
  local finalStates = self.network:forward(batch:getSource())
  local context = table.remove(finalStates) -- pops, i think
  return finalStates, context
end

--[[ Backward pass (only called during training)

  Parameters:

  * `batch` - must be same as for forward
  * `gradStatesOutput` gradient of loss wrt last state
  * `gradContextOutput` - gradient of loss wrt full context.

  Returns: `gradInputs` of input network.
--]]
function BoxTableEncoder:backward(batch, gradStatesOutput, gradContextOutput)
    local encGradOut = {}
    for i = 1, self.args.effectiveDecLayers do -- ignore input feed (and attn outputs)
        table.insert(encGradOut, gradStatesOutput[i])
    end
    table.insert(encGradOut, gradContextOutput)
    local gradInputs = self.network:backward(batch:getSource(), encGradOut)
    return gradInputs
end
