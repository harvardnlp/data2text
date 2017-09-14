local status, module = pcall(require, 'cudnn')
cudnn = status and module or nil
--require 'cudnn'
--[[ Unit to decode a sequence of output tokens.

     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     .      .      .             .
     |      |      |             |
    h_1 => h_2 => h_3 => ... => h_n
     |      |      |             |
     |      |      |             |
    x_1    x_2    x_3           x_n

Inherits from [onmt.Sequencer](onmt+modules+Sequencer).

--]]
local ConvRecDecoder, parent = torch.class('onmt.ConvRecDecoder', 'onmt.Sequencer')


--[[ Construct a decoder layer.

Parameters:

  * `inputNetwork` - input nn module.
  * `rnn` - recurrent module, such as [onmt.LSTM](onmt+modules+LSTM).
  * `generator` - optional, an output [onmt.Generator](onmt+modules+Generator).
  * `inputFeed` - bool, enable input feeding.
--]]
function ConvRecDecoder:__init(inputNetwork, rnn, generator, inputFeed,
    doubleOutput, rec, recViewer, rho, discrec, discdist)
  self.rnn = rnn
  self.inputNet = inputNetwork

  self.args = {}
  self.args.rnnSize = self.rnn.outputSize
  self.args.numEffectiveLayers = self.rnn.numEffectiveLayers
  self.args.rho = rho
  self.args.discrec = discrec
  self.args.discdist = discdist

  self.args.inputIndex = {}
  self.args.outputIndex = {}

  -- Input feeding means the decoder takes an extra
  -- vector each time representing the attention at the
  -- previous step.
  self.args.inputFeed = inputFeed
  self.args.doubleOutput = doubleOutput

  parent.__init(self, self:_buildModel())

  -- The generator use the output of the decoder sequencer to generate the
  -- likelihoods over the target vocabulary.
  self.generator = generator
  self:add(self.generator)
  self.recViewer = recViewer
  self.rec = rec
  self:add(self.rec)
  if self.args.discdist > 0 then
      -- just assuming we're cuda'ing
      local parallelDistCrit = nn.ParallelCriterion()
      local hellinger = self.args.discdist == 2
      -- add one for each distribution
      parallelDistCrit:add(nn.PairwiseDistDist(hellinger))
      parallelDistCrit:add(nn.PairwiseDistDist(hellinger))
      parallelDistCrit:add(nn.PairwiseDistDist(hellinger))
      self.parallelDistCrit = onmt.utils.Cuda.convert(parallelDistCrit)

      for ii = 1, #self.parallelDistCrit.criterions do
          for jj = 1, #self.parallelDistCrit.criterions[ii].crits do
              self.parallelDistCrit.criterions[ii].crits[jj] = onmt.utils.Cuda.convert(self.parallelDistCrit.criterions[ii].crits[jj])
          end
      end
  end

  self:resetPreallocation()
end

--[[ Return a new ConvRecDecoder using the serialized data `pretrained`. ]]
function ConvRecDecoder.load(pretrained)
  local self = torch.factory('onmt.ConvRecDecoder')()

  self.args = pretrained.args

  parent.__init(self, pretrained.modules[1])
  self.generator = pretrained.modules[2]
  self:add(self.generator)

  self:resetPreallocation()

  return self
end

--[[ Return data to serialize. ]]
function ConvRecDecoder:serialize()
  return {
    modules = self.modules,
    args = self.args
  }
end

function ConvRecDecoder:resetPreallocation()
  if self.args.inputFeed then
    self.inputFeedProto = torch.Tensor()
  end

  -- Prototype for preallocated hidden and cell states.
  self.stateProto = torch.Tensor()

  -- Prototype for preallocated output gradients.
  self.gradOutputProto = torch.Tensor()

  -- Prototype for preallocated context gradient.
  self.gradContextProto = torch.Tensor()
end

--[[ Build a default one time-step of the decoder

Returns: An nn-graph mapping

  $${(c^1_{t-1}, h^1_{t-1}, .., c^L_{t-1}, h^L_{t-1}, x_t, con/H, if) =>
  (c^1_{t}, h^1_{t}, .., c^L_{t}, h^L_{t}, a)}$$

  Where ${c^l}$ and ${h^l}$ are the hidden and cell states at each layer,
  ${x_t}$ is a sparse word to lookup,
  ${con/H}$ is the context/source hidden states for attention,
  ${if}$ is the input feeding, and
  ${a}$ is the context vector computed at this timestep.
--]]
function ConvRecDecoder:_buildModel()
  local inputs = {}
  local states = {}

  -- Inputs are previous layers first.
  for _ = 1, self.args.numEffectiveLayers do
    local h0 = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, h0)
    table.insert(states, h0)
  end

  local x = nn.Identity()() -- batchSize
  table.insert(inputs, x)
  self.args.inputIndex.x = #inputs

  local context = nn.Identity()() -- batchSize x sourceLength x rnnSize
  table.insert(inputs, context)
  self.args.inputIndex.context = #inputs

  local inputFeed
  if self.args.inputFeed then
    inputFeed = nn.Identity()() -- batchSize x rnnSize
    table.insert(inputs, inputFeed)
    self.args.inputIndex.inputFeed = #inputs
  end

  -- Compute the input network.
  local input = self.inputNet(x)

  -- If set, concatenate previous decoder output.
  if self.args.inputFeed then
    input = nn.JoinTable(2)({input, inputFeed})
  end
  table.insert(states, input)

  -- Forward states and input into the RNN.
  local outputs = self.rnn(states)

  -- The output of a subgraph is a node: split it to access the last RNN output.
  outputs = { outputs:split(self.args.numEffectiveLayers) }

  -- Compute the attention here using h^L as query.
  local attnLayer = onmt.GlobalAttention(self.args.rnnSize)
  attnLayer.name = 'decoderAttn'
  local preAttnLayer
  if self.args.doubleOutput then
      preAttnLayer = nn.Narrow(2, 1, self.args.rnnSize)(outputs[#outputs])
  else
      preAttnLayer = outputs[#outputs]
  end
  local attnOutput = attnLayer({preAttnLayer, context})
  if self.rnn.dropout > 0 then
    attnOutput = nn.Dropout(self.rnn.dropout)(attnOutput)
  end
  table.insert(outputs, attnOutput)
  return nn.gModule(inputs, outputs)
end


--[[ Mask padding means that the attention-layer is constrained to
  give zero-weight to padding. This is done by storing a reference
  to the softmax attention-layer.

  Parameters:

  * See  [onmt.MaskedSoftmax](onmt+modules+MaskedSoftmax).
--]]
function ConvRecDecoder:maskPadding(sourceSizes, sourceLength, beamSize)
  if not self.decoderAttn then
    self.network:apply(function (layer)
      if layer.name == 'decoderAttn' then
        self.decoderAttn = layer
      end
    end)
  end

  self.decoderAttn:replace(function(module)
    if module.name == 'softmaxAttn' then
      local mod
      if sourceSizes ~= nil then
        mod = onmt.MaskedSoftmax(sourceSizes, sourceLength, beamSize)
      else
        mod = nn.SoftMax()
      end

      mod.name = 'softmaxAttn'
      mod:type(module._type)
      self.softmaxAttn = mod
      return mod
    else
      return module
    end
  end)
end

function ConvRecDecoder:remember()
    self._remember = true
end

function ConvRecDecoder:forget()
    self._remember = false
end

-- in remember mode still need to reset at beginning of new sequence
function ConvRecDecoder:resetLastStates()
    self.lastStates = nil
end

--[[ Run one step of the decoder.

Parameters:

  * `input` - input to be passed to inputNetwork.
  * `prevStates` - stack of hidden states (batch x layers*model x rnnSize)
  * `context` - encoder output (batch x n x rnnSize)
  * `prevOut` - previous distribution (batch x #words)
  * `t` - current timestep

Returns:

 1. `out` - Top-layer hidden state.
 2. `states` - All states.
--]]
function ConvRecDecoder:forwardOne(input, prevStates, context, prevOut, t)
  local inputs = {}

  -- Create RNN input (see sequencer.lua `buildNetwork('dec')`).
  onmt.utils.Table.append(inputs, prevStates)
  table.insert(inputs, input)
  table.insert(inputs, context)
  local inputSize
  if torch.type(input) == 'table' then
    inputSize = input[1]:size(1)
  else
    inputSize = input:size(1)
  end

  if self.args.inputFeed then
    if prevOut == nil then
      table.insert(inputs, onmt.utils.Tensor.reuseTensor(self.inputFeedProto,
                                                         { inputSize, self.args.rnnSize }))
    else
      table.insert(inputs, prevOut)
    end
  end

  -- Remember inputs for the backward pass.
  if self.train then
    self.inputs[t] = inputs
  end

  local outputs = self:net(t):forward(inputs)
  local out = outputs[#outputs]
  local states = {}
  for i = 1, #outputs - 1 do
    table.insert(states, outputs[i])
  end

  return out, states
end

local function getProtoSizes(batchSize, rnnSize, numEffectiveLayers, doubleOutput, forGradOut)
    local sizes
    if doubleOutput then
        sizes = {}
        for i = 1, numEffectiveLayers/2 -1 do
             table.insert(sizes, {batchSize, rnnSize})
             table.insert(sizes, {batchSize, rnnSize})
        end
        table.insert(sizes, {batchSize, 2*rnnSize})
        table.insert(sizes, {batchSize, 2*rnnSize})
        if forGradOut then
            table.insert(sizes, {batchSize, rnnSize})
        end
    else
        sizes = {batchSize, rnnSize}
    end
    return sizes
end


--[[Compute all forward steps.

  Parameters:

  * `batch` - `Batch` object
  * `encoderStates` -
  * `context` -
  * `func` - Calls `func(out, t)` each timestep.
--]]

function ConvRecDecoder:forwardAndApply(batch, encoderStates, context, func)
  -- TODO: Make this a private method.

  local laySizes = getProtoSizes(batch.size, self.args.rnnSize,
    self.args.numEffectiveLayers, self.args.doubleOutput)

  if self.statesProto == nil then
    self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                         self.stateProto,
                                                         laySizes)
  end

  local states, prevOut
  if self._remember and self.lastStates then -- N.B. this probably breaks if BPTT window < 2
      prevOut = self.lastStates[#self.lastStates]
      states = {} -- could probably really just pop
      for i = 1, #self.lastStates-1 do
          table.insert(states, self.lastStates[i])
      end
  else
      if self.args.doubleOutput then
          states = onmt.utils.Tensor.copyTensorTableHalf(self.statesProto, encoderStates)
      else
          states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)
      end
  end

  --local states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)

  --local prevOut

  for t = 1, batch.targetLength do
    prevOut, states = self:forwardOne(batch:getTargetInput(t), states, context, prevOut, t)
    func(prevOut, t, states[#states])
  end

  if self._remember then -- save a pointer to the last output
      self.lastStates = self:net(batch.targetLength).output
  end
end

--[[Compute all forward steps.

  Parameters:

  * `batch` - a `Batch` object.
  * `encoderStates` - a batch of initial decoder states (optional) [0]
  * `context` - the context to apply attention to.

  Returns: Table of top hidden state for each timestep.
--]]
function ConvRecDecoder:forward(batch, encoderStates, context)
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })
  if self.train then
    self.inputs = {}
  end

  local outputs = {}

  self:forwardAndApply(batch, encoderStates, context, function (out)
    table.insert(outputs, out)
  end)

  return outputs
end

--[[ Compute the backward update.

Parameters:

  * `batch` - a `Batch` object
  * `outputs` - expected outputs
  * `criterion` - a single target criterion object

  Note: This code runs both the standard backward and criterion forward/backward.
  It returns both the gradInputs and the loss.
  -- ]]
function ConvRecDecoder:backward(batch, outputs, criterion, ctxLen, recCrit)
  local laySizes = getProtoSizes(batch.size, self.args.rnnSize,
      self.args.numEffectiveLayers, self.args.doubleOutput, true)
  if self.gradOutputsProto == nil then
    self.gradOutputsProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers + 1,
                                                              self.gradOutputProto,
                                                              laySizes)
  end

  local ctxLen = ctxLen or batch.sourceLength -- for back compat
  local gradStatesInput = onmt.utils.Tensor.reuseTensorTable(self.gradOutputsProto,
                                                             laySizes)
  local gradContextInput = onmt.utils.Tensor.reuseTensor(self.gradContextProto,
                                                         { batch.size, ctxLen, self.args.rnnSize })

  local loss = 0

  local context = self.inputs[1][self.args.inputIndex.context]

  -- rec loss and gradients
  local recloss = 0
  local moarloss = 0
  local recStepGradOuts
  if batch.targetLength >= 5 then
      self.recViewer:resetSize(batch.size, -1, self.args.rnnSize)
      local recpreds = self.rec:forward(outputs)
      local recOutGradOut, recCtxGradOut, moarOutGradOut
      if self.args.discrec then
          if self.args.discdist > 0 then
                recloss = recCrit:forward(recpreds[1], batch:getSourceTriples())*self.args.rho
                local moarInput = {recpreds[2], recpreds[3], recpreds[4]}
                moarloss = self.parallelDistCrit:forward(moarInput, {}) -- no targets necessary
                recOutGradOut = recCrit:backward(recpreds[1], batch:getSourceTriples())
                moarOutGradOut = self.parallelDistCrit:backward(moarInput, {})
		--for jj = 1, 3 do moarOutGradOut[jj]:neg() end
          else
                recloss = recCrit:forward(recpreds, batch:getSourceTriples())*self.args.rho
                recOutGradOut = recCrit:backward(recpreds, batch:getSourceTriples())
          end
      else
          recloss = recCrit:forward(recpreds, context)*self.args.rho
          recOutGradOut, recCtxGradOut = recCrit:backward(recpreds, context)
          -- add encoder grads
          gradContextInput:add(self.args.rho/batch.totalSize, recCtxGradOut)
      end
      if self.args.discrec and self.args.discdist > 0 then
          recStepGradOuts = self.rec:backward(outputs, {recOutGradOut, moarOutGradOut[1], moarOutGradOut[2], moarOutGradOut[3]})
      else
          recStepGradOuts = self.rec:backward(outputs, recOutGradOut)
      end
  end


  for t = batch.targetLength, 1, -1 do
    -- Compute decoder output gradients.
    -- Note: This would typically be in the forward pass.
    --local pred = self.generator:forward(outputs[t])
    local genInp = {outputs[t], context, self:net(t).output[self.args.numEffectiveLayers], batch:getSourceWords()}
    local pred = self.generator:forward(genInp)
    local output = batch:getTargetOutput(t)

    loss = loss + criterion:forward(pred, output)

    -- Compute the criterion gradient.
    local genGradOut = criterion:backward(pred, output)
    for j = 1, #genGradOut do
      genGradOut[j]:div(batch.totalSize)
    end

    -- Compute the final layer gradient.
    local decGradOut = self.generator:backward(genInp, genGradOut)
    --gradStatesInput[#gradStatesInput]:add(decGradOut)
    gradStatesInput[#gradStatesInput]:add(decGradOut[1])
    if recStepGradOuts then
        gradStatesInput[#gradStatesInput]:add(self.args.rho/batch.totalSize, recStepGradOuts[t])
    end
    gradContextInput:add(decGradOut[2])
    --if self.args.doubleOutput then
    gradStatesInput[self.args.numEffectiveLayers]:add(decGradOut[3])
    --end

    -- Compute the standard backward.
    local gradInput = self:net(t):backward(self.inputs[t], gradStatesInput)

    -- Accumulate encoder output gradients.
    gradContextInput:add(gradInput[self.args.inputIndex.context])
    gradStatesInput[#gradStatesInput]:zero()

    -- Accumulate previous output gradients with input feeding gradients.
    if self.args.inputFeed and t > 1 then
      gradStatesInput[#gradStatesInput]:add(gradInput[self.args.inputIndex.inputFeed])
    end

    -- Prepare next decoder output gradients.
    for i = 1, #self.statesProto do
      gradStatesInput[i]:copy(gradInput[i])
    end
  end

  -- if self.args.doubleOutput then
  --     gradStatesInput[self.args.numEffectiveLayers] = gradStatesInput[self.args.numEffectiveLayers]:narrow(2, 1, self.args.rnnSize)
  --     gradStatesInput[self.args.numEffectiveLayers-1] = gradStatesInput[self.args.numEffectiveLayers-1]:narrow(2, 1, self.args.rnnSize)
  -- end
  local finalLayer, rnnSize = self.args.numEffectiveLayers, self.args.rnnSize
  if self.args.doubleOutput then
      gradStatesInput[finalLayer]:narrow(2,1,rnnSize):add(gradStatesInput[finalLayer]:narrow(2,rnnSize+1,rnnSize))
      gradStatesInput[finalLayer] = gradStatesInput[finalLayer]:narrow(2,1,rnnSize)
      gradStatesInput[finalLayer-1]:narrow(2,1,rnnSize):add(gradStatesInput[finalLayer-1]:narrow(2,rnnSize+1,rnnSize))
      gradStatesInput[finalLayer-1] = gradStatesInput[finalLayer-1]:narrow(2,1,rnnSize)
  end


  if batch.targetOffset > 0 then -- this is a hack, but the pt is that only used encoder's last state on first piece
      for i = 1, #self.statesProto do
          gradStatesInput[i]:zero()
      end
  end
  return gradStatesInput, gradContextInput, loss, recloss
end

--[[ Compute the loss on a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.
  * `criterion` - a pointwise criterion.

--]]
function ConvRecDecoder:computeLoss(batch, encoderStates, context, criterion, recCrit)
  -- don't do this unless the whole seq is like less than the window size
  local encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local loss = 0
  local outputs = {}

  self:forwardAndApply(batch, encoderStates, context, function (out, t, finalState)
    --print(torch.abs(out):sum())
    table.insert(outputs, out)
    local genInp = {out, context, finalState, batch:getSourceWords()}
    local pred = self.generator:forward(genInp)
    local output = batch:getTargetOutput(t)
    loss = loss + criterion:forward(pred, output)
    --print("loss", loss)
  end)

  -- self.recViewer:resetSize(batch.size, -1, self.args.rnnSize)
  -- local recpreds = self.rec:forward(outputs)
  -- local recOutGradOut, recCtxGradOut
  -- assert(self.args.discrec)
  -- local recloss = recCrit:forward(recpreds, batch:getSourceTriples())*self.args.rho

  return loss --+ recloss
end


--[[ Compute the score of a batch.

Parameters:

  * `batch` - a `Batch` to score.
  * `encoderStates` - initialization of decoder.
  * `context` - the attention context.

--]]
function ConvRecDecoder:computeScore(batch, encoderStates, context)
  assert(false)
  encoderStates = encoderStates
    or onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                         onmt.utils.Cuda.convert(torch.Tensor()),
                                         { batch.size, self.args.rnnSize })

  local score = {}

  self:forwardAndApply(batch, encoderStates, context, function (out, t, finalState)
    --local genInp = {out, context, batch:getSourceWords()}
    local genInp = {out, context, finalState, batch:getSourceWords()}
    local pred = self.generator:forward(genInp)
    for b = 1, batch.size do
      if t <= batch.targetSize[b] then
        score[b] = (score[b] or 0) + pred[1][b][batch.targetOutput[t][b]]
      end
    end
  end)

  return score
end

function ConvRecDecoder:greedyFixedFwd(batch, encoderStates, context, probBuf)
    if not self.greedy_inp then
        self.greedy_inp = torch.CudaTensor()
        self.maxes = torch.CudaTensor()
        self.argmaxes = torch.CudaLongTensor()
    end
    local PAD, EOS = onmt.Constants.PAD, onmt.Constants.EOS
    self.greedy_inp:resize(batch.targetLength+1, batch.size):fill(PAD)
    self.maxes:resize(batch.size, 1)
    self.argmaxes:resize(batch.size, 1)

    local laySizes = getProtoSizes(batch.size, self.args.rnnSize,
      self.args.numEffectiveLayers, self.args.doubleOutput)

    if self.statesProto == nil then
      self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                           self.stateProto,
                                                           laySizes)
    end

    local states, prevOut
    if self.args.doubleOutput then
        states = onmt.utils.Tensor.copyTensorTableHalf(self.statesProto, encoderStates)
    else
        states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)
    end

    self.greedy_inp[1]:copy(batch:getTargetInput(1)) -- should be start token
    for t = 1, batch.targetLength do
      prevOut, states = self:forwardOne(self.greedy_inp[t], states, context, prevOut, t)
      --local genInp = {prevOut, context, batch:getSourceWords()}
      local genInp = {prevOut, context, states[#states], batch:getSourceWords()}
      local preds = self.generator:forward(genInp)[1]

      torch.max(self.maxes, self.argmaxes, preds, 2)
      if probBuf then
          probBuf[t]:copy(self.maxes:view(-1))
      end
      self.greedy_inp[t+1]:copy(self.argmaxes:view(-1))
    end
    return self.greedy_inp
end

function ConvRecDecoder:greedyFixedFwd2(batch, encoderStates, context)
    if not self.greedy_inp then
        self.greedy_inp = torch.CudaTensor()
        self.maxes = torch.CudaTensor()
        self.argmaxes = torch.CudaLongTensor()
    end
    local PAD, EOS = onmt.Constants.PAD, onmt.Constants.EOS
    self.greedy_inp:resize(batch.targetLength+1, batch.size):fill(PAD)
    self.maxes:resize(batch.size, 1)
    self.argmaxes:resize(batch.size, 1)

    local laySizes = getProtoSizes(batch.size, self.args.rnnSize,
      self.args.numEffectiveLayers, self.args.doubleOutput)

    if self.statesProto == nil then
      self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                           self.stateProto,
                                                           laySizes)
    end

    local stuff = {}

    local states, prevOut
    if self.args.doubleOutput then
        states = onmt.utils.Tensor.copyTensorTableHalf(self.statesProto, encoderStates)
    else
        states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)
    end

    local smlayer
    self.generator.modules[1]:apply(function(mod)
        if torch.type(mod) == 'nn.SoftMax' then
            smlayer = mod
        end
    end)

    self.greedy_inp[1]:copy(batch:getTargetInput(1)) -- should be start token
    for t = 1, batch.targetLength do
      stuff[t] = {}
      prevOut, states = self:forwardOne(self.greedy_inp[t], states, context, prevOut, t)
      --local genInp = {prevOut, context, batch:getSourceWords()}
      local genInp = {prevOut, context, states[#states], batch:getSourceWords()}
      local preds = self.generator:forward(genInp)[1]


      torch.max(self.maxes, self.argmaxes, preds, 2)
      for n = 1, batch.size do
          stuff[t][n] = {}
          local argmax = self.argmaxes[n][1]
          table.insert(stuff[t][n], smlayer.output[n][argmax])
          for j = 1, genInp[4]:size(2) do
              if genInp[4][n][j] == argmax then
                  table.insert(stuff[t][n], smlayer.output[n][self.generator.outputSize+j])
              end
          end
      end

      self.greedy_inp[t+1]:copy(self.argmaxes:view(-1))
    end
    return self.greedy_inp, stuff
end

function ConvRecDecoder:greedyFixedFwd3(batch, encoderStates, context)
    if not self.greedy_inp then
        self.greedy_inp = torch.CudaTensor()
        self.maxes = torch.CudaTensor()
        self.argmaxes = torch.CudaLongTensor()
    end
    local PAD, EOS = onmt.Constants.PAD, onmt.Constants.EOS
    self.greedy_inp:resize(batch.targetLength+1, batch.size):fill(PAD)
    self.maxes:resize(batch.size, 1)
    self.argmaxes:resize(batch.size, 1)

    local laySizes = getProtoSizes(batch.size, self.args.rnnSize,
      self.args.numEffectiveLayers, self.args.doubleOutput)

    if self.statesProto == nil then
      self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                           self.stateProto,
                                                           laySizes)
    end

    local stuff = {}

    local states, prevOut
    if self.args.doubleOutput then
        states = onmt.utils.Tensor.copyTensorTableHalf(self.statesProto, encoderStates)
    else
        states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)
    end

    local attnlayer
    self.generator.modules[1]:apply(function(mod)
        if torch.type(mod) == 'nn.Sum' then
            attnlayer = mod
        end
    end)

    self.greedy_inp[1]:copy(batch:getTargetInput(1)) -- should be start token
    for t = 1, batch.targetLength do
      stuff[t] = {}
      prevOut, states = self:forwardOne(self.greedy_inp[t], states, context, prevOut, t)
      --local genInp = {prevOut, context, batch:getSourceWords()}
      local genInp = {prevOut, context, states[#states], batch:getSourceWords()}
      local preds = self.generator:forward(genInp)[1]

      torch.max(self.maxes, self.argmaxes, preds, 2)
      local sortedAttn, sortedIdxs = torch.sort(attnlayer.output:float(), 2, true)
      for n = 1, batch.size do
          stuff[t][n] = {}
          for k = 1, 5 do
              -- fmt is word,idx,score
              local idx = sortedIdxs[n][k]
              table.insert(stuff[t][n], {genInp[4][n][idx], idx, sortedAttn[n][k]})
          end
      end

      self.greedy_inp[t+1]:copy(self.argmaxes:view(-1))
    end
    return self.greedy_inp, stuff
end

-- assumes seqs starts w/ start_token
function ConvRecDecoder:scoreSequences(batch, encoderStates, context, seqs)
    local laySizes = getProtoSizes(batch.size, self.args.rnnSize,
      self.args.numEffectiveLayers, self.args.doubleOutput)

    if self.statesProto == nil then
      self.statesProto = onmt.utils.Tensor.initTensorTable(self.args.numEffectiveLayers,
                                                           self.stateProto,
                                                           laySizes)
    end

    local scores = torch.Tensor():resize(seqs:size(1)-1, seqs:size(2), 3):zero()

    local states, prevOut
    if self.args.doubleOutput then
        states = onmt.utils.Tensor.copyTensorTableHalf(self.statesProto, encoderStates)
    else
        states = onmt.utils.Tensor.copyTensorTable(self.statesProto, encoderStates)
    end

    local smlayer
    self.generator.modules[1]:apply(function(mod)
        if torch.type(mod) == 'nn.SoftMax' then
            smlayer = mod
        end
    end)
    assert(smlayer)

    for t = 1, seqs:size(1)-1 do
      prevOut, states = self:forwardOne(seqs[t], states, context, prevOut, t)
      local genInp = {prevOut, context, states[#states], batch:getSourceWords()}
      local preds = self.generator:forward(genInp)[1]
      for n = 1, batch.size do
          local next_word = seqs[t+1][n]
          scores[t][n][1] = preds[n][next_word]
          scores[t][n][2] = smlayer.output[n][next_word]
          -- for now i'll add the pointer probs...
          for j = 1, genInp[4]:size(2) do
              if genInp[4][n][j] == next_word then
                  scores[t][n][3] = scores[t][n][3] + smlayer.output[n][self.generator.outputSize+j]
              end
          end
      end
    end
    return scores
end
