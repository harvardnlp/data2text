--[[
 This takes ctx, and topstate and produces a log distribution over source.
--]]
local PointerGenerator, parent = torch.class('onmt.PointerGenerator', 'nn.Container')


function PointerGenerator:__init(rnnSize, tanhQuery, doubleOutput, multilabel)
  parent.__init(self)
  self.net = self:_buildGenerator(rnnSize, tanhQuery, doubleOutput, multilabel)
  self:add(self.net)
end

function PointerGenerator:_buildGenerator(rnnSize, tanhQuery, doubleOutput, multilabel)
    local context = nn.Identity()()
    local pstate = nn.Identity()()

    -- get unnormalized attn scores
    local qstate = doubleOutput and nn.Narrow(2, rnnSize+1, rnnSize)(pstate) or pstate
    local targetT = nn.Linear(rnnSize, rnnSize)(qstate)
    if tanhQuery then
        targetT = nn.Tanh()(targetT)
    end
    local attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
    attn = nn.Sum(3)(attn) -- batchL x sourceL
    local output = multilabel and nn.SoftMax()(attn) or nn.LogSoftMax()(attn)
    local inputs = {context, pstate}
    return nn.gModule(inputs, {output})
end

function PointerGenerator:updateOutput(input)
  --self.output = {self.net:updateOutput(input)}
  self.output = self.net:updateOutput(input)
  return self.output
end

function PointerGenerator:updateGradInput(input, gradOutput)
  --self.gradInput = self.net:updateGradInput(input, gradOutput[1])
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function PointerGenerator:accGradParameters(input, gradOutput, scale)
  --self.net:accGradParameters(input, gradOutput[1], scale)
  self.net:accGradParameters(input, gradOutput, scale)
end
