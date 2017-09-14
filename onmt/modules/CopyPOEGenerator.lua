--[[
 This takes ctx, gets unnormalized attn, and adds those scores to unnormalized
 word scores, and then logsoftmaxes. This is a product of experts model, so either
 attn or output can veto.
 A regular ClassNLLCriterion should be used.
--]]
local CopyPOEGenerator, parent = torch.class('onmt.CopyPOEGenerator', 'nn.Container')


function CopyPOEGenerator:__init(rnnSize, outputSize, tanhQuery, doubleOutput)
  parent.__init(self)
  self.net = self:_buildGenerator(rnnSize, outputSize, tanhQuery, doubleOutput)
  self:add(self.net)
  self.outputSize = outputSize
end

-- N.B. this uses attnLayer, but should maybe use last real layer (in which case we need 3 inputs)
function CopyPOEGenerator:_buildGenerator(rnnSize, outputSize, tanhQuery, doubleOutput)
    local tstate = nn.Identity()() -- attnlayer (numEffectiveLayers+1)
    local context = nn.Identity()()
    local pstate = nn.Identity()()
    local srcIdxs = nn.Identity()()

    -- get unnormalized attn scores
    local qstate = doubleOutput and nn.Narrow(2, rnnSize+1, rnnSize)(pstate) or pstate
    local targetT = nn.Linear(rnnSize, rnnSize)(qstate)
    if tanhQuery then
        targetT = nn.Tanh()(targetT)
    end
    local attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
    attn = nn.Sum(3)(attn) -- batchL x sourceL

    -- add scores to regular output shit
    local regularOutput = nn.Linear(rnnSize, outputSize)(tstate)
    local addedOutput = nn.CIndexAddTo()({regularOutput, attn, srcIdxs})
    local scores = nn.LogSoftMax()(addedOutput)
    local inputs = {tstate, context, pstate, srcIdxs}
    return nn.gModule(inputs, {scores})

end

function CopyPOEGenerator:updateOutput(input)
  self.output = {self.net:updateOutput(input)}
  return self.output
end

function CopyPOEGenerator:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput[1])
  return self.gradInput
end

function CopyPOEGenerator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput[1], scale)
end
