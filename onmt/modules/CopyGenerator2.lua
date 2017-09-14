--[[
 This takes ctx, gets unnormalized attn, gets regular unnormalized linear word-scores,
 and then softmaxes the whole thing. Thus we get p(word=w,copy=z) for each word w and z \in {0, 1}.
 This is appropriate for a criterion that will then marginalize over z.
--]]
local CopyGenerator2, parent = torch.class('onmt.CopyGenerator2', 'nn.Container')


function CopyGenerator2:__init(rnnSize, outputSize, tanhQuery, doubleOutput)
  parent.__init(self)
  self.net = self:_buildGenerator(rnnSize, outputSize, tanhQuery, doubleOutput)
  self:add(self.net)
  self.outputSize = outputSize
end

function CopyGenerator2:_buildGenerator(rnnSize, outputSize, tanhQuery, doubleOutput)
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

    -- concatenate with regular output shit
    local regularOutput = nn.Linear(rnnSize, outputSize)(tstate)
    local catDist = nn.SoftMax()(nn.JoinTable(2)({regularOutput, attn}))
    local rulDist = nn.Narrow(2,1,outputSize)(catDist)
    local ptrDist = nn.Narrow(2,outputSize+1,-1)(catDist)
    local logmarginals = nn.Log()(nn.CIndexAddTo()({rulDist, ptrDist, srcIdxs}))
    local inputs = {tstate, context, pstate, srcIdxs}
    return nn.gModule(inputs, {logmarginals})
end

function CopyGenerator2:updateOutput(input)
  self.output = {self.net:updateOutput(input)}
  return self.output
end

function CopyGenerator2:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput[1])
  return self.gradInput
end

function CopyGenerator2:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput[1], scale)
end
