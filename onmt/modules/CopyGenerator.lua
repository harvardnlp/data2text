--[[Simple CopyGenerator. Given RNN state and (unnormalized) attn scores produce categorical distribution.

--]]
local CopyGenerator, parent = torch.class('onmt.CopyGenerator', 'nn.Container')


function CopyGenerator:__init(rnnSize, outputSize)
  parent.__init(self)
  self.net = self:_buildGenerator(rnnSize, outputSize)
  self:add(self.net)
end

function CopyGenerator:_buildGenerator(rnnSize, outputSize)
  return nn.Sequential()
           :add(nn.ParallelTable()
              :add(nn.Linear(rnnSize, outputSize))
              :add(nn.Identity()))
           :add(nn.JoinTable(2))
           :add(nn.SoftMax())
end

function CopyGenerator:updateOutput(input)
  self.output = {self.net:updateOutput(input)}
  return self.output
end

function CopyGenerator:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput[1])
  return self.gradInput
end

function CopyGenerator:accGradParameters(input, gradOutput, scale)
  self.net:accGradParameters(input, gradOutput[1], scale)
end
