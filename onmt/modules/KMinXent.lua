require 'nn'

local KMinXent, parent = torch.class('nn.KMinXent', 'nn.Criterion')

function KMinXent:__init()
   parent.__init(self)
   self.sizeAverage = true
   self.net = nn.Sequential()
                :add(nn.MM(false, true)) -- batchSize x numPreds x M
                :add(nn.Max(3))          -- batchSize x numPreds
                -- check if View(-1) is faster...
                :add(nn.Sum(2))          -- batchSize; doesn't seem like we can sum over everything at once
                :add(nn.Sum())           -- 1
                :add(nn.MulConstant(-1)) -- 1
   self.netGradOut = torch.ones(1) -- could rid of MulConstant and just make this negative
end

-- input is batchSize x numPreds x sum[outVocabSizes], where each dist is log normalized.
-- target is binary batchsize x M x sum[outVocabSizes], where target[b][m] is concatenation of 1 hot vectors.
-- loss: - sum_k max_m \sum_j ln q^(j)(m_j)  = sum_k min_m \sum_j xent(q^(j), m_j)
function KMinXent:updateOutput(input, target)
    if self.sizeAverage then
        self.net:get(5).constant_scalar = -1/input:size(1)
    else
        self.net:get(5).constant_scalar = -1
    end
    self.output = self.net:forward({input, target})[1]
    return self.output
end


function KMinXent:updateGradInput(input, target)
    self.net:backward({input, target}, self.netGradOut)
    self.gradInput = self.net.gradInput[1]
    return self.gradInput
end
