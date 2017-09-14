require 'nn'

local PairwiseDistDist, parent = torch.class('nn.PairwiseDistDist', 'nn.Criterion')

function PairwiseDistDist:__init(hellinger)
   parent.__init(self)
   self.sizeAverage = false
   self.hellinger = hellinger
    -- just assuming 3 pairs
   if self.hellinger then
       self.crits = {nn.MSECriterion(), nn.MSECriterion(), nn.MSECriterion()}
       self.sqrt = nn.Sqrt()
       self.sqrtGradOut = torch.Tensor()
   else
       self.crits = {nn.AbsCriterion(), nn.AbsCriterion(), nn.AbsCriterion()}
   end
   for i = 1, #self.crits do
       self.crits[i].sizeAverage = self.sizeAverage
   end
end

-- input assumed to be batch x 3 x V, and already softmaxed
function PairwiseDistDist:updateOutput(input)
    local predInput = self.hellinger and self.sqrt:forward(input) or input
    local preds1 = predInput:select(2, 1)
    local preds2 = predInput:select(2, 2)
    local preds3 = predInput:select(2, 3)
    -- do the pairs
    local loss1 = self.crits[1]:forward(preds1, preds2)
    local loss2 = self.crits[2]:forward(preds1, preds3)
    local loss3 = self.crits[3]:forward(preds2, preds3)

    self.output = loss1 + loss2 + loss3
    return self.output
end



function PairwiseDistDist:updateGradInput(input)
    local gradInput = self.hellinger and self.sqrtGradOut or self.gradInput
    gradInput:resizeAs(input):zero()

    local predInput = self.hellinger and self.sqrt.output or input
    local preds1 = predInput:select(2, 1)
    local preds2 = predInput:select(2, 2)
    local preds3 = predInput:select(2, 3)

    local gradIn1 = self.crits[1]:backward(preds1, preds2)
    gradInput:select(2, 1):add(gradIn1)
    gradInput:select(2, 2):add(-1, gradIn1)

    local gradIn2 = self.crits[2]:backward(preds1, preds3)
    gradInput:select(2, 1):add(gradIn2)
    gradInput:select(2, 3):add(-1, gradIn2)

    local gradIn3 = self.crits[3]:backward(preds2, preds3)
    gradInput:select(2, 2):add(gradIn3)
    gradInput:select(2, 3):add(-1, gradIn3)

    if self.hellinger then
        self.gradInput = self.sqrt:backward(input, gradInput)
    end

    -- sometimes we get nans
    self.gradInput[self.gradInput:ne(self.gradInput)] = 0

    return self.gradInput
end


-- torch.manualSeed(2)
--
-- crit = nn.PairwiseDistDist(false)
-- --crit = nn.PairwiseDistDist(true)
-- local sm = nn.SoftMax()
-- X = sm:forward(torch.randn(2, 3, 4))
--
-- crit:forward(X)
-- gradIn = crit:backward(X)
-- gradIn = gradIn:clone()
--
-- local eps = 1e-5
--
--
-- local function getLoss()
--     return crit:forward(X)
-- end
--
-- print("X")
-- Xflat = X:view(-1)
--
-- for i = 1, Xflat:size(1) do
--     Xflat[i] = Xflat[i] + eps
--     local rloss = getLoss()
--     Xflat[i] = Xflat[i] - 2*eps
--     local lloss = getLoss()
--     local fd = (rloss - lloss)/(2*eps)
--     print(gradIn:view(-1)[i], fd)
--     Xflat[i] = Xflat[i] + eps
-- end
