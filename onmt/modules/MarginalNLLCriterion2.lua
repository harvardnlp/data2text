-- require 'nn'
-- onmt = {}

local MarginalNLLCriterion, parent = torch.class('onmt.MarginalNLLCriterion', 'nn.Criterion')

function MarginalNLLCriterion:__init(ignoreIdx)
   parent.__init(self)
   self.sizeAverage = true
end

--[[ This will output the negative log marginal, even though we'll ignore the log when doing gradients

Parameters:

  * `input` - an NxV tensor of probabilities.
  * `target` - a mask with 0s for probabilities to be ignored and positive numbers for probabilities to be added

--]]
function MarginalNLLCriterion:updateOutput(input, target)
    if not self.buf then
        self.buf = torch.Tensor():typeAs(input)
        self.rowSums = torch.Tensor():typeAs(input)
        self.gradInput:typeAs(input)
    end
    self.buf:resizeAs(input)
    self.buf:cmul(input, target)
    self.rowSums:resize(input:size(1), 1)
    self.rowSums:sum(self.buf, 2) -- will store for backward
    -- set rowSums = 0 to 1 since we're gonna log; dunno if there's a faster way
    for i = 1, input:size(1) do
        if self.rowSums[i][1] <= 0 then
            self.rowSums[i][1] = 1
        end
    end
    -- use buf
    local logRowSums = self.buf:narrow(2, 1, 1)
    logRowSums:log(self.rowSums)
    self.output = -logRowSums:sum()
    if self.sizeAverage then
        self.output = self.output/input:size(1)
    end

    return self.output
end


function MarginalNLLCriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input)
    self.gradInput:copy(target)
    self.rowSums:neg()
    self.gradInput:cdiv(self.rowSums:expand(input:size(1), input:size(2)))
    if self.sizeAverage then
        self.gradInput:div(input:size(1))
    end
    return self.gradInput
end


-- local mine = true
--
-- torch.manualSeed(2)
-- local mlp = nn.Sequential()
--          :add(nn.Linear(4,5))
-- if mine then
--      mlp:add(nn.SoftMax())
-- else
--     mlp:add(nn.LogSoftMax())
-- end
--
-- local crit
-- if mine then
--     crit = onmt.MarginalNLLCriterion()
-- else
--     crit = nn.ClassNLLCriterion(torch.Tensor({0,1,1,1,1}))
-- end
-- --crit.sizeAverage = false
--
-- local X = torch.randn(3, 4)
-- -- local T = torch.LongTensor({{2, 3},
-- --                             {4, 1},
-- --                             {1, 1}})
--
--
-- local T = torch.LongTensor({{2, 3, 2},
--                             {4, 1, 1},
--                             {1, 1, 0}})--:view(-1)
--
-- local maskT = torch.Tensor({{0,1,1,0,0},
--                             {0,0,0,1,0},
--                             {0,0,0,0,0}})
--
-- if not mine then
--     assert(false)
--     T = T:select(2, 1)
-- end
--
-- local nugtarg = maskT --T
--
--
-- mlp:zeroGradParameters()
-- mlp:forward(X)
-- print("loss", crit:forward(mlp.output, nugtarg))
-- local gradOut = crit:backward(mlp.output, nugtarg)
-- print("gradOut", gradOut)
-- mlp:backward(X, gradOut)
--
-- local eps = 1e-5
--
-- local function getLoss()
--     mlp:forward(X)
--     return crit:forward(mlp.output, nugtarg)
-- end
--
-- local W = mlp:get(1).weight
-- for i = 1, W:size(1) do
--     for j = 1, W:size(2) do
--         W[i][j] = W[i][j] + eps
--         local rloss = getLoss()
--         W[i][j] = W[i][j] - 2*eps
--         local lloss = getLoss()
--         local fd = (rloss - lloss)/(2*eps)
--         print(mlp:get(1).gradWeight[i][j], fd)
--         W[i][j] = W[i][j] + eps
--     end
--     print("")
-- end
