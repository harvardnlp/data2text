-- require 'nn'
-- onmt = {}

local MarginalNLLCriterion, parent = torch.class('nn.MarginalNLLCriterion', 'nn.Criterion')

function MarginalNLLCriterion:__init()
   parent.__init(self)
   self.sizeAverage = true
end


--[[

Parameters:

  * `input` - an NxV tensor of probabilities.
  * `target` - an Nx(numNZ+1) tensor, where last column says how many nonzero indices there are
--]]

function MarginalNLLCriterion:updateOutput(input, target)
    if not self.buf then
        self.buf = torch.Tensor():typeAs(input)
        self.rowSums = torch.Tensor():typeAs(input)
        self.gradInput:typeAs(input)
    end

    local maxIndices = target:size(2)-1
    self.buf:resize(target:size(1), maxIndices):zero()
    self.buf:select(2, 1):fill(1) -- if we ignore a row it will sum to 1, so no loss
    self.rowSums:resize(input:size(1), 1)

    -- could do this w/o looping, but would require a lot of extra arithmetic
    -- that might not end up being much more efficient
    for i = 1, target:size(1) do
        local nnz_i = target[i][maxIndices+1]
        if nnz_i > 0 then
            self.buf[i]:sub(1, nnz_i)
              :index(input[i], 1, target[i]:sub(1, nnz_i))
        end
    end

    self.rowSums:sum(self.buf, 2)

    local logRowSums = self.buf:narrow(2, 1, 1)
    logRowSums:log(self.rowSums)
    self.output = -logRowSums:sum()
    if self.sizeAverage then
        self.output = self.output/input:size(1)
    end

    return self.output
end

function MarginalNLLCriterion:updateGradInput(input, target)
    self.gradInput:resizeAs(input):zero()

    if self.sizeAverage then
        self.rowSums:mul(input:size(1))
    end

    local maxIndices = target:size(2)-1
    for i = 1, target:size(1) do
        local nnz_i = target[i][maxIndices+1]
        if nnz_i > 0 then
            self.gradInput[i]:indexFill(1, target[i]:sub(1, nnz_i), 1)
        end
    end

    -- faster than doing the arithmetic up there for some reason
    self.rowSums:neg()
    self.gradInput:cdiv(self.rowSums:expand(input:size(1), input:size(2)))

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
--     crit = onmt.MarginalNLLCriterion(1)
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
-- if not mine then
--     T = T:select(2, 1)
-- end
--
-- local nugtarg = T
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
