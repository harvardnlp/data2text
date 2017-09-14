require 'nn'

local KMinDist, parent = torch.class('nn.KMinDist', 'nn.Criterion')

-- will square the distance for p=2, tho maybe we shouldn't...
function KMinDist:__init(p, maxBatchSize, maxK)
   parent.__init(self)
   self.sizeAverage = true
   self.p = p or 2
   assert(self.p == 1 or self.p == 2)
   local maxBatchSize = maxBatchSize or 1024
   local maxK = maxK or 3
   self.range = torch.range(0, maxBatchSize*maxK-1)
end

-- input is batchsize x K*dim; target is batchsize x M x dim
-- loss: \sum_k min_m dist(input_k, target_m)
function KMinDist:updateOutput(input, target)
    local bsz, dim, M, K = input:size(1), target:size(3), target:size(2), input:size(2)/target:size(3)
    self.diff = self.diff or input.new()
    self.sums = self.sums or input.new()
    self.mins = self.mins or input.new()
    if not self.argmins then
        self.argmins = torch.type(self.mins) == "torch.CudaTensor"
            and torch.CudaLongTensor() or torch.LongTensor()
    end
    self.diff:resize(bsz, K, M, dim)
    self.sums:resize(bsz, K, M, 1)
    self.mins:resize(bsz, K, 1)
    self.argmins:resize(bsz, K, 1)

    local diff, sums = self.diff, self.sums
    diff:add(input:view(bsz, K, 1, dim):expand(bsz, K, M, dim),
          -1, target:view(bsz, 1, M, dim):expand(bsz, K, M, dim))
    if self.p == 1 then
        diff:abs()
    else -- p == 2
        diff:pow(2)
    end
    sums:sum(diff, 4) -- bsz x K x M
    -- if self.p == 2 then
    --     sums:sqrt()
    -- end
    torch.min(self.mins, self.argmins, sums:squeeze(4), 3)
    self.output = self.mins:sum()

    if self.p == 2 then
        self.output = self.output/2
    end

    if self.sizeAverage then
        self.output = self.output/bsz
    end

    return self.output
end

-- returns 2 things, to be compatible w/ usual criteria
function KMinDist:updateGradInput(input, target)
    local bsz, dim, M, K = input:size(1), target:size(3), target:size(2), input:size(2)/target:size(3)
    self.gradTarget = self.gradTarget or target.new()
    self.gradInput:resizeAs(input)
    self.gradTarget:resizeAs(target):zero()

    self.diff:resize(bsz, K, M, dim)
    local diff = self.diff
    -- could really save this from fwd pass if we double the memory
    diff:add(input:view(bsz, K, 1, dim):expand(bsz, K, M, dim),
          -1, target:view(bsz, 1, M, dim):expand(bsz, K, M, dim))

    -- recalculate argmins so we can index into a 2d tensor
    self.newIdxs = self.newIdxs or self.argmins.new()
    local newIdxs = self.newIdxs
    newIdxs:resize(bsz*K):copy(self.range:sub(1, bsz*K)):mul(M)
    newIdxs:add(self.argmins:view(-1))
    self.gradInput:view(-1, dim):index(diff:view(-1, dim), 1, newIdxs) -- holds (input_k - target_m)

    if self.p == 1 then
        self.gradInput:sign()
    end

    -- the diffs in gradInput now need to be distributed into gradTarget
    --print(self.argmins)
    newIdxs:sub(1, bsz):copy(self.range:sub(1, bsz))
    self.argmins:view(bsz, K):add(M, newIdxs:sub(1, bsz):view(bsz, 1):expand(bsz, K))
    --print(self.argmins)
    self.gradTarget:view(-1, dim):indexAdd(1, self.argmins:view(-1), self.gradInput:view(-1, dim))
    self.gradTarget:neg()

    if self.sizeAverage then
        self.gradInput:div(bsz)
        self.gradTarget:div(bsz)
    end

    return self.gradInput, self.gradTarget

end


-- torch.manualSeed(2)
-- local M = 5
-- local dim = 5
-- local K = 3
--
-- crit = nn.KMinDist(2)
-- --crit = nn.KMinDist(1)
--
-- X = torch.randn(2, K*dim)
--
-- Y = torch.randn(2, M, dim)
--
--
-- crit:forward(X, Y)
-- gradIn, gradTarg = crit:backward(X, Y)
-- gradIn = gradIn:clone()
-- gradTarg = gradTarg:clone()
--
-- local eps = 1e-5
--
--
-- local function getLoss()
--     return crit:forward(X, Y)
-- end
--
-- print("X")
-- for i = 1, X:size(1) do
--     for j = 1, X:size(2) do
--         X[i][j] = X[i][j] + eps
--         local rloss = getLoss()
--         X[i][j] = X[i][j] - 2*eps
--         local lloss = getLoss()
--         local fd = (rloss - lloss)/(2*eps)
--         print(gradIn[i][j], fd)
--         X[i][j] = X[i][j] + eps
--     end
--     print("")
-- end
--
-- print("")
-- print("Y")
-- rY = Y:view(-1, dim)
-- for i = 1, rY:size(1) do
--     for j = 1, rY:size(2) do
--         rY[i][j] = rY[i][j] + eps
--         local rloss = getLoss()
--         rY[i][j] = rY[i][j] - 2*eps
--         local lloss = getLoss()
--         local fd = (rloss - lloss)/(2*eps)
--         print(gradTarg:view(-1, dim)[i][j], fd)
--         rY[i][j] = rY[i][j] + eps
--     end
--     print("")
-- end
