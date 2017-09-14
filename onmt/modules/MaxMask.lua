--require 'nn'

local MaxMask, parent = torch.class('nn.MaxMask', 'nn.Module')

function MaxMask:__init()
    parent.__init(self)
end

function MaxMask:updateOutput(input)
    if not self.maxes then
        if torch.type(input) == 'torch.CudaTensor' then
            self.maxes = torch.CudaTensor()
            self.argmaxes = torch.CudaLongTensor()
        else
            self.maxes = torch.Tensor()
            self.argmaxes = torch.LongTensor()
        end
    end
    self.maxes:resize(input:size(1), 1)
    self.argmaxes:resize(input:size(1), 1)
    torch.max(self.maxes, self.argmaxes, input, 2)
    self.output:resizeAs(input):zero()
    self.output:scatter(2, self.argmaxes, self.maxes)
    return self.output
end

function MaxMask:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(input):zero()
    self.gradInput:scatter(2, self.argmaxes, 1)
    self.gradInput:cmul(gradOutput)
    return self.gradInput
end

--
-- mlp = nn.Sequential()
--               :add(nn.Linear(5,6))
--               :add(nn.MaxMask())
--               :add(nn.CMul(6))
--               :add(nn.Sum(2))
--
--
--
-- myx = torch.randn(2, 5)
-- myy = torch.randn(2,1)
-- crit = nn.MSECriterion()
--
-- feval = function(x)
--     return crit:forward(mlp:forward(x), myy)
-- end
--
-- crit:forward(mlp:forward(myx), myy)
-- dpdc = crit:backward(mlp.output, myy)
-- mlp:backward(myx, dpdc)
--
--
-- -- mlp:forward(myx)
-- -- gi = mlp:backward(myx, torch.ones(2))
-- eps = 1e-5
--
-- for i = 1, myx:size(1) do
--     for j = 1, myx:size(2) do
--         local orig  = myx[i][j]
--         myx[i][j] = myx[i][j] + eps
--         local rloss = feval(myx)
--         myx[i][j] = myx[i][j] - 2*eps
--         local lloss = feval(myx)
--         local fd = (rloss - lloss)/(2*eps)
--         print(fd, mlp.gradInput[i][j])
--         myx[i][j] = orig
--     end
-- end
