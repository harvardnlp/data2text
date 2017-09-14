local CIndexAddTo, parent = torch.class('nn.CIndexAddTo', 'nn.Module')

function CIndexAddTo:__init(ip, maxbatchsize, maxcols)
    parent.__init(self)
    self.inplace = ip -- only good for one arg
    self.gradInput = {}
    self.maxbatchsize = maxbatchsize or 1024
    self.maxcols = maxcols or 1000
    self.range = torch.range(0, self.maxbatchsize-1)
    self.cols = torch.Tensor(self.maxcols)
    self.outerprod = torch.Tensor()
end

function CIndexAddTo:updateOutput(input) -- expects input to be 3 things
    local dst, src, idxs = input[1], input[2], input[3]

    -- if torch.type(dst) == 'torch.CudaTensor' and torch.type(self.range) ~= 'torch.CudaTensor' then
    --     local range = torch.CudaTensor():resize(self.range:size(1)):copy(self.range)
    --     self.range = range
    --     self.cols = self.cols:cuda()
    --     self.outerprod = self.outerprod:cuda()
    -- end

    -- number of examples, number of idxs per example, and width of dst
    local N, K, V = src:size(1), src:size(2), dst:size(2)

    local range = self.range:sub(1, N)
    local cols = self.cols:sub(1, K):fill(V)
    local newidxs = self.outerprod
    newidxs:resize(N, K)
    newidxs:ger(range, cols)
    if torch.type(idxs) == 'torch.LongTensor' then
        newidxs = newidxs:long()
    end

    self.opcopy = self.opcopy or idxs.new() -- in case idxs are CudaLongTensors
    self.opcopy:resize(newidxs:size(1), newidxs:size(2))
    self.opcopy:copy(newidxs):add(idxs)
    newidxs = self.opcopy

    --newidxs:add(idxs)
    --newidxs = newidxs:long()
    self.newidxs = newidxs

    if self.inplace then
        self.output:set(dst)
    else
        self.output:resizeAs(dst):copy(dst)
    end
    self.output:view(-1):indexAdd(1, newidxs:view(-1), src:view(-1))
    return self.output
end

function CIndexAddTo:updateGradInput(input, gradOutput)
    local dst, src, idxs = input[1], input[2], input[3]
    self.gradInput[1] = self.gradInput[1] or dst.new()
    self.gradInput[2] = self.gradInput[2] or src.new()
    self.gradInput[3] = nil
    if self.inplace then
        self.gradInput[1]:set(gradOutput)
    else
        self.gradInput[1]:resizeAs(dst):copy(gradOutput)
    end
    self.gradInput[2]:resizeAs(src)
    local newidxs = self.newidxs
    self.gradInput[2]:view(-1):index(gradOutput:view(-1), 1, newidxs:view(-1))
    -- the below shouldn't actually ever happen
    for i = #input+1, #self.gradInput do
        self.gradInput[i] = nil
    end
    return self.gradInput
end
