local CIndexAddTo2, parent = torch.class('nn.CIndexAddTo2', 'nn.Module')

function CIndexAddTo2:__init(ip)
    parent.__init(self)
    self.inplace = ip -- only good for one arg
    self.gradInput = {}
end

function CIndexAddTo2:updateOutput(input) -- expects input to be 3 things
    local dst, src, idxs = input[1], input[2], input[3]
    if self.inplace then
        self.output:set(dst)
    else
        self.output:resizeAs(dst):copy(dst)
    end
    for i = 1, dst:size(1) do
        self.output[i]:indexAdd(1, idxs[i], src[i])
    end
    return self.output
end

function CIndexAddTo2:updateGradInput(input, gradOutput)
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
    for i = 1, dst:size(1) do
        self.gradInput[2][i]:index(gradOutput[i], 1, idxs[i])
    end
    -- the below shouldn't actually ever happen
    for i = #input+1, #self.gradInput do
        self.gradInput[i] = nil
    end
    return self.gradInput
end
