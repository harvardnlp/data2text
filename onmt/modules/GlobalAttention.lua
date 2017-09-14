require('nngraph')

--[[ Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


    H_1 H_2 H_3 ... H_n
     q   q   q       q
      |  |   |       |
       \ |   |      /
           .....
         \   |  /
             a

Constructs a unit mapping:
  $$(H_1 .. H_n, q) => (a)$$
  Where H is of `batch x n x dim` and q is of `batch x dim`.

  The full function is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.

--]]
local GlobalAttention, parent = torch.class('onmt.GlobalAttention', 'nn.Container')

--[[A nn-style module computing attention.

  Parameters:

  * `dim` - dimension of the context vectors.
  * `returnAttnScores` - also out unnormalized attn scores
  * `tanhQuery` - use tanh(q) as query vector
--]]
function GlobalAttention:__init(dim, returnAttnScores, tanhQuery)
  parent.__init(self)
  self.returnAttnScores = returnAttnScores
  self.tanhQuery = tanhQuery
  self.net = self:_buildModel(dim)
  self:add(self.net)
end

function GlobalAttention:_buildModel(dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())

  local targetT = nn.Linear(dim, dim, false)(inputs[1]) -- batchL x dim
  if self.tanhQuery then
    targetT = nn.Tanh()(targetT)
  end
  local context = inputs[2] -- batchL x sourceTimesteps x dim

  -- Get attention.
  local attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
  attn = nn.Sum(3)(attn)
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  local attnDist = softmaxAttn(attn)
  attnDist = nn.Replicate(1,2)(attnDist) -- batchL x 1 x sourceL

  -- Apply attention to context.
  local contextCombined = nn.MM()({attnDist, context}) -- batchL x 1 x dim
  contextCombined = nn.Sum(2)(contextCombined) -- batchL x dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x dim*2
  local contextOutput = nn.Tanh()(nn.Linear(dim*2, dim, false)(contextCombined))
  local outputs = {contextOutput}
  if self.returnAttnScores then
    table.insert(outputs, attn)
  end
  return nn.gModule(inputs, outputs)
end

function GlobalAttention:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function GlobalAttention:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function GlobalAttention:accGradParameters(input, gradOutput, scale)
  return self.net:accGradParameters(input, gradOutput, scale)
end
