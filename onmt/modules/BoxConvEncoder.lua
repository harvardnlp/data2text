local BoxConvEncoder, parent = torch.class('onmt.BoxConvEncoder', 'nn.Container')

function BoxConvEncoder:__init(nRows, nCols, encDim)
    parent.__init(self)

    self.nRows = nRows
    self.nCols = nCols
    -- have stuff for both cells and hiddens
    self.conv = self:_buildModel(nRows, nCols, encDim)
    self:add(self.conv)
end

-- K is the same for kW,kH
function BoxConvEncoder:_buildModel(nRows, nCols, encDim, nLayers, K)
    -- exects nRows*srcLen x batchSize tensor of word indices as input
    local K = K or 3
    local mod = nn.Sequential()
                  :add(nn.LookupTable(vocabSize, encDim)) -- nRows*srcLen x batchSize x encDim
                  :add(nn.Transpose({1,2}, {2,3})) -- batchSize x encDim x nRows*srcLen
                  :add(nn.Reshape(encDim, nRows, nCols)) -- batchSize x encDim x nRows x nCols
    for i = 1, nLayers do
        -- nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH
        mod:add(cudnn.SpatialConvolution(encDim, encDim, K, K, 1, 1, (K-1)/2, (K-1)/2))
        mod:add(cudnn.SpatialBatchNormalization(encDim))
        mod:add(cudnn.ReLU()) -- make nn.LeakyReLU(0.2)?
    end
    return mod
end
