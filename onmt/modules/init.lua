onmt = onmt or {}

require('onmt.modules.Sequencer')
require('onmt.modules.Encoder')
require('onmt.modules.BiEncoder')
require('onmt.modules.Decoder')

require('onmt.modules.LSTM')

require('onmt.modules.MaskedSoftmax')
require('onmt.modules.WordEmbedding')
require('onmt.modules.FeaturesEmbedding')
require('onmt.modules.GlobalAttention')

require('onmt.modules.Generator')
require('onmt.modules.FeaturesGenerator')

require('onmt.modules.Aggregator')
require('onmt.modules.BoxTableEncoder')

require('onmt.modules.Decoder2')
require('onmt.modules.CopyGenerator')
require('onmt.modules.CopyGenerator2')
require('onmt.modules.MarginalNLLCriterion')
require('onmt.modules.KMinDist')
require('onmt.modules.KMinXent')
require('onmt.modules.ConvRecDecoder')
require('onmt.modules.PairwiseDistDist')

require('onmt.modules.SwitchingDecoder')
require('onmt.modules.PointerGenerator')

--require('onmt.modules.CopyPOEGenerator')
require('onmt.modules.CIndexAddTo')
--require('onmt.modules.MaxMask')
--require('onmt.modules.StupidMaxThing')


return onmt
