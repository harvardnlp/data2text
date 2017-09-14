local translate = {}

translate.Advancer = require('onmt.translate.Advancer')
translate.Beam = require('onmt.translate.Beam')
translate.BeamSearcher = require('onmt.translate.BeamSearcher')
translate.DecoderAdvancer = require('onmt.translate.DecoderAdvancer')
translate.PhraseTable = require('onmt.translate.PhraseTable')
--translate.Translator = require('onmt.translate.Translator')

translate.Decoder2Advancer = require('onmt.translate.Decoder2Advancer')
translate.SwitchingDecoderAdvancer = require('onmt.translate.SwitchingDecoderAdvancer')
return translate
