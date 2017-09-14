require('onmt.init')

local tds = require('tds')
local path = require('pl.path')
local cjson = require('cjson')
local stringx = require('pl.stringx')

local cmd = torch.CmdLine()

cmd:text("")
cmd:text("box_preprocess.lua")
cmd:text("")
cmd:text("**Preprocess Options**")
cmd:text("")
cmd:text("")
cmd:option('-config', '', [[Read options from this file]])

cmd:option('-json_data_dir', '', [[Path to directory containing json data]])

cmd:option('-save_data', '', [[Output file for the prepared data]])

cmd:option('-src_vocab_size', 50000, [[Size of the source vocabulary]])
cmd:option('-tgt_vocab_size', 50000, [[Size of the target vocabulary]])
cmd:option('-src_vocab', '', [[Path to an existing source vocabulary]])
cmd:option('-tgt_vocab', '', [[Path to an existing target vocabulary]])
cmd:option('-features_vocabs_prefix', '', [[Path prefix to existing features vocabularies]])
cmd:option('-ptr_fi', '', [[Path to pointer file (for conditional copy attn)]])

cmd:option('-src_seq_length', 5000000, [[Maximum source sequence length]])
cmd:option('-tgt_seq_length', 5000000, [[Maximum target sequence length]])
cmd:option('-shuffle', 1, [[Shuffle data]])
cmd:option('-seed', 3435, [[Random seed]])

cmd:option('-players_per_team', 13, [[Max players per team]])

cmd:option('-report_every', 100000, [[Report status every this many sentences]])

local opt = cmd:parse(arg)

local function hasFeatures(filename)
  local reader = onmt.utils.FileReader.new(filename)
  local _, _, numFeatures = onmt.utils.Features.extract(reader:next())
  reader:close()
  return numFeatures > 0
end

local bs_keys = {"PLAYER_NAME", "START_POSITION", "MIN", "PTS", "FGM", "FGA",
     "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB",
     "DREB", "REB", "AST", "TO", "STL", "BLK", "PF", "FIRST_NAME",
     "SECOND_NAME"}

local ls_keys = {"TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4", "TEAM-PTS",
   "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB", "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES",
   "TEAM-CITY", "TEAM-NAME"}

-- this will make vocab for every word in summary or in a table cell or header
local function makeVocabulary(jsondat, size)
  local wordVocab = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                                         onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD})
  local featuresVocabs = {}

  local colVocab = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD}) -- UNK not really necessary
  for i = 1, #bs_keys do
      colVocab:add(bs_keys[i])
  end
  for i = 1, #ls_keys do
      colVocab:add(ls_keys[i])
  end

  local rowVocab = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD})
  local cellVocab = onmt.utils.Dict.new({onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD})
  cellVocab:add("N/A")

  for i = 1, #jsondat do
      local game = jsondat[i]
      -- add all the words in the summary
      for j = 1, #game.summary do
          wordVocab:add(game.summary[j])
      end
      -- add all the box score poop
      for t = 1, #bs_keys do
          local k = bs_keys[t]
          local tbl = game.box_score[k]
          for idx, val in pairs(tbl) do
              wordVocab:add(val)
              cellVocab:add(val)
          end
      end

      -- add all the linescore stuff
      for t = 1, #ls_keys do
          local k = ls_keys[t]
          local v = game.home_line[k]
          wordVocab:add(v)
          cellVocab:add(v)
          wordVocab:add(game.vis_line[k])
          cellVocab:add(game.vis_line[k])
      end

      for k, v in pairs(game.box_score.PLAYER_NAME) do
          rowVocab:add(v)
      end
      rowVocab:add(game.home_line["TEAM-NAME"])
      rowVocab:add(game.vis_line["TEAM-NAME"])
  end

  local originalSize = wordVocab:size()
  wordVocab = wordVocab:prune(size)
  print('Created dictionary of size ' .. wordVocab:size() .. ' (pruned from ' .. originalSize .. ')')

  return wordVocab, featuresVocabs, colVocab, rowVocab, cellVocab
end

local function initVocabulary(name, jsondat, vocabFile, vocabSize, featuresVocabsFiles)
  local wordVocab, colVocab, rowVocab, cellVocab
  local featuresVocabs = {}

  if vocabFile:len() > 0 then
    -- If given, load existing word dictionary.
    print('Reading ' .. name .. ' vocabulary from \'' .. vocabFile .. '\'...')
    wordVocab = onmt.utils.Dict.new()
    wordVocab:loadFile(vocabFile)
    print('Loaded ' .. wordVocab:size() .. ' ' .. name .. ' words')
  end

  if featuresVocabsFiles:len() > 0 then
    -- If given, discover existing features dictionaries.
    local j = 1

    while true do
      local file = featuresVocabsFiles .. '.' .. name .. '_feature_' .. j .. '.dict'

      if not path.exists(file) then
        break
      end

      print('Reading ' .. name .. ' feature ' .. j .. ' vocabulary from \'' .. file .. '\'...')
      featuresVocabs[j] = onmt.utils.Dict.new()
      featuresVocabs[j]:loadFile(file)
      print('Loaded ' .. featuresVocabs[j]:size() .. ' labels')

      j = j + 1
    end
  end

  if wordVocab == nil or (#featuresVocabs == 0 and hasFeatures(dataFile)) then
    -- If a dictionary is still missing, generate it.
    print('Building ' .. name  .. ' vocabulary...')
    local genWordVocab, genFeaturesVocabs, genColVocab, genRowVocab, genCellVocab = makeVocabulary(jsondat, vocabSize)

    if wordVocab == nil then
      wordVocab = genWordVocab
      colVocab = genColVocab
      rowVocab = genRowVocab
      cellVocab = genCellVocab
    end
    if #featuresVocabs == 0 then
      featuresVocabs = genFeaturesVocabs
    end
  end

  print('')

  return {
    words = wordVocab,
    features = featuresVocabs,
    cols = colVocab,
    rows = rowVocab,
    cells = cellVocab
  }
end

local function saveVocabulary(name, vocab, file)
  print('Saving ' .. name .. ' vocabulary to \'' .. file .. '\'...')
  vocab:writeFile(file)
end

local function saveFeaturesVocabularies(name, vocabs, prefix)
  for j = 1, #vocabs do
    local file = prefix .. '.' .. name .. '_feature_' .. j .. '.dict'
    print('Saving ' .. name .. ' feature ' .. j .. ' vocabulary to \'' .. file .. '\'...')
    vocabs[j]:writeFile(file)
  end
end

local function vecToTensor(vec)
  local t = torch.Tensor(#vec)
  for i, v in pairs(vec) do
    t[i] = v
  end
  return t
end

local function get_player_idxs(game, max_per_team)
    local home_players, vis_players = {}, {}
    -- count total number of players
    local nplayers = 0
    for k,v in pairs(game.box_score['PTS']) do
        nplayers = nplayers + 1
    end

    local num_home, num_vis = 0, 0
    for i = 1, nplayers do
        local player_city = game.box_score.TEAM_CITY[tostring(i-1)]
        if player_city == game.home_city then
            if #home_players < max_per_team then
                table.insert(home_players, tostring(i-1))
                num_home = num_home + 1
            end
        else
            if #vis_players < max_per_team then
                table.insert(vis_players, tostring(i-1))
                num_vis = num_vis + 1
            end
        end
    end
    --print("adding", num_home, num_vis, "players")
    return home_players, vis_players
end

local function makeData(jsondat, srcDicts, tgtDicts, shuffle)

  local players_per_team = opt.players_per_team
  -- make a src for each row
  local srcs = {}
  for i = 1, 2*players_per_team+2 do -- 2 teams
      table.insert(srcs, tds.Vec())
  end
  local srcFeatures = tds.Vec()

  local srcTriples = tds.Vec()

  local tgt = tds.Vec()
  local tgtFeatures = tds.Vec()

  local sizes = tds.Vec() -- will be target sizes...

  local count = 0
  local ignored = 0


  for i = 1, #jsondat do
    local game = jsondat[i]
    -- get player_idxs for each team, since there're not always 13 of each
    local home_players, vis_players = get_player_idxs(game, players_per_team)

    local tgtTokens = game.summary

    -- row, col, val
    -- leave out PLAYER_NAME, FIRST_NAME, SECOND_NAME in bs_keys, and TEAM-NAME, TEAM-CITY in ls_keys
    local gameTriples = torch.IntTensor(2*players_per_team*(#bs_keys-3) + 2*(#ls_keys-2), 3):fill(1)

    if #tgtTokens > 0 and #tgtTokens <= opt.tgt_seq_length then
      local tgtWords = tgtTokens

      local tripleIdx = 1
      for ii, player_list in ipairs({home_players, vis_players}) do
          for j = 1, players_per_team do
              local src_j = {}
              local player_key = player_list[j] -- can be nil if not enough

              local playerIdx = srcDicts.rows:lookup(game.box_score.PLAYER_NAME[player_key])
              if not playerIdx and not shuffle then -- validation
                  playerIdx = 2 -- UNK
              end
              assert(playerIdx or not player_key)

              for k, key in ipairs(bs_keys) do
                  local val = game.box_score[key][player_key]
                  assert(val or (not player_key))
                  table.insert(src_j, val or "N/A")
                  if player_key and key ~= "PLAYER_NAME" and key ~= "FIRST_NAME" and key ~= "SECOND_NAME" then
                      local colIdx = srcDicts.cols:lookup(key)
                      assert(colIdx)
                      local valIdx = srcDicts.cells:lookup(val)
                      assert(valIdx)
                      gameTriples[tripleIdx][1] = playerIdx
                      gameTriples[tripleIdx][2] = colIdx
                      gameTriples[tripleIdx][3] = valIdx
                      tripleIdx = tripleIdx + 1
                  end
              end
              local idxs = srcDicts.words:convertToIdx(src_j, onmt.Constants.UNK_WORD)
              assert(idxs:dim() > 0)
              srcs[(ii-1)*players_per_team+j]:insert(idxs)
          end
      end

      -- make line scores the same size as box scores by pre-padding
      local home_src, vis_src = {}, {}
      for j = 1, (#bs_keys - #ls_keys) do
          table.insert(home_src, onmt.Constants.PAD_WORD)
          table.insert(vis_src, onmt.Constants.PAD_WORD)
      end

      -- add rest of the stuff
      local homeIdx = srcDicts.rows:lookup(game.home_line["TEAM-NAME"])
      assert(homeIdx)
      local visIdx = srcDicts.rows:lookup(game.vis_line["TEAM-NAME"])
      assert(visIdx)
      for k, key in ipairs(ls_keys) do
          local colIdx = srcDicts.cols:lookup(key)
          assert(colIdx)

          table.insert(home_src, game.home_line[key])
          local homeValIdx = srcDicts.cells:lookup(game.home_line[key])
          assert(homeValIdx)

          table.insert(vis_src, game.vis_line[key])
          local visValIdx = srcDicts.cells:lookup(game.vis_line[key])
          if not visValIdx and not shuffle then --Validation
              visValIdx = 2
          end
          assert(visValIdx)

          if key ~= "TEAM-NAME" and key ~= "TEAM-CITY" then
              gameTriples[tripleIdx][1] = homeIdx
              gameTriples[tripleIdx][2] = colIdx
              gameTriples[tripleIdx][3] = homeValIdx
              tripleIdx = tripleIdx + 1

              gameTriples[tripleIdx][1] = visIdx
              gameTriples[tripleIdx][2] = colIdx
              gameTriples[tripleIdx][3] = visValIdx
              tripleIdx = tripleIdx + 1
          end

      end

      assert(#home_src == srcs[1][1]:size(1))
      assert(#vis_src == srcs[1][1]:size(1))
      local idxs = srcDicts.words:convertToIdx(home_src, onmt.Constants.UNK_WORD)
      assert(idxs:dim() > 0)
      srcs[2*players_per_team+1]:insert(idxs)
      idxs = srcDicts.words:convertToIdx(vis_src, onmt.Constants.UNK_WORD)
      assert(idxs:dim() > 0)
      srcs[2*players_per_team+2]:insert(idxs)

      srcTriples:insert(gameTriples)

      --src:insert(srcDicts.words:convertToIdx(srcWords, onmt.Constants.UNK_WORD))
      tgt:insert(tgtDicts.words:convertToIdx(tgtWords,
                                             onmt.Constants.UNK_WORD,
                                             onmt.Constants.BOS_WORD,
                                             onmt.Constants.EOS_WORD))

      if #srcDicts.features > 0 then
        srcFeatures:insert(onmt.utils.Features.generateSource(srcDicts.features, srcFeats, true))
      end
      if #tgtDicts.features > 0 then
        tgtFeatures:insert(onmt.utils.Features.generateTarget(tgtDicts.features, tgtFeats, true))
      end

      sizes:insert(#tgtWords)
    else
      ignored = ignored + 1
    end

    count = count + 1

    if count % opt.report_every == 0 then
      print('... ' .. count .. ' sentences prepared')
    end
  end     -- end for i = 1, #jsondat

  local function reorderData(perm)
    tgt = onmt.utils.Table.reorder(tgt, perm, true)

    for j = 1, #srcs do
        srcs[j] = onmt.utils.Table.reorder(srcs[j], perm, true)
    end

    srcTriples = onmt.utils.Table.reorder(srcTriples, perm, true)

    if opt.ptr_fi:len() > 0 then
        g_ptrStuff = onmt.utils.Table.reorder(g_ptrStuff, perm, true)
    end

    if #srcDicts.features > 0 then
      srcFeatures = onmt.utils.Table.reorder(srcFeatures, perm, true)
    end
    if #tgtDicts.features > 0 then
      tgtFeatures = onmt.utils.Table.reorder(tgtFeatures, perm, true)
    end
  end

  if opt.ptr_fi:len() > 0 then
      assert(not shuffle or #g_ptrStuff == #srcs[1])
  end

  if shuffle then
    print('... shuffling sentences')
    local perm = torch.randperm(#tgt)
    sizes = onmt.utils.Table.reorder(sizes, perm, true)
    reorderData(perm)
  end

  if shuffle then
      print('... sorting sentences by size')
      local _, perm = torch.sort(vecToTensor(sizes))
      reorderData(perm)
  end

  print('Prepared ' .. #tgt .. ' sentences (' .. ignored
          .. ' ignored due to source length > ' .. opt.src_seq_length
          .. ' or target length > ' .. opt.tgt_seq_length .. ')')

  local srcData = {
    words = srcs,
    features = srcFeatures,
    triples = srcTriples
  }

  local tgtData = {
    words = tgt,
    features = tgtFeatures,
    pointers = shuffle and g_ptrStuff
  }

  return srcData, tgtData
end

local function main()
  local requiredOptions = {
    "json_data_dir",
    "save_data"
  }

  onmt.utils.Opt.init(opt, requiredOptions)
  local jsondir = opt.json_data_dir
  if jsondir:sub(jsondir:len(), jsondir:len()) ~= '/' then
      jsondir = jsondir .. '/'
  end

  local f = io.open(jsondir .. "train.json")
  local jsondat_train = cjson.decode(f:read("*all"))
  f:close()

  local f = io.open(jsondir .. "valid.json")
  local jsondat_valid = cjson.decode(f:read("*all"))
  f:close()

  local f = io.open(jsondir .. "test.json")
  local jsondat_test = cjson.decode(f:read("*all"))
  f:close()

  if opt.ptr_fi:len() > 0 then -- mapping from target values to table values
      g_ptrStuff = tds.Vec()
      local fi = assert(io.open(opt.ptr_fi, "r"))
      while true do
          local line = fi:read()
          if line == nil then
              break
          end
          local pieces = stringx.split(line)
          local lineTuples = {}
          local maxTupleLen = 0
          for j = 1, #pieces do
              local tuple = stringx.split(pieces[j], ',')
              table.insert(lineTuples, tuple)
              if #tuple > maxTupleLen then
                  maxTupleLen = #tuple
              end
          end
          assert(#pieces == #lineTuples)
          -- put these in a tensor
          local tupleTensor = torch.IntTensor(#lineTuples, maxTupleLen+1) -- last idx will have length
          for j = 1, #lineTuples do
              tupleTensor[j][maxTupleLen+1] = #lineTuples[j]-1 -- number of labels/ptrsrcs
              tupleTensor[j][1] = tonumber(lineTuples[j][1])+1 -- make 1-indexed
              for k = 2, #lineTuples[j] do
                  tupleTensor[j][k] = tonumber(lineTuples[j][k])+1 -- make 1-indexed
              end
          end
          g_ptrStuff:insert(tupleTensor)
      end
      fi:close()
  end

  local data = {}

  data.dicts = {}
  data.dicts.src = initVocabulary('source', jsondat_train, opt.src_vocab,
                                  opt.src_vocab_size, opt.features_vocabs_prefix)
  --for k,v in pairs(data.dicts.src) do print(k) end

  data.dicts.tgt = data.dicts.src

  print('Preparing training data...')
  data.train = {}
  data.train.src, data.train.tgt = makeData(jsondat_train,
                                            data.dicts.src, data.dicts.tgt, true)
  print('')

  print('Preparing validation data...')
  data.valid = {}
  data.valid.src, data.valid.tgt = makeData(jsondat_valid,
                                            data.dicts.src, data.dicts.tgt, false)

  data.test = {}
  data.test.src, data.test.tgt = makeData(jsondat_test, data.dicts.src, data.dicts.tgt, false)
  print('')

  if opt.src_vocab:len() == 0 then
    saveVocabulary('source', data.dicts.src.words, opt.save_data .. '.src.dict')
  end

  if opt.tgt_vocab:len() == 0 then
    saveVocabulary('target', data.dicts.tgt.words, opt.save_data .. '.tgt.dict')
  end

  if opt.features_vocabs_prefix:len() == 0 then
    saveFeaturesVocabularies('source', data.dicts.src.features, opt.save_data)
    saveFeaturesVocabularies('target', data.dicts.tgt.features, opt.save_data)
  end

  print('Saving data to \'' .. opt.save_data .. '-train.t7\'...')
  torch.save(opt.save_data .. '-train.t7', data, 'binary', false)

end

main()
