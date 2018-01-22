import sys, codecs, json, os
from collections import Counter, defaultdict
from nltk import sent_tokenize, word_tokenize
import numpy as np
import h5py
# import re
import random
import math
from text2num import text2num, NumberException
import argparse

random.seed(2)


prons = set(["he", "He", "him", "Him", "his", "His", "they", "They", "them", "Them", "their", "Their"]) # leave out "it"
singular_prons = set(["he", "He", "him", "Him", "his", "His"])
plural_prons = set(["they", "They", "them", "Them", "their", "Their"])

number_words = set(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                    "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
                    "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"])


def get_ents(dat):
    players = set()
    teams = set()
    cities = set()
    for thing in dat:
        teams.add(thing["vis_name"])
        teams.add(thing["vis_line"]["TEAM-NAME"])
        teams.add(thing["vis_city"] + " " + thing["vis_name"])
        teams.add(thing["vis_city"] + " " + thing["vis_line"]["TEAM-NAME"])
        teams.add(thing["home_name"])
        teams.add(thing["home_line"]["TEAM-NAME"])
        teams.add(thing["home_city"] + " " + thing["home_name"])
        teams.add(thing["home_city"] + " " + thing["home_line"]["TEAM-NAME"])
        # special case for this
        if thing["vis_city"] == "Los Angeles":
            teams.add("LA" + thing["vis_name"])
        if thing["home_city"] == "Los Angeles":
            teams.add("LA" + thing["home_name"])
        # sometimes team_city is different
        cities.add(thing["home_city"])
        cities.add(thing["vis_city"])
        players.update(thing["box_score"]["PLAYER_NAME"].values())
        cities.update(thing["box_score"]["TEAM_CITY"].values())

    for entset in [players, teams, cities]:
        for k in list(entset):
            pieces = k.split()
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in ["II", "III", "Jr.", "Jr"]:
                        entset.add(piece)

    all_ents = players | teams | cities

    return all_ents, players, teams, cities


def deterministic_resolve(pron, players, teams, cities, curr_ents, prev_ents, max_back=1):
    # we'll just take closest compatible one.
    # first look in current sentence; if there's an antecedent here return None, since
    # we'll catch it anyway
    for j in xrange(len(curr_ents)-1, -1, -1):
        if pron in singular_prons and curr_ents[j][2] in players:
            return None
        elif pron in plural_prons and curr_ents[j][2] in teams:
            return None
        elif pron in plural_prons and curr_ents[j][2] in cities:
            return None

    # then look in previous max_back sentences
    if len(prev_ents) > 0:
        for i in xrange(len(prev_ents)-1, len(prev_ents)-1-max_back, -1):
            for j in xrange(len(prev_ents[i])-1, -1, -1):
                if pron in singular_prons and prev_ents[i][j][2] in players:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in teams:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in cities:
                    return prev_ents[i][j]
    return None


def extract_entities(sent, all_ents, prons, prev_ents=None, resolve_prons=False,
        players=None, teams=None, cities=None):
    sent_ents = []
    i = 0
    while i < len(sent):
        if sent[i] in prons:
            if resolve_prons:
                referent = deterministic_resolve(sent[i], players, teams, cities, sent_ents, prev_ents)
                if referent is None:
                    sent_ents.append((i, i+1, sent[i], True)) # is a pronoun
                else:
                    #print "replacing", sent[i], "with", referent[2], "in", " ".join(sent)
                    sent_ents.append((i, i+1, referent[2], False)) # pretend it's not a pron and put in matching string
            else:
                sent_ents.append((i, i+1, sent[i], True)) # is a pronoun
            i += 1
        elif sent[i] in all_ents: # findest longest spans; only works if we put in words...
            j = 1
            while i+j <= len(sent) and " ".join(sent[i:i+j]) in all_ents:
                j += 1
            sent_ents.append((i, i+j-1, " ".join(sent[i:i+j-1]), False))
            i += j-1
        else:
            i += 1
    return sent_ents


def annoying_number_word(sent, i):
    ignores = set(["three point", "three - point", "three - pt", "three pt"])
    return " ".join(sent[i:i+3]) not in ignores and " ".join(sent[i:i+2]) not in ignores

def extract_numbers(sent):
    sent_nums = []
    i = 0
    ignores = set(["three point", "three-point", "three-pt", "three pt"])
    #print sent
    while i < len(sent):
        toke = sent[i]
        a_number = False
        try:
            itoke = int(toke)
            a_number = True
        except ValueError:
            pass
        if a_number:
            sent_nums.append((i, i+1, int(toke)))
            i += 1
        elif toke in number_words and not annoying_number_word(sent, i): # get longest span  (this is kind of stupid)
            j = 1
            while i+j <= len(sent) and sent[i+j] in number_words and not annoying_number_word(sent, i+j):
                j += 1
            try:
                sent_nums.append((i, i+j, text2num(" ".join(sent[i:i+j]))))
            except NumberException:
                print sent
                print sent[i:i+j]
                assert False
            i += j
        else:
            i += 1
    return sent_nums


def get_player_idx(bs, entname):
    keys = []
    for k, v in bs["PLAYER_NAME"].iteritems():
         if entname == v:
             keys.append(k)
    if len(keys) == 0:
        for k,v in bs["SECOND_NAME"].iteritems():
            if entname == v:
                keys.append(k)
        if len(keys) > 1: # take the earliest one
            keys.sort(key = lambda x: int(x))
            keys = keys[:1]
            #print "picking", bs["PLAYER_NAME"][keys[0]]
    if len(keys) == 0:
        for k,v in bs["FIRST_NAME"].iteritems():
            if entname == v:
                keys.append(k)
        if len(keys) > 1: # if we matched on first name and there are a bunch just forget about it
            return None
    #if len(keys) == 0:
        #print "Couldn't find", entname, "in", bs["PLAYER_NAME"].values()
    assert len(keys) <= 1, entname + " : " + str(bs["PLAYER_NAME"].values())
    return keys[0] if len(keys) > 0 else None


def get_rels(entry, ents, nums, players, teams, cities):
    """
    this looks at the box/line score and figures out which (entity, number) pairs
    are candidate true relations, and which can't be.
    if an ent and number don't line up (i.e., aren't in the box/line score together),
    we give a NONE label, so for generated summaries that we extract from, if we predict
    a label we'll get it wrong (which is presumably what we want).
    N.B. this function only looks at the entity string (not position in sentence), so the
    string a pronoun corefers with can be snuck in....
    """
    rels = []
    bs = entry["box_score"]
    for i, ent in enumerate(ents):
        if ent[3]: # pronoun
            continue # for now
        entname = ent[2]
        # assume if a player has a city or team name as his name, they won't use that one (e.g., Orlando Johnson)
        if entname in players and entname not in cities and entname not in teams:
            pidx = get_player_idx(bs, entname)
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup[2])
                if pidx is not None: # player might not actually be in the game or whatever
                    for colname, col in bs.iteritems():
                        if col[pidx] == strnum: # allow multiple for now
                            rels.append((ent, numtup, "PLAYER-" + colname, pidx))
                            found = True
                if not found:
                    rels.append((ent, numtup, "NONE", None))

        else: # has to be city or team
            entpieces = entname.split()
            linescore = None
            is_home = None
            if entpieces[0] in entry["home_city"] or entpieces[-1] in entry["home_name"]:
                linescore = entry["home_line"]
                is_home = True
            elif entpieces[0] in entry["vis_city"] or entpieces[-1] in entry["vis_name"]:
                linescore = entry["vis_line"]
                is_home = False
            elif "LA" in entpieces[0]:
                if entry["home_city"] == "Los Angeles":
                    linescore = entry["home_line"]
                    is_home = True
                elif entry["vis_city"] == "Los Angeles":
                    linescore = entry["vis_line"]
                    is_home = False
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup[2])
                if linescore is not None:
                    for colname, val in linescore.iteritems():
                        if val == strnum:
                            #rels.append((ent, numtup, "TEAM-" + colname, is_home))
                            # apparently I appended TEAM- at some pt...
                            rels.append((ent, numtup, colname, is_home))
                            found = True
                if not found:
                    rels.append((ent, numtup, "NONE", None)) # should i specialize the NONE labels too?
    return rels

def append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities, candrels):
    """
    appends tuples of form (sentence_tokens, [rels]) to candrels
    """
    sents = sent_tokenize(summ)
    for j, sent in enumerate(sents):
        #tokes = word_tokenize(sent)
        tokes = sent.split()
        ents = extract_entities(tokes, all_ents, prons)
        nums = extract_numbers(tokes)
        rels = get_rels(entry, ents, nums, players, teams, cities)
        if len(rels) > 0:
            candrels.append((tokes, rels))
    return candrels

def get_datasets(path="../boxscore-data/rotowire"):

    with codecs.open(os.path.join(path, "train.json"), "r", "utf-8") as f:
        trdata = json.load(f)

    all_ents, players, teams, cities = get_ents(trdata)

    with codecs.open(os.path.join(path, "valid.json"), "r", "utf-8") as f:
        valdata = json.load(f)

    extracted_stuff = []
    datasets = [trdata, valdata]
    for dataset in datasets:
        nugz = []
        for i, entry in enumerate(dataset):
            summ = " ".join(entry['summary'])
            append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities, nugz)

        extracted_stuff.append(nugz)

    del all_ents
    del players
    del teams
    del cities
    return extracted_stuff

def append_to_data(tup, sents, lens, entdists, numdists, labels, vocab, labeldict, max_len):
    """
    tup is (sent, [rels]);
    each rel is ((ent_start, ent_ent, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    for rel in tup[1]:
        ent, num, label, idthing = rel
        sents.append(sent)
        lens.append(sentlen)
        ent_dists = [j-ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in xrange(max_len)]
        entdists.append(ent_dists)
        num_dists = [j-num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in xrange(max_len)]
        numdists.append(num_dists)
        labels.append(labeldict[label])


def append_multilabeled_data(tup, sents, lens, entdists, numdists, labels, vocab, labeldict, max_len):
    """
    used for val, since we have contradictory labelings...
    tup is (sent, [rels]);
    each rel is ((ent_start, ent_end, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    # get all the labels for the same rel
    unique_rels = defaultdict(list)
    for rel in tup[1]:
        ent, num, label, idthing = rel
        unique_rels[ent, num].append(label)

    for rel, label_list in unique_rels.iteritems():
        ent, num = rel
        sents.append(sent)
        lens.append(sentlen)
        ent_dists = [j-ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0 for j in xrange(max_len)]
        entdists.append(ent_dists)
        num_dists = [j-num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0 for j in xrange(max_len)]
        numdists.append(num_dists)
        labels.append([labeldict[label] for label in label_list])

def append_labelnums(labels):
    labelnums = [len(labellist) for labellist in labels]
    max_num_labels = max(labelnums)
    print "max num labels", max_num_labels

    # append number of labels to labels
    for i, labellist in enumerate(labels):
        labellist.extend([-1]*(max_num_labels - len(labellist)))
        labellist.append(labelnums[i])

# for full sentence IE training
def save_full_sent_data(outfile, path="../boxscore-data/rotowire", multilabel_train=False, nonedenom=0):
    datasets = get_datasets(path)
    # make vocab and get labels
    word_counter = Counter()
    [word_counter.update(tup[0]) for tup in datasets[0]]
    for k in word_counter.keys():
        if word_counter[k] < 2:
            del word_counter[k] # will replace w/ unk
    word_counter["UNK"] = 1
    vocab = dict(((wrd, i+1) for i, wrd in enumerate(word_counter.keys())))
    labelset = set()
    [labelset.update([rel[2] for rel in tup[1]]) for tup in datasets[0]]
    labeldict = dict(((label, i+1) for i, label in enumerate(labelset)))

    # save stuff
    trsents, trlens, trentdists, trnumdists, trlabels = [], [], [], [], []
    valsents, vallens, valentdists, valnumdists, vallabels = [], [], [], [], []
    #testsents, testlens, testentdists, testnumdists, testlabels = [], [], [], [], []

    max_trlen = max((len(tup[0]) for tup in datasets[0]))
    print "max tr sentence length:", max_trlen

    # do training data
    for tup in datasets[0]:
        if multilabel_train:
            append_multilabeled_data(tup, trsents, trlens, trentdists, trnumdists, trlabels, vocab, labeldict, max_trlen)
        else:
            append_to_data(tup, trsents, trlens, trentdists, trnumdists, trlabels, vocab, labeldict, max_trlen)

    if multilabel_train:
        append_labelnums(trlabels)

    if nonedenom > 0:
        # don't keep all the NONE labeled things
        none_idxs = [i for i, labellist in enumerate(trlabels) if labellist[0] == labeldict["NONE"]]
        random.shuffle(none_idxs)
        # allow at most 1/(nonedenom+1) of NONE-labeled
        num_to_keep = int(math.floor(float(len(trlabels)-len(none_idxs))/nonedenom))
        print "originally", len(trlabels), "training examples"
        print "keeping", num_to_keep, "NONE-labeled examples"
        ignore_idxs = set(none_idxs[num_to_keep:])

        # get rid of most of the NONE-labeled examples
        trsents = [thing for i,thing in enumerate(trsents) if i not in ignore_idxs]
        trlens = [thing for i,thing in enumerate(trlens) if i not in ignore_idxs]
        trentdists = [thing for i,thing in enumerate(trentdists) if i not in ignore_idxs]
        trnumdists = [thing for i,thing in enumerate(trnumdists) if i not in ignore_idxs]
        trlabels = [thing for i,thing in enumerate(trlabels) if i not in ignore_idxs]

    print len(trsents), "training examples"

    # do val, which we also consider multilabel
    max_vallen = max((len(tup[0]) for tup in datasets[1]))
    for tup in datasets[1]:
        #append_to_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_len)
        append_multilabeled_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_vallen)

    append_labelnums(vallabels)

    print len(valsents), "validation examples"


    # if len(datasets) > 2:
    #     # do test, which we also consider multilabel
    #     max_testlen = max((len(tup[0]) for tup in datasets[2]))
    #     for tup in datasets[2]:
    #         #append_to_data(tup, valsents, vallens, valentdists, valnumdists, vallabels, vocab, labeldict, max_len)
    #         append_multilabeled_data(tup, testsents, testlens, testentdists, testnumdists, testlabels, vocab, labeldict, max_testlen)
    #
    #     append_labelnums(testlabels)
    #
    #     print len(testsents), "test examples"

    h5fi = h5py.File(outfile, "w")
    h5fi["trsents"] = np.array(trsents, dtype=int)
    h5fi["trlens"] = np.array(trlens, dtype=int)
    h5fi["trentdists"] = np.array(trentdists, dtype=int)
    h5fi["trnumdists"] = np.array(trnumdists, dtype=int)
    h5fi["trlabels"] = np.array(trlabels, dtype=int)
    h5fi["valsents"] = np.array(valsents, dtype=int)
    h5fi["vallens"] = np.array(vallens, dtype=int)
    h5fi["valentdists"] = np.array(valentdists, dtype=int)
    h5fi["valnumdists"] = np.array(valnumdists, dtype=int)
    h5fi["vallabels"] = np.array(vallabels, dtype=int)
    h5fi.close()

    # if len(datasets) > 2:
    #     h5fi = h5py.File("test-" + outfile, "w")
    #     h5fi["testsents"] = np.array(testsents, dtype=int)
    #     h5fi["testlens"] = np.array(testlens, dtype=int)
    #     h5fi["testentdists"] = np.array(testentdists, dtype=int)
    #     h5fi["testnumdists"] = np.array(testnumdists, dtype=int)
    #     h5fi["testlabels"] = np.array(testlabels, dtype=int)
    #     h5fi.close()
    ## h5fi["vallabelnums"] = np.array(vallabelnums, dtype=int)
    ## h5fi.close()

    # write dicts
    revvocab = dict(((v,k) for k,v in vocab.iteritems()))
    revlabels = dict(((v,k) for k,v in labeldict.iteritems()))
    with codecs.open(outfile.split('.')[0] + ".dict", "w+", "utf-8") as f:
        for i in xrange(1, len(revvocab)+1):
            f.write("%s %d \n" % (revvocab[i], i))

    with codecs.open(outfile.split('.')[0] + ".labels", "w+", "utf-8") as f:
        for i in xrange(1, len(revlabels)+1):
            f.write("%s %d \n" % (revlabels[i], i))


def prep_generated_data(genfile, dict_pfx, outfile, path="../boxscore-data/rotowire", test=False):
    # recreate vocab and labeldict
    vocab = {}
    with codecs.open(dict_pfx+".dict", "r", "utf-8") as f:
        for line in f:
            pieces = line.strip().split()
            vocab[pieces[0]] = int(pieces[1])

    labeldict = {}
    with codecs.open(dict_pfx+".labels", "r", "utf-8") as f:
        for line in f:
            pieces = line.strip().split()
            labeldict[pieces[0]] = int(pieces[1])

    with codecs.open(genfile, "r", "utf-8") as f:
        gens = f.readlines()

    with codecs.open(os.path.join(path, "train.json"), "r", "utf-8") as f:
        trdata = json.load(f)

    all_ents, players, teams, cities = get_ents(trdata)

    valfi = "test.json" if test else "valid.json"
    with codecs.open(os.path.join(path, valfi), "r", "utf-8") as f:
        valdata = json.load(f)

    assert len(valdata) == len(gens)

    nugz = [] # to hold (sentence_tokens, [rels]) tuples
    sent_reset_indices = {0} # sentence indices where a box/story is reset
    for i, entry in enumerate(valdata):
        summ = gens[i]
        append_candidate_rels(entry, summ, all_ents, prons, players, teams, cities, nugz)
        sent_reset_indices.add(len(nugz))

    # save stuff
    max_len = max((len(tup[0]) for tup in nugz))
    psents, plens, pentdists, pnumdists, plabels = [], [], [], [], []

    rel_reset_indices = []
    for t, tup in enumerate(nugz):
        if t in sent_reset_indices: # then last rel is the last of its box
            assert len(psents) == len(plabels)
            rel_reset_indices.append(len(psents))
        append_multilabeled_data(tup, psents, plens, pentdists, pnumdists, plabels, vocab, labeldict, max_len)

    append_labelnums(plabels)

    print len(psents), "prediction examples"

    h5fi = h5py.File(outfile, "w")
    h5fi["valsents"] = np.array(psents, dtype=int)
    h5fi["vallens"] = np.array(plens, dtype=int)
    h5fi["valentdists"] = np.array(pentdists, dtype=int)
    h5fi["valnumdists"] = np.array(pnumdists, dtype=int)
    h5fi["vallabels"] = np.array(plabels, dtype=int)
    h5fi["boxrestartidxs"] = np.array(np.array(rel_reset_indices)+1, dtype=int) # 1-indexed
    h5fi.close()

################################################################################

bs_keys = ["PLAYER-PLAYER_NAME", "PLAYER-START_POSITION", "PLAYER-MIN", "PLAYER-PTS",
     "PLAYER-FGM", "PLAYER-FGA", "PLAYER-FG_PCT", "PLAYER-FG3M", "PLAYER-FG3A",
     "PLAYER-FG3_PCT", "PLAYER-FTM", "PLAYER-FTA", "PLAYER-FT_PCT", "PLAYER-OREB",
     "PLAYER-DREB", "PLAYER-REB", "PLAYER-AST", "PLAYER-TO", "PLAYER-STL", "PLAYER-BLK",
     "PLAYER-PF", "PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME"]

ls_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4",
    "TEAM-PTS", "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB",
    "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES", "TEAM-CITY", "TEAM-NAME"]

NUM_PLAYERS = 13


def get_player_idxs(entry):
    nplayers = 0
    home_players, vis_players = [], []
    for k,v in entry["box_score"]["PTS"].iteritems():
        nplayers += 1

    num_home, num_vis = 0, 0
    for i in xrange(nplayers):
        player_city = entry["box_score"]["TEAM_CITY"][str(i)]
        if player_city == entry["home_city"]:
            if len(home_players) < NUM_PLAYERS:
                home_players.append(str(i))
                num_home += 1
        else:
            if len(vis_players) < NUM_PLAYERS:
                vis_players.append(str(i))
                num_vis += 1
    return home_players, vis_players

def box_preproc2(trdata):
    """
    just gets src for now
    """
    srcs = [[] for i in xrange(2*NUM_PLAYERS+2)]

    for entry in trdata:
        home_players, vis_players = get_player_idxs(entry)
        for ii, player_list in enumerate([home_players, vis_players]):
            for j in xrange(NUM_PLAYERS):
                src_j = []
                player_key = player_list[j] if j < len(player_list) else None
                for k, key in enumerate(bs_keys):
                    rulkey = key.split('-')[1]
                    val = entry["box_score"][rulkey][player_key] if player_key is not None else "N/A"
                    src_j.append(val)
                srcs[ii*NUM_PLAYERS + j].append(src_j)

        home_src, vis_src = [], []
        for k in xrange(len(bs_keys) - len(ls_keys)):
            home_src.append("PAD")
            vis_src.append("PAD")

        for k, key in enumerate(ls_keys):
            home_src.append(entry["home_line"][key])
            vis_src.append(entry["vis_line"][key])

        srcs[-2].append(home_src)
        srcs[-1].append(vis_src)

    return srcs

def linearized_preproc(srcs):
    """
    maps from a num-rows length list of lists of ntrain to an
    ntrain-length list of concatenated rows
    """
    lsrcs = []
    for i in xrange(len(srcs[0])):
        src_i = []
        for j in xrange(len(srcs)):
            src_i.extend(srcs[j][i][1:]) # b/c in lua we ignore first thing
        lsrcs.append(src_i)
    return lsrcs

def fix_target_idx(summ, assumed_idx, word, neighborhood=5):
    """
    tokenization can mess stuff up, so look around
    """
    for i in xrange(1, neighborhood+1):
        if assumed_idx + i < len(summ) and summ[assumed_idx + i] == word:
            return assumed_idx + i
        elif assumed_idx - i >= 0 and assumed_idx - i < len(summ) and summ[assumed_idx - i] == word:
            return assumed_idx - i
    return None

# for each target word want to know where it could've been copied from
def make_pointerfi(outfi, inp_file="rotowire/train.json", resolve_prons=False):
    """
    N.B. this function only looks at string equality in determining pointerness.
    this means that if we sneak in pronoun strings as their referents, we won't point to the
    pronoun if the referent appears in the table; we may use this tho to point to the correct number
    """
    with codecs.open(inp_file, "r", "utf-8") as f:
        trdata = json.load(f)

    rulsrcs = linearized_preproc(box_preproc2(trdata))

    all_ents, players, teams, cities = get_ents(trdata)

    skipped = 0

    train_links = []
    for i, entry in enumerate(trdata):
        home_players, vis_players = get_player_idxs(entry)
        inv_home_players = {pkey: jj for jj, pkey in enumerate(home_players)}
        inv_vis_players  = {pkey: (jj + NUM_PLAYERS) for jj, pkey in enumerate(vis_players)}
        summ = " ".join(entry['summary'])
        sents = sent_tokenize(summ)
        words_so_far = 0
        links = []
        prev_ents = []
        for j, sent in enumerate(sents):
            tokes = word_tokenize(sent) # just assuming this gives me back original tokenization
            ents = extract_entities(tokes, all_ents, prons, prev_ents, resolve_prons,
                players, teams, cities)
            if resolve_prons:
                prev_ents.append(ents)
            nums = extract_numbers(tokes)
            # should return a list of (enttup, numtup, rel-name, identifier) for each rel licensed by the table
            rels = get_rels(entry, ents, nums, players, teams, cities)
            for (enttup, numtup, label, idthing) in rels:
                if label != 'NONE':
                    # try to find corresponding words (for both ents and nums)
                    ent_start, ent_end, entspan, _ = enttup
                    num_start, num_end, numspan = numtup
                    if isinstance(idthing, bool): # city or team
                        # get entity indices if any
                        for k, word in enumerate(tokes[ent_start:ent_end]):
                            src_idx = None
                            if word == entry["home_name"]:
                                src_idx = (2*NUM_PLAYERS+1)*(len(bs_keys)-1) -1 # last thing
                            elif word == entry["home_city"]:
                                src_idx = (2*NUM_PLAYERS+1)*(len(bs_keys)-1) -2 # second to last thing
                            elif word == entry["vis_name"]:
                                src_idx = (2*NUM_PLAYERS+2)*(len(bs_keys)-1) -1 # last thing
                            elif word == entry["vis_city"]:
                                src_idx = (2*NUM_PLAYERS+2)*(len(bs_keys)-1) -2 # second to last thing
                            if src_idx is not None:
                                targ_idx = words_so_far + ent_start + k
                                if entry["summary"][targ_idx] != word:
                                    targ_idx = fix_target_idx(entry["summary"], targ_idx, word)
                                #print word, rulsrcs[i][src_idx], entry["summary"][words_so_far + ent_start + k]
                                if targ_idx is None:
                                    skipped += 1
                                else:
                                    assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                    links.append((src_idx, targ_idx))  # src_idx, target_idx

                        # get num indices if any
                        for k, word in enumerate(tokes[num_start:num_end]):
                            src_idx = None
                            if idthing: # home, so look in the home row
                                if entry["home_line"][label] == word:
                                    col_idx = ls_keys.index(label)
                                    src_idx = 2*NUM_PLAYERS*(len(bs_keys)-1)+ len(bs_keys)-len(ls_keys) + col_idx -1 # -1 b/c we trim first col
                            else:
                                if entry["vis_line"][label] == word:
                                    col_idx = ls_keys.index(label)
                                    src_idx = (2*NUM_PLAYERS+1)*(len(bs_keys)-1)+ len(bs_keys)-len(ls_keys) + col_idx - 1
                            if src_idx is not None:
                                targ_idx = words_so_far + num_start + k
                                if targ_idx >= len(entry["summary"]) or entry["summary"][targ_idx] != word:
                                    targ_idx = fix_target_idx(entry["summary"], targ_idx, word)
                                #print word, rulsrcs[i][src_idx], entry["summary"][words_so_far + num_start + k]
                                if targ_idx is None:
                                    skipped += 1
                                else:
                                    assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                    links.append((src_idx, targ_idx))
                    else: # players
                        # get row corresponding to this player
                        player_row = None
                        if idthing in inv_home_players:
                            player_row = inv_home_players[idthing]
                        elif idthing in inv_vis_players:
                            player_row = inv_vis_players[idthing]
                        if player_row is not None:
                            # ent links
                            for k, word in enumerate(tokes[ent_start:ent_end]):
                                src_idx = None
                                if word == entry["box_score"]["FIRST_NAME"][idthing]:
                                    src_idx = (player_row+1)*(len(bs_keys)-1) -2 # second to last thing
                                elif word == entry["box_score"]["SECOND_NAME"][idthing]:
                                    src_idx = (player_row+1)*(len(bs_keys)-1) -1 # last thing
                                if src_idx is not None:
                                    targ_idx = words_so_far + ent_start + k
                                    if entry["summary"][targ_idx] != word:
                                        targ_idx = fix_target_idx(entry["summary"], targ_idx, word)
                                    if targ_idx is None:
                                        skipped += 1
                                    else:
                                        assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                        links.append((src_idx, targ_idx))  # src_idx, target_idx
                            # num links
                            for k, word in enumerate(tokes[num_start:num_end]):
                                src_idx = None
                                if word == entry["box_score"][label.split('-')[1]][idthing]:
                                    src_idx = player_row*(len(bs_keys)-1) + bs_keys.index(label)-1 # subtract 1 because we ignore first col
                                if src_idx is not None:
                                    targ_idx = words_so_far + num_start + k
                                    if targ_idx >= len(entry["summary"]) or entry["summary"][targ_idx] != word:
                                        targ_idx = fix_target_idx(entry["summary"], targ_idx, word)
                                    #print word, rulsrcs[i][src_idx], entry["summary"][words_so_far + num_start + k]
                                    if targ_idx is None:
                                        skipped += 1
                                    else:
                                        assert rulsrcs[i][src_idx] == word and entry["summary"][targ_idx] == word
                                        links.append((src_idx, targ_idx))

            words_so_far += len(tokes)
        train_links.append(links)
    print "SKIPPED", skipped

    # collapse multiple links
    trlink_dicts = []
    for links in train_links:
        links_dict = defaultdict(list)
        [links_dict[targ_idx].append(src_idx) for src_idx, targ_idx in links]
        trlink_dicts.append(links_dict)

    # write in fmt:
    # targ_idx,src_idx1[,src_idx...]
    with open(outfi, "w+") as f:
        for links_dict in trlink_dicts:
            targ_idxs = sorted(links_dict.keys())
            fmtd = [",".join([str(targ_idx)]+[str(thing) for thing in set(links_dict[targ_idx])])
                       for targ_idx in targ_idxs]
            f.write("%s\n" % " ".join(fmtd))


# for coref prediction stuff
# we'll use string equality for now
def save_coref_task_data(outfile, inp_file="full_newnba_prepdata2.json"):
    with codecs.open(inp_file, "r", "utf-8") as f:
        data = json.load(f)

    all_ents, players, teams, cities = get_ents(data["train"])
    datasets = []

    # labels are nomatch, match, pron
    for dataset in [data["train"], data["valid"]]:
        examples = []
        for i, entry in enumerate(dataset):
            summ = entry["summary"]
            ents = extract_entities(summ, all_ents, prons)
            for j in xrange(1, len(ents)):
                # just get all the words from previous mention till this one starts
                prev_start, prev_end, prev_str, _ = ents[j-1]
                curr_start, curr_end, curr_str, curr_pron = ents[j]
                #window = summ[prev_start:curr_start]
                window = summ[prev_end:curr_start]
                label = None
                if curr_pron: # prons
                    label = 3
                else:
                    #label = 2 if prev_str == curr_str else 1
                    label = 2 if prev_str in curr_str or curr_str in prev_str else 1
                examples.append((window, label))
        datasets.append(examples)

    # make vocab and get labels
    word_counter = Counter()
    [word_counter.update(tup[0]) for tup in datasets[0]]
    for k in word_counter.keys():
        if word_counter[k] < 2:
            del word_counter[k] # will replace w/ unk
    word_counter["UNK"] = 1
    vocab = dict(((wrd, i+1) for i, wrd in enumerate(word_counter.keys())))
    labeldict = {"NOMATCH": 1, "MATCH": 2, "PRON": 3}

    max_trlen = max((len(tup[0]) for tup in datasets[0]))
    max_vallen = max((len(tup[0]) for tup in datasets[1]))
    print "max sentence lengths:", max_trlen, max_vallen

    # map words to indices
    trwindows = [[vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in window]
                   + [-1]*(max_trlen - len(window)) for (window, label) in datasets[0]]
    trlabels = [label for (window, label) in datasets[0]]
    valwindows = [[vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in window]
                   + [-1]*(max_vallen - len(window)) for (window, label) in datasets[1]]
    vallabels = [label for (window, label) in datasets[1]]

    print len(trwindows), "training examples"
    print len(valwindows), "validation examples"
    print Counter(trlabels)
    print Counter(vallabels)

    h5fi = h5py.File(outfile, "w")
    h5fi["trwindows"] = np.array(trwindows, dtype=int)
    h5fi["trlens"] = np.array([len(window) for (window, label) in datasets[0]], dtype=int)
    h5fi["trlabels"] = np.array(trlabels, dtype=int)
    h5fi["valwindows"] = np.array(valwindows, dtype=int)
    h5fi["vallens"] = np.array([len(window) for (window, label) in datasets[1]], dtype=int)
    h5fi["vallabels"] = np.array(vallabels, dtype=int)
    #h5fi["vallabelnums"] = np.array(vallabelnums, dtype=int)
    h5fi.close()

    # write dicts
    revvocab = dict(((v,k) for k,v in vocab.iteritems()))
    revlabels = dict(((v,k) for k,v in labeldict.iteritems()))
    with codecs.open(outfile.split('.')[0] + ".dict", "w+", "utf-8") as f:
        for i in xrange(1, len(revvocab)+1):
            f.write("%s %d \n" % (revvocab[i], i))

    with codecs.open(outfile.split('.')[0] + ".labels", "w+", "utf-8") as f:
        for i in xrange(1, len(revlabels)+1):
            f.write("%s %d \n" % (revlabels[i], i))


# if sys.argv[1] == "prep_gen":
#     generated_input = sys.argv[2]
#     dict_pfx = sys.argv[3]
#     output_fi = sys.argv[4]
#     if len(sys.argv) > 5:
#         start_after = int(sys.argv[5])
#         prep_generated_data(generated_input, dict_pfx, output_fi, start_after)
#     else:
#         prep_generated_data(generated_input, dict_pfx, output_fi)
# else:
#     train_output_fi = sys.argv[2]
#     multilabel_train = sys.argv[3].lower() == "true"
#     save_full_sent_data(train_output_fi, multilabel_train=multilabel_train)




parser = argparse.ArgumentParser(description='Utility Functions')
parser.add_argument('-input_path', type=str, default="",
                    help="path to input")
parser.add_argument('-output_fi', type=str, default="",
                    help="desired path to output file")
parser.add_argument('-gen_fi', type=str, default="",
                    help="path to file containing generated summaries")
parser.add_argument('-dict_pfx', type=str, default="roto-ie",
                    help="prefix of .dict and .labels files")
parser.add_argument('-mode', type=str, default='ptrs',
                    choices=['ptrs', 'make_ie_data', 'prep_gen_data'],
                    help="what utility function to run")
parser.add_argument('-test', action='store_true', help='use test data')

args = parser.parse_args()

if args.mode == 'ptrs':
    make_pointerfi(args.output_fi, inp_file=args.input_path)
elif args.mode == 'make_ie_data':
    save_full_sent_data(args.output_fi, path=args.input_path, multilabel_train=True)
elif args.mode == 'prep_gen_data':
    prep_generated_data(args.gen_fi, args.dict_pfx, args.output_fi, path=args.input_path,
                        test=args.test)
