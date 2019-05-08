import sys
import codecs, json

first_line = "The %s ( %d - %d ) defeated the %s ( %d - %d ) %d - %d ."
player_line = "%s scored %d points ( %d - %d FG , %d - %d 3Pt , %d - %d FT ) to go with %d rebounds ."
last_line = "The %s ' next game will be at home against the Dallas Mavericks , while the %s will travel to play the Bulls ."

def get_line_info(line):
    city = line["TEAM-CITY"]
    name = line["TEAM-NAME"]
    wins = int(line["TEAM-WINS"])
    losses = int(line["TEAM-LOSSES"])
    pts = int(line["TEAM-PTS"])
    return city, name, wins, losses, pts

def get_best_players(bs, k):
    """
    for now just take players w/ most points.
    returns (name, pts, fgm, fga, 3pm, 3pa, ftm, fta, reb)
    """
    player_pts = list(bs["PTS"].iteritems())
    player_pts.sort(key=lambda x: -int(x[1]) if x[1] != "N/A" else 10000)
    player_tups = []
    for (pid, pts) in player_pts[:k]:
        player_tups.append(
          (bs["PLAYER_NAME"][pid], int(bs["PTS"][pid]), int(bs["FGM"][pid]), int(bs["FGA"][pid]),
          int(bs["FG3M"][pid]), int(bs["FG3A"][pid]), int(bs["FTM"][pid]), int(bs["FTA"][pid]), int(bs["REB"][pid]))
        )
    return player_tups


def rule_gen2(entry, k=6):
    home_city, home_name, home_wins, home_losses, home_score = get_line_info(entry["home_line"])
    vis_city, vis_name, vis_wins, vis_losses, vis_score = get_line_info(entry["vis_line"])
    home_won = home_score > vis_score
    summ = []
    if home_won:
        summ.append(first_line % (home_city + " " + home_name, home_wins, home_losses,
               vis_city + " " + vis_name, vis_wins, vis_losses, home_score, vis_score))
    else:
        summ.append(first_line % (vis_city + " " + vis_name, vis_wins, vis_losses,
               home_city + " " + home_name, home_wins, home_losses, vis_score, home_score))
    k_best = get_best_players(entry["box_score"], k)
    for player_tup in k_best:
        summ.append(player_line % (player_tup))
    summ.append(last_line % (vis_name, home_name))
    return " ".join(summ)

def rule_gen(entry, k=6):
    home_city, home_name, home_wins, home_losses, home_score = get_line_info(entry["home_line"])
    vis_city, vis_name, vis_wins, vis_losses, vis_score = get_line_info(entry["vis_line"])
    home_won = home_score > vis_score
    summ = []
    if home_won:
        summ.append(first_line % (home_city + " " + home_name, home_wins, home_losses,
               vis_city + " " + vis_name, vis_wins, vis_losses, home_score, vis_score))
    else:
        summ.append(first_line % (vis_city + " " + vis_name, vis_wins, vis_losses,
               home_city + " " + home_name, home_wins, home_losses, vis_score, home_score))
    k_best = get_best_players(entry["box_score"], k)
    for player_tup in k_best:
        summ.append(player_line % (player_tup))
    summ.append(last_line % (vis_name, home_name))
    return " ".join(summ)

def doit(inp_file, out_file):
    with codecs.open(inp_file, "r", "utf-8") as f:
        data = json.load(f)
    with codecs.open(out_file, "w+", "utf-8") as g:
        for thing in data:
            g.write("%s\n" % rule_gen(thing))

doit(sys.argv[1], sys.argv[2])
