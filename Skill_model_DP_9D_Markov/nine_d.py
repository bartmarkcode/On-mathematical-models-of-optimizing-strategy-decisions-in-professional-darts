import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import dartboard as onb
import evaluate_policy_and_aming_grid as epag

directory_of_action_set = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/Programms for MA/result_grid'
directory_of_score_prob = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/Programms for MA/Score_probabilities_grid'
my_directory ='C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/created_sub_tables'
# ==================================================================================================================
# Introducing dictionary that assigns each field on the board the respective numerical score (i.e. our function h(.)):

num_z = {"M0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5, "S6": 6, "S7": 7, "S8": 8, "S9": 9, "S10": 10, "S11": 11,
         "S12": 12, "S13": 13, "S14": 14, "S15": 15, "S16": 16, "S17": 17, "S18": 18, "S19": 19, "S20": 20, "SB25": 25,
         "D1": 2, "D2": 4, "D3": 6, "D4": 8, "D5": 10, "D6": 12, "D7": 14, "D8": 16, "D9": 18, "D10": 20, "D11": 22,
         "D12": 24, "D13": 26, "D14": 28, "D15": 30, "D16": 32, "D17": 34, "D18": 36, "D19": 38, "D20": 40, "DB25": 50,
         "T1": 3, "T2": 6, "T3": 9, "T4": 12, "T5": 15, "T6": 18, "T7": 21, "T8": 24, "T9": 27, "T10": 30, "T11": 33,
         "T12": 36, "T13": 39, "T14": 42, "T15": 45, "T16": 48, "T17": 51, "T18": 54, "T19": 57, "T20": 60}
# =====================================================================================================================
# create list of all existing fields on the board; exclude "M0" as no one would miss throw on purpose
# allowed_fields = ["S1","D1","T1","S2",...,"T20","SB25","DB25"]
allowed_fields = list(num_z.keys())
allowed_fields.remove("M0")
# create also a list just for all possible double fields:
# allowed_doubles = ["D1","D2",...,"D20","DB25"]
allowed_doubles = [k for k, l in num_z.items() if k.startswith("D")]
# ==================================================================================================================
# CREATE the skill set dictionary K of an information item:
# contains tuples (TR, p(TR)) e.g. (T20, 0.39) means that the success probability on T20 if the player aims at T20 is 39%
# 'al' represents 'allowed_fields'
def skill_dict(al, item):
    file_it = directory_of_action_set + '/v2_item{}'.format(item) + "_gaussin_prob_grid.pkl"
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
     prob_grid_bullscore] = epag.load_aiming_grid(file_it)

    dict = {}
    for i in al:
        dict[i] = 0
        n = 0
        if i == "DB25":
            for j in range(0, 81):
                xj, yj = aiming_grid[j][0] - 170, aiming_grid[j][1] - 170
                if onb.get_score_and_multiplier(xj, yj) == [50, 2]:
                    dict[i] += prob_grid_bullscore[j][1]
                    n += 1
            dict[i] /= n
        elif i == "SB25":
            for j in range(0, 81):
                xj, yj = aiming_grid[j][0] - 170, aiming_grid[j][1] - 170
                if onb.get_score_and_multiplier(xj, yj) != [50, 2]:
                    dict[i] += prob_grid_bullscore[j][0]
                    n += 1
            dict[i] /= n
        else:
            st = i[0]
            nr = int(i.removeprefix(st))
            for j in range(81, len(aiming_grid)):
                xj, yj = aiming_grid[j][0] - 170, aiming_grid[j][1] - 170
                if st == "S" and onb.get_score_and_multiplier(xj, yj) == [nr, 1]:
                    dict[i] += prob_grid_singlescore[j][nr - 1]
                    n += 1
                elif st == "D" and onb.get_score_and_multiplier(xj, yj) == [nr * 2, 2]:
                    dict[i] += prob_grid_doublescore[j][nr - 1]
                    n += 1
                elif st == "T" and onb.get_score_and_multiplier(xj, yj) == [nr * 3, 3]:
                    dict[i] += prob_grid_triplescore[j][nr - 1]
                    n += 1
            dict[i] /= n
    return dict

# print(skill_dict(allowed_fields, 4))
# ===================================================================================================================
# Create skill set using the whole 1mm grid instead of the reduced adapted action set:
# results don't deviate much:
def skill_dict_grid(al, item):
    item_filename = directory_of_score_prob + '/item{}_gaussin_prob_grid.pkl'.format(item)
    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = \
        epag.load_prob_grid(item_filename)

    dict = {}
    for i in al:
        dict[i] = 0
        n = 0
        for x_j in range(341):
            for y_j in range(341):
                if i == "DB25" and onb.get_score_and_multiplier(x_j - 170, y_j - 170) == [50, 2]:
                    dict[i] += prob_grid_bullscore[x_j, y_j, 1]
                    n += 1
                elif i == "SB25" and onb.get_score_and_multiplier(x_j - 170, y_j - 170) == [25, 1]:
                    dict[i] += prob_grid_bullscore[x_j, y_j, 0]
                    n += 1
                else:
                    if i != "SB25" and i != "DB25":
                        st = i[0]
                        nr = int(i.removeprefix(st))
                        if st == "S" and onb.get_score_and_multiplier(x_j - 170, y_j - 170) == [nr, 1]:
                            dict[i] += prob_grid_singlescore[x_j, y_j, nr - 1]
                            n += 1
                        elif st == "D" and onb.get_score_and_multiplier(x_j - 170, y_j - 170) == [nr * 2, 2]:
                            dict[i] += prob_grid_doublescore[x_j, y_j, nr - 1]
                            n += 1
                        elif st == "T" and onb.get_score_and_multiplier(x_j - 170, y_j - 170) == [nr * 3, 3]:
                            dict[i] += prob_grid_triplescore[x_j, y_j, nr - 1]
                            n += 1
        dict[i] /= n
    return dict

# print(skill_dict(allowed_fields,11))
# print(skill_dict_grid(allowed_fields,11))
# =============================================================================================================
# CREATE the skill maps for an information item:

def skill_heatmaps(item):
    skill = skill_dict(allowed_fields, item)

    # using 1mm grid:
    # skill = skill_dict_grid(allowed_fields,item)

    skill["-"] = 0
    segment = ["Single", "Double", "Triple"]
    number = [k for k in range(1, 21)] + [25]
    perc = pd.DataFrame(np.array(list(skill.values())).reshape((3, 21)), columns=number)
    sns.heatmap(perc, annot=True, yticklabels=segment, linewidths=2, linecolor='black', cmap="YlGnBu",
                cbar_kws={'label': 'Success Probability', "location": "right"})
    plt.show()

# skill_heatmaps(1)
# =================================================================================================================
# =================================================================================================================
# create some helper functions for the final 9D-algorithm
def track_prob(a, b):
    """
    :param a: List of fields/ the track, e.g. [T20,T20,D20]
    :param b: Dictionary of success probabilities i.e. the output of the function skill_dict()
    :return: Probability of successfully hitting the complete track 'a' without any misses given skill set 'b'.
    """

    c = 0
    for i in range(len(a)):
        if c == 0:
            c += b[a[i]]
        else:
            c = c * b[a[i]]
    return c

# =================================================================================================================
# check if the intersection set of two lists is non-empty.
# used to check if our track is feasible in the sense that it contains at least one double field.
def intersec(a, b):
    c = [t for t in a if t in b]
    if len(c) == 0:
        return False
    else:
        return True

# ===============================================================================================================
# If the double field is not at the end of a track 'ls', do so.
def double_finish(ls, dl):
    for j in range(len(ls) - 1):
        if ls[j] in dl:
            ls[j], ls[-1] = ls[-1], ls[j]
    return ls
# -----------------------------------------------------------------------------------------------------------------
def algo_9D(S, al, dl, skill, num_z, memo={}):
    """
    :param S: Current Score of a player
    :param al: List of all existing fields
    :param dl: List of all possible double fields (to finish a leg)
    :param skill: Dictionary of success probabilities for a given field, i.e. output from function skill_dict() (= skill set K)
    :param num_z: Dictionary that assigns each field on the board the respective numerical score, see above (= function h(.)=
    :param memo: Dictionary to store previous results (--> Dynamic programming)
    :return: An individual track of minimum length (with the highest success probability) to finish from a total score
             of S
    """
    # request already stored values:
    if S in memo:
        return memo[S]
    # otherwise:
    if S == 0:
        return []
    elif S < 0 or S == 1:
        return None

    best_track = None
    for s in al:
        if num_z[s] <= S:
            remainder = S - num_z[s]
            remainder_combi = algo_9D(remainder, al, dl, skill, num_z, memo)
            if remainder_combi is not None and (intersec(dl, remainder_combi) or s in dl):
                combination = remainder_combi + [s]
                if best_track is None or len(combination) < len(best_track):
                    best_track = combination
                    best_track.sort(reverse=True)
                if len(combination) == len(best_track) and track_prob(combination, skill) > track_prob(best_track,skill):
                    best_track = combination
                    best_track.sort(reverse=True)

    memo[S] = double_finish(best_track, dl)
    return memo[S]

# t = algo_9D(80,allowed_fields,allowed_doubles,skill_dict(allowed_fields,2),num_z)
# print(t)
# ==================================================================================================================
# Compute 9D-tracks for all total scores S=2,...,501:
def rec_list(item):
    dict = {}
    for s in range(2, 502):
        val = algo_9D(s, allowed_fields, allowed_doubles, skill_dict(allowed_fields, item), num_z, memo={})
        dict[s] = val
    #print(dict)
    return dict
# ---------------------------------------------------------------------------------------------------------------------
# Create all tracks for an information item and store them (and additional information gathered) in an excel file:
def create_track_table(item):
    recomendation = rec_list(item)
    numerical_dict = {}
    success_list = []
    strat_list = []
    value_list = []
    for suc in recomendation.keys():
        success_list.append(track_prob(recomendation[suc], skill_dict(allowed_fields, item)))
        first_target = recomendation[suc][0]
        strat_list.append(first_target)
        value_list.append(num_z[first_target])
        numerical_dict[suc] = []
        for i in recomendation[suc]:
            numerical_dict[suc].append(num_z[i])

    output = pd.DataFrame({"Score": list(recomendation.keys()),
                           "Track": list(recomendation.values()),
                           "Optimal strategy z*": strat_list,
                           "Track of numerical scores": list(numerical_dict.values()),
                           "Optimal numerical value s*": value_list,
                           "Success Probability": success_list})

    return output

def write_9D():
    DVdB19_9D = create_track_table(1)
    DVdB20_9D = create_track_table(2)
    DVdB21_9D = create_track_table(3)
    AL19_9D = create_track_table(4)
    AL20_9D = create_track_table(5)
    JdS20_9D = create_track_table(6)
    JdS21_9D = create_track_table(7)
    DvD20_9D = create_track_table(8)
    DvD21_9D = create_track_table(9)
    NA18_9D = create_track_table(10)
    NA19_9D = create_track_table(11)

    with pd.ExcelWriter(my_directory + '/9D_recommendations.xlsx') as writer:
        DVdB19_9D.to_excel(writer, sheet_name="DVdB19")
        DVdB20_9D.to_excel(writer, sheet_name="DVdB20")
        DVdB21_9D.to_excel(writer, sheet_name="DVdB21")
        AL19_9D.to_excel(writer, sheet_name="AL19")
        AL20_9D.to_excel(writer, sheet_name="AL20")
        JdS20_9D.to_excel(writer, sheet_name="JdS20")
        JdS21_9D.to_excel(writer, sheet_name="JdS21")
        DvD20_9D.to_excel(writer, sheet_name="DvD20")
        DvD21_9D.to_excel(writer, sheet_name="DvD21")
        NA18_9D.to_excel(writer, sheet_name="NA18")
        NA19_9D.to_excel(writer, sheet_name="NA19")

# write_9D()