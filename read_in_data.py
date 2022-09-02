import pandas as pd
import os
import numpy as np

# =======================================================================
# Import the data set:
my_directory = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit'
file_name = 'Barthdataextractcopy.xlsx'

os.chdir(my_directory)
datatable = pd.read_excel(file_name)
datatable.set_index('Index #', inplace=True)

# Remove unnecessary columns i.e. columns with information which we don't need for our task e.g. the referee of a game:
datatable.drop(['Unnamed: 93', 'Unnamed: 89', 'Unnamed: 85', 'Unnamed: 81', 'Unnamed: 77', 'Unnamed: 73',
                'Unnamed: 69', 'Unnamed: 65', 'Unnamed: 61', 'Unnamed: 57', 'Unnamed: 53', 'Unnamed: 49',
                'Unnamed: 45', 'Unnamed: 41', 'Unnamed: 37', 'Unnamed: 15', 'Referee'], axis=1, inplace=True)

# Rename several columns that were unnamed or to recognize them better:
datatable.rename(columns={'Big number (0=miss, 1=hit)': 'Big_number_1d',
                          'Unnamed: 91': 'Big_number_2d',
                          'Unnamed: 92': 'Big_number_3d',
                          'Unnamed: 88': 'not_T17_hit_on_T17_attempt_3d',
                          'Unnamed: 87': 'not_T17_hit_on_T17_attempt_2d',
                          'S17/2/3/T2/T3/Bounce hit on T17 attempt, darts 1/2/3': 'not_T17_hit_on_T17_attempt_1d',
                          'Unnamed: 84': 'T17_hit_on_T17_attempt_3d',
                          'Unnamed: 83': 'T17_hit_on_T17_attempt_2d',
                          'T17 hit on T17 attempt, darts 1/2/3': 'T17_hit_on_T17_attempt_1d',
                          'Unnamed: 80': 'not_T18_hit_on_T18_attempt_3d',
                          'Unnamed: 79': 'not_T18_hit_on_T18_attempt_2d',
                          'S18/1/4/T1/T4/Bounce hit on T18 attempt, darts 1/2/3': 'not_T18_hit_on_T18_attempt_1d',
                          'Unnamed: 76': 'T18_hit_on_T18_attempt_3d',
                          'Unnamed: 75': 'T18_hit_on_T18_attempt_2d',
                          'T18 hit on T18 attempt, darts 1/2/3': 'T18_hit_on_T18_attempt_1d',
                          'Unnamed: 72': 'not_T19_hit_on_T19_attempt_3d',
                          'Unnamed: 71': 'not_T19_hit_on_T19_attempt_2d',
                          '3/7/T3/T7/bounce hit on T19 attempt, darts 1/2/3': 'not_T19_hit_on_T19_attempt_1d',
                          'Unnamed: 68': 'S19_hit_on_T19_attempt_3d',
                          'Unnamed: 67': 'S19_hit_on_T19_attempt_2d',
                          'S19 hit on T19 attempt, darts 1/2/3': 'S19_hit_on_T19_attempt_1d',
                          'Unnamed: 64': 'T19_hit_on_T19_attempt_3d',
                          'Unnamed: 63': 'T19_hit_on_T19_attempt_2d',
                          'T19 hit on T19 attempt, darts 1/2/3': 'T19_hit_on_T19_attempt_1d',
                          'Unnamed: 60': 'not_T20_hit_on_T20_attempt_3d',
                          'Unnamed: 59': 'not_T20_hit_on_T20_attempt_2d',
                          '1/5/T1/T5/bounce hit on T20 attempt, darts 1/2/3': 'not_T20_hit_on_T20_attempt_1d',
                          'Unnamed: 56': 'S20_hit_on_T20_attempt_3d',
                          'Unnamed: 55': 'S20_hit_on_T20_attempt_2d',
                          'S20 hit on T20 attempt, darts 1/2/3': 'S20_hit_on_T20_attempt_1d',
                          'Unnamed: 52': 'T20_hit_on_T20_attempt_3d',
                          'Unnamed: 51': 'T20_hit_on_T20_attempt_2d',
                          'T20 hit on T20 attempt, darts 1/2/3': 'T20_hit_on_T20_attempt_1d',
                          'Unnamed: 21': 'Unsure',
                          "Sportradar input data": "sportradar_total",
                          "Unnamed: 24": "sportradar_points",
                          "Unnamed: 25": "sportradar_first",
                          "Unnamed: 26": "sportradar_second",
                          "Unnamed: 27": "sportradar_third",
                          }, inplace=True)
# ================================================================================================
# Adjust some cells that the information is consistent.
# For example, points thrown by a player in the 3 sportsradar columns should be registered uniformly
# in the same columns.
# Specifications regarding attempted segments (double, treble, etc) were changed to be uniform for later usage:

for z in range(len(datatable.index.tolist())):
    if datatable["sportradar_total"].values[z] == "Leg":
        datatable["sportradar_total"].values[z] = datatable["sportradar_first"].values[z]
        datatable["sportradar_points"].values[z] = datatable["sportradar_second"].values[z]
        datatable["sportradar_first"].values[z] = datatable["sportradar_third"].values[z]
        datatable["sportradar_second"].values[z] = datatable["Unnamed: 28"].values[z]
        datatable["sportradar_third"].values[z] = datatable["Unnamed: 29"].values[z]
        datatable["Unnamed: 28"].values[z] = None
        datatable["Unnamed: 29"].values[z] = None
    if datatable["1st dart attempt (T/S/D/DD/Bull)"].values[z] == "DD":
        datatable["1st dart attempt (T/S/D/DD/Bull)"].values[z] = "D"
    if datatable["2nd dart attempt (T/S/D/DD/Bull)"].values[z] == "DD":
        datatable["2nd dart attempt (T/S/D/DD/Bull)"].values[z] = "D"
    if datatable["3rd dart attempt (T/S/D/DD/Bull)"].values[z] == "DD":
        datatable["3rd dart attempt (T/S/D/DD/Bull)"].values[z] = "D"
    if datatable["1st dart attempt (T/S/D/DD/Bull)"].values[z] == "Bull":
        datatable["1st dart attempt (T/S/D/DD/Bull)"].values[z] = "SB"
    if datatable["2nd dart attempt (T/S/D/DD/Bull)"].values[z] == "Bull":
        datatable["2nd dart attempt (T/S/D/DD/Bull)"].values[z] = "SB"
    if datatable["3rd dart attempt (T/S/D/DD/Bull)"].values[z] == "Bull":
        datatable["3rd dart attempt (T/S/D/DD/Bull)"].values[z] = "SB"
# ================================================================================
# Constructing separate tables for each player:
DVdB_table = datatable[datatable.Player == 'Dimitri Van den Bergh']
AL_table = datatable[datatable.Player == 'Adrian Lewis']
JdS_table = datatable[datatable.Player == 'Jose de Sousa']
DvD_table = datatable[datatable.Player == 'Dirk van Duijvenbode']
NA_table = datatable[datatable.Player == 'Nathan Aspinall']
# ================================================================================
# Player-by-year-tables (i.e. creating separate tables for each information item):
DVdB_table19 = DVdB_table[pd.DatetimeIndex(DVdB_table['Date']).year == 2019]
DVdB_table20 = DVdB_table[pd.DatetimeIndex(DVdB_table['Date']).year == 2020]
DVdB_table21 = DVdB_table[pd.DatetimeIndex(DVdB_table['Date']).year == 2021]
AL_table19 = AL_table[pd.DatetimeIndex(AL_table['Date']).year == 2019]
AL_table20 = AL_table[pd.DatetimeIndex(AL_table['Date']).year == 2020]
JdS_table20 = JdS_table[pd.DatetimeIndex(JdS_table['Date']).year == 2020]
JdS_table21 = JdS_table[pd.DatetimeIndex(JdS_table['Date']).year == 2021]
DvD_table20 = DvD_table[pd.DatetimeIndex(DvD_table['Date']).year == 2020]
DvD_table21 = DvD_table[pd.DatetimeIndex(DvD_table['Date']).year == 2021]
NA_table18 = NA_table[pd.DatetimeIndex(NA_table['Date']).year == 2018]
NA_table19 = NA_table[pd.DatetimeIndex(NA_table['Date']).year == 2019]
# ==================================================================================================================
# We also stored the tables for each information items (see the folder 'Player-specific Tables:
player_specific_table_directory = my_directory + '/created_sub_tables/Player_specific_tables'

DVdB_table19.to_excel(player_specific_table_directory + '/vandenBergh19.xlsx')
DVdB_table20.to_excel(player_specific_table_directory + '/vandenBergh20.xlsx')
DVdB_table21.to_excel(player_specific_table_directory + '/vandenBergh21.xlsx')
AL_table19.to_excel(player_specific_table_directory + '/Lewis19.xlsx')
AL_table20.to_excel(player_specific_table_directory + '/Lewis20.xlsx')
JdS_table20.to_excel(player_specific_table_directory + '/deSousa20.xlsx')
JdS_table21.to_excel(player_specific_table_directory + '/deSousa21.xlsx')
DvD_table20.to_excel(player_specific_table_directory + '/vanDuijvenbode20.xlsx')
DvD_table21.to_excel(player_specific_table_directory + '/vanDuijvenbode21.xlsx')
NA_table19.to_excel(player_specific_table_directory + '/Aspinall19.xlsx')
NA_table18.to_excel(player_specific_table_directory + '/Aspinall18.xlsx')
# =====================================================================================================================
# =====================================================================================================================
# PRESENTING DATA AS (TR,z,n) combinations:
# Introducing dictionary that assigns each field on the board the respective numerical score (i.e. our function h(.)):

num_z = {"M0": 0, "S1": 1, "D1": 2, "T1": 3, "S2": 2, "D2": 4, "T2": 6, "S3": 3, "D3": 6, "T3": 9, "S4": 4, "D4": 8,
         "T4": 12,
         "S5": 5, "D5": 10, "T5": 15, "S6": 6, "D6": 12, "T6": 18, "S7": 7, "D7": 14, "T7": 21, "S8": 8, "D8": 16,
         "T8": 24,
         "S9": 9, "D9": 18, "T9": 27, "S10": 10, "D10": 20, "T10": 30, "S11": 11, "D11": 22, "T11": 33, "S12": 12,
         "D12": 24,
         "T12": 36, "S13": 13, "D13": 26, "T13": 39, "S14": 14, "D14": 28, "T14": 42, "S15": 15, "D15": 30, "T15": 45,
         "S16": 16, "D16": 32, "T16": 48, "S17": 17, "D17": 34, "T17": 51, "S18": 18, "D18": 36, "T18": 54, "S19": 19,
         "D19": 38, "T19": 57, "S20": 20, "D20": 40, "T20": 60, "SB25": 25, "DB25": 50}
# Introducing dictionary that lists all possible z scores a (professional) player may hits when targeting the respective
# target region R.
# For example, when targeting D20, there are only 7 different possible outcomes a professional player will hit
# [D20, S20, D5, S5, S1, D1, M0]
# All others are irrelevant because professional players have a certain accuracy:
poss_z = {"T20": ["S5", "T5", "S20", "T20", "S1", "T1"],
          "T19": ["T19", "S19", "T3", "S3", "S7", "T7"],
          "T18": ["T18", "S18", "S4", "T4", "S1", "T1"],
          "T17": ["T17", "S17", "S3", "T3", "S2", "T2"],
          "D1": ["D1", "S1", "D20", "S20", "D18", "S20", "M0"],
          "D2": ["D2", "S2", "S15", "S17", "D17", "D15", "M0"],
          "D3": ["D3", "S3", "D17", "S17", "D19", "S19", "M0"],
          "D4": ["D4", "S4", "D18", "S18", "S13", "D13", "M0"],
          "D5": ["D5", "S5", "S12", "D12", "S20", "D20", "M0"],
          "D6": ["D6", "S6", "S13", "D13", "S10", "D10", "M0"],
          "D7": ["D7", "S7", "D19", "S19", "D16", "S16", "M0"],
          "D8": ["D8", "S8", "D11", "S11", "S16", "D16", "M0"],
          "D9": ["D9", "S9", "D12", "D14", "S12", "S14", "M0"],
          "D10": ["D10", "S10", "D6", "S6", "D15", "S15", "M0"],
          "D11": ["D11", "S11", "D8", "S8", "D14", "S14", "M0"],
          "D12": ["D12", "S12", "S9", "D9", "S5", "D5", "M0"],
          "D13": ["D13", "S13", "D4", "S4", "D6", "S6", "M0"],
          "D14": ["D14", "S14", "D9", "S9", "D11", "S11", "M0"],
          "D15": ["D15", "S15", "S2", "D2", "D10", "S10", "M0"],
          "D16": ["D16", "S8", "S16", "D8", "D7", "S7", "M0"],
          "D17": ["D17", "S17", "S2", "D2", "D3", "S3", "M0"],
          "D18": ["D18", "S18", "S1", "D1", "D4", "S4", "M0"],
          "D19": ["D19", "S19", "S7", "D7", "S3", "D3", "M0"],
          "D20": ["D20", "S20", "D5", "S5", "S1", "D1", "M0"],
          "DB25": ["DB25", "SB25", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12",
                   "S13", "S14", "S15", "S16", "S17", "S18", "S19", "S20"]}

# create dictionary that helps iterating through the table with respect to throw 1/2/3 of a turn:
attempt_dict = {"one": ["1st dart attempt (T/S/D/DD/Bull)",
                        "1st dart attempt (segment 1-20)",
                        "sportradar_first"],
                "two": ["2nd dart attempt (T/S/D/DD/Bull)", "2nd dart attempt (segment 1-20)", "sportradar_second"],
                "three": ["3rd dart attempt (T/S/D/DD/Bull)", "3rd dart attempt (segment 1-20)", "sportradar_third"]}


# helper function for the function combi_file():
# get a list of all keys of a dictionary 'd' with given value 'val'
# i.e. using dictionary num_z, return a list with all possible fields that could score a value of 'val':
def poss_targets(dic, val):
    targets = [k for k, l in dic.items() if val == l]
    return targets


def combi_file(table, num_z, poss_z, attempt_dict, control_d):
    """
    :param table: data(-table) of the information item
    :param num_z: dictionary constructed above
    :param poss_z: dictionary constructed above
    :param attempt_dict: dictionary constructed above
    :param control_d: column of 'table' that represents that the player finished the leg (i.e. with a double field)
                    in our case we check if the column  "Player finishing score" is equal to 0.
    :return: List of tuples of the form (TR,z,n), i.e. (TR = target region, z = score hit while aiming at TR,
            n = number of times (TR,z) occurs)
    """
    dic = {}
    combi_list = []
    # Go through throws 1/2/3 of a turn
    for u in attempt_dict:
        # Go through all turns i.e. each corresponding column in the table
        for cell in range(len(table.index.tolist())):
            # Skip cells that are empty (happens when a player only throws one or two darts in a turn)
            if pd.isna(table[attempt_dict[u][1]].values[cell]):
                continue
            elif pd.isna(table[attempt_dict[u][2]].values[cell]):
                continue
            # Identify the target region TR as well as the numerical value 'score_val' thrown
            else:
                TR = str(table[attempt_dict[u][0]].values[cell]) + str(int(table[attempt_dict[u][1]].values[cell]))
                if TR == "D25":
                    TR = "DB25"
                score_val = int(table[attempt_dict[u][2]].values[cell])

                # We still have to assign a thrown score to the correct field, for example when the player
                # targeted TR = T20 and a achieved a score of 20 it should give us z = S20  and not D10.
                check_list = poss_targets(num_z, score_val)

                for t in check_list:
                    # special case of score 16 has to be treated separately
                    if TR == "D8" and score_val == 16 and table[control_d].values[cell] == 0:
                        pair1 = (TR, "D8")
                        if pair1 not in dic:
                            dic[pair1] = 1
                        else:
                            dic[pair1] = dic[pair1] + 1
                    if TR == "D8" and score_val == 16 and table[control_d].values[cell] != 0:
                        pair1 = (TR, "S16")
                        if pair1 not in dic:
                            dic[pair1] = 1
                        else:
                            dic[pair1] = dic[pair1] + 1
                    if TR == "D16" and score_val == 16:
                        pair1 = (TR, "S16")
                        if pair1 not in dic:
                            dic[pair1] = 1
                        else:
                            dic[pair1] = dic[pair1] + 1
                        break

                    # for all other TR's check the neighbourhood and assign the correct z scores:
                    if TR in list(poss_z.keys()) and t in poss_z[TR]:
                        pair1 = (TR, t)
                        if pair1 not in dic:
                            dic[pair1] = 1
                        else:
                            dic[pair1] = dic[pair1] + 1
                        break
                    else:
                        continue
    # Transform the created dictionary (which have tuples as keys) to a list
    # containing all different combinations (TR,z,n):
    for o in range(len(dic)):
        a, b = list(dic.keys())[o][0], list(dic.keys())[o][1]
        combi_list += [(a, b, dic[(a, b)])]

    return combi_list


# ============================================================================================================
# Function that translates the combination list to a pandas dataframe such that we can create excel sheets
# for each information item (i.e. each player in each year) containing all combinations
def ls_to_df(ls):
    one, two, three = [], [], []
    for a in range(len(ls)):
        one += [ls[a][0]]
        two += [ls[a][1]]
        three += [ls[a][2]]
    d = {"TR": one, "z": two, "n": three}
    df = pd.DataFrame(d)
    return df


cf_DVdB19 = ls_to_df(combi_file(DVdB_table19, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_DVdB20 = ls_to_df(combi_file(DVdB_table20, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_DVdB21 = ls_to_df(combi_file(DVdB_table21, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_AL19 = ls_to_df(combi_file(AL_table19, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_AL20 = ls_to_df(combi_file(AL_table20, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_JdS20 = ls_to_df(combi_file(JdS_table20, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_JdS21 = ls_to_df(combi_file(JdS_table21, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_DvD20 = ls_to_df(combi_file(DvD_table20, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_DvD21 = ls_to_df(combi_file(DvD_table21, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_NA18 = ls_to_df(combi_file(NA_table18, num_z, poss_z, attempt_dict, "Player finishing score"))
cf_NA19 = ls_to_df(combi_file(NA_table19, num_z, poss_z, attempt_dict, "Player finishing score"))

with pd.ExcelWriter(my_directory + '/created_sub_tables/combinations_all_players.xlsx') as writer:
    cf_DVdB19.to_excel(writer, sheet_name="DVdB19", index=False)
    cf_DVdB20.to_excel(writer, sheet_name="DVdB20", index=False)
    cf_DVdB21.to_excel(writer, sheet_name="DVdB21", index=False)
    cf_AL19.to_excel(writer, sheet_name="AL19", index=False)
    cf_AL20.to_excel(writer, sheet_name="AL20", index=False)
    cf_JdS20.to_excel(writer, sheet_name="JdS20", index=False)
    cf_JdS21.to_excel(writer, sheet_name="JdS21", index=False)
    cf_DvD20.to_excel(writer, sheet_name="DvD20", index=False)
    cf_DvD21.to_excel(writer, sheet_name="DvD21", index=False)
    cf_NA18.to_excel(writer, sheet_name="NA18", index=False)
    cf_NA19.to_excel(writer, sheet_name="NA19", index=False)
# =============================================================================================================
# =============================================================================================================
# SEVERAL GAME INFORMATION:
# We extract additional statistics for a better analysis of a player's performance:
# -------------------------------------------------------------------------------------------------------------
#                                   Number of turns each player had (in each year):
nr_turns_DVdB_total = len(DVdB_table)
nr_turns_DVdB_19 = len(DVdB_table19)
nr_turns_DVdB_20 = len(DvD_table20)
nr_turns_DVdB_21 = len(DVdB_table21)

nr_turns_AL_total = len(AL_table)
nr_turns_AL_19 = len(AL_table19)
nr_turns_AL_20 = len(AL_table20)

nr_turns_JdS_total = len(JdS_table)
nr_turns_JdS_20 = len(JdS_table20)
nr_turns_JdS_21 = len(JdS_table21)

nr_turns_DvD_total = len(DvD_table)
nr_turns_DvD_20 = len(DvD_table20)
nr_turns_DvD_21 = len(DvD_table21)

nr_turns_NA_total = len(NA_table)
nr_turns_NA_18 = len(NA_table18)
nr_turns_NA_19 = len(NA_table19)
# ---------------------------------------------------------------------------------------------------------------
#                               How many throws each player made (in each year):
#  !!! We first mixed up the "number of turns" with the "number of throws". We calculated again the total number of
#  throws in the end when we compared our models (i.e. in the file 'comparison.py')
#  We then just filled in the values here by hand !!!
nr_throws_DVdB_19 = 5259
nr_throws_DVdB_20 = 6199
nr_throws_DVdB_21 = 10950
nr_throws_DVdB_total = 22408
nr_throws_AL_19 = 6741
nr_throws_AL_20 = 2865
nr_throws_AL_total = 9606
nr_throws_JdS_20 = 6987
nr_throws_JdS_21 = 13856
nr_throws_JdS_total = 20843
nr_throws_DvD_20 = 5020
nr_throws_DvD_21 = 6386
nr_throws_DvD_total = 11406
nr_throws_NA_18 = 9505
nr_throws_NA_19 = 2886
nr_throws_NA_total = 12391
# ---------------------------------------------------------------------------------------------------------------
#                                   How many games a player played (each year):

nr_games_DVdB_19 = len(DVdB_table19[["Date","Opponent"]].drop_duplicates())
nr_games_DVdB_20 = len(DVdB_table20[["Date","Opponent"]].drop_duplicates())
nr_games_DVdB_21 = len(DVdB_table21[["Date","Opponent"]].drop_duplicates())
nr_games_DVdB_total = nr_games_DVdB_20 +nr_games_DVdB_21 +nr_games_DVdB_19

nr_games_AL_19 = len(AL_table19[["Date","Opponent"]].drop_duplicates())
nr_games_AL_20 = len(AL_table20[["Date","Opponent"]].drop_duplicates())
nr_games_AL_total = nr_games_AL_20 + nr_games_AL_19

nr_games_JdS_20 = len(JdS_table20[["Date","Opponent"]].drop_duplicates())
nr_games_JdS_21 = len(JdS_table21[["Date","Opponent"]].drop_duplicates())
nr_games_JdS_total = nr_games_JdS_21+nr_games_JdS_20

nr_games_DvD_20 = len(DvD_table20[["Date","Opponent"]].drop_duplicates())
nr_games_DvD_21 = len(DvD_table21[["Date","Opponent"]].drop_duplicates())
nr_games_DvD_total = nr_games_DvD_21 + nr_games_DvD_20

nr_games_NA_18 = len(NA_table18[["Date","Opponent"]].drop_duplicates())
nr_games_NA_19 = len(NA_table19[["Date","Opponent"]].drop_duplicates())
nr_games_NA_total = nr_games_NA_19 + nr_games_NA_18
# --------------------------------------------------------------------------------------------------------------
#                               How many different tournaments a player played (in each year):
DVdB_only_column_tournament = DVdB_table["Tournament"]
nr_tournaments_DVdB_total = len(DVdB_only_column_tournament.drop_duplicates())
nr_tournaments_DVdB_19 = len(DVdB_table19["Tournament"].drop_duplicates())
nr_tournaments_DVdB_20 = len(DVdB_table20["Tournament"].drop_duplicates())
nr_tournaments_DVdB_21 = len(DVdB_table21["Tournament"].drop_duplicates())

nr_tournaments_AL_total = len(AL_table["Tournament"].drop_duplicates())
nr_tournaments_AL_19 = len(AL_table19["Tournament"].drop_duplicates())
nr_tournaments_AL_20 = len(AL_table20["Tournament"].drop_duplicates())

nr_tournaments_JdS_total = len(JdS_table["Tournament"].drop_duplicates())
nr_tournaments_JdS_20 = len(JdS_table20["Tournament"].drop_duplicates())
nr_tournaments_JdS_21 = len(JdS_table21["Tournament"].drop_duplicates())

nr_tournaments_DvD_total = len(DvD_table["Tournament"].drop_duplicates())
nr_tournaments_DvD_20 = len(DvD_table20["Tournament"].drop_duplicates())
nr_tournaments_DvD_21 = len(DvD_table21["Tournament"].drop_duplicates())

nr_tournaments_NA_total = len(NA_table["Tournament"].drop_duplicates())
nr_tournaments_NA_18 = len(NA_table18["Tournament"].drop_duplicates())
nr_tournaments_NA_19 = len(NA_table19["Tournament"].drop_duplicates())
# ----------------------------------------------------------------------------------------------------------------
#                           average score of (dart1/2/3/all3) of each information item:

def avg_score(table, col):
    player_c = 0
    for cell in range(len(table.index.tolist())):
        if pd.isna(table[col].values[cell]):
            continue
        else:
            player_c += table[col].values[cell]

    return player_c / len(table.index.tolist())


Avg_score_dart1_DVdB_19 = avg_score(DVdB_table19, "sportradar_first")
Avg_score_dart2_DVdB_19 = avg_score(DVdB_table19, "sportradar_second")
Avg_score_dart3_DVdB_19 = avg_score(DVdB_table19, "sportradar_third")
Avg_score_all_DVdB_19 = (Avg_score_dart1_DVdB_19 + Avg_score_dart2_DVdB_19 + Avg_score_dart3_DVdB_19)

Avg_score_dart1_DVdB_20 = avg_score(DVdB_table20, "sportradar_first")
Avg_score_dart2_DVdB_20 = avg_score(DVdB_table20, "sportradar_second")
Avg_score_dart3_DVdB_20 = avg_score(DVdB_table20, "sportradar_third")
Avg_score_all_DVdB_20 = (Avg_score_dart1_DVdB_20 + Avg_score_dart2_DVdB_20 + Avg_score_dart3_DVdB_20)

Avg_score_dart1_DVdB_21 = avg_score(DVdB_table21, "sportradar_first")
Avg_score_dart2_DVdB_21 = avg_score(DVdB_table21, "sportradar_second")
Avg_score_dart3_DVdB_21 = avg_score(DVdB_table21, "sportradar_third")
Avg_score_all_DVdB_21 = (Avg_score_dart1_DVdB_21 + Avg_score_dart2_DVdB_21 + Avg_score_dart3_DVdB_21)

Avg_score_dart1_AL_19 = avg_score(AL_table19, "sportradar_first")
Avg_score_dart2_AL_19 = avg_score(AL_table19, "sportradar_second")
Avg_score_dart3_AL_19 = avg_score(AL_table19, "sportradar_third")
Avg_score_all_AL_19 = (Avg_score_dart1_AL_19 + Avg_score_dart2_AL_19 + Avg_score_dart3_AL_19)

Avg_score_dart1_AL_20 = avg_score(AL_table20, "sportradar_first")
Avg_score_dart2_AL_20 = avg_score(AL_table20, "sportradar_second")
Avg_score_dart3_AL_20 = avg_score(AL_table20, "sportradar_third")
Avg_score_all_AL_20 = (Avg_score_dart1_AL_20 + Avg_score_dart2_AL_20 + Avg_score_dart3_AL_20)

Avg_score_dart1_JdS_20 = avg_score(JdS_table20, "sportradar_first")
Avg_score_dart2_JdS_20 = avg_score(JdS_table20, "sportradar_second")
Avg_score_dart3_JdS_20 = avg_score(JdS_table20, "sportradar_third")
Avg_score_all_JdS_20 = (Avg_score_dart1_JdS_20 + Avg_score_dart2_JdS_20 + Avg_score_dart3_JdS_20)

Avg_score_dart1_JdS_21 = avg_score(JdS_table21, "sportradar_first")
Avg_score_dart2_JdS_21 = avg_score(JdS_table21, "sportradar_second")
Avg_score_dart3_JdS_21 = avg_score(JdS_table21, "sportradar_third")
Avg_score_all_JdS_21 = (Avg_score_dart1_JdS_21 + Avg_score_dart2_JdS_21 + Avg_score_dart3_JdS_21)

Avg_score_dart1_DvD_20 = avg_score(DvD_table20, "sportradar_first")
Avg_score_dart2_DvD_20 = avg_score(DvD_table20, "sportradar_second")
Avg_score_dart3_DvD_20 = avg_score(DvD_table20, "sportradar_third")
Avg_score_all_DvD_20 = (Avg_score_dart1_DvD_20 + Avg_score_dart2_DvD_20 + Avg_score_dart3_DvD_20)

Avg_score_dart1_DvD_21 = avg_score(DvD_table21, "sportradar_first")
Avg_score_dart2_DvD_21 = avg_score(DvD_table21, "sportradar_second")
Avg_score_dart3_DvD_21 = avg_score(DvD_table21, "sportradar_third")
Avg_score_all_DvD_21 = (Avg_score_dart1_DvD_21 + Avg_score_dart2_DvD_21 + Avg_score_dart3_DvD_21)

Avg_score_dart1_NA_18 = avg_score(NA_table18, "sportradar_first")
Avg_score_dart2_NA_18 = avg_score(NA_table18, "sportradar_second")
Avg_score_dart3_NA_18 = avg_score(NA_table18, "sportradar_third")
Avg_score_all_NA_18 = (Avg_score_dart1_NA_18 + Avg_score_dart2_NA_18 + Avg_score_dart3_NA_18)

Avg_score_dart1_NA_19 = avg_score(NA_table19, "sportradar_first")
Avg_score_dart2_NA_19 = avg_score(NA_table19, "sportradar_second")
Avg_score_dart3_NA_19 = avg_score(NA_table19, "sportradar_third")
Avg_score_all_NA_19 = (Avg_score_dart1_NA_19 + Avg_score_dart2_NA_19 + Avg_score_dart3_NA_19)

# -----------------------------------------------------------------------------------------------------------------
#                                   Average number of darts needed to finish a leg:

leg_darts_DVdB_total = DVdB_table[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_DVdB_19 = DVdB_table19[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_DVdB_20 = DVdB_table20[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_DVdB_21 = DVdB_table21[["Date", "Winning leg darts"]].drop_duplicates()

leg_darts_AL_total = AL_table[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_AL_19 = AL_table19[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_AL_20 = AL_table20[["Date", "Winning leg darts"]].drop_duplicates()

leg_darts_JdS_total = JdS_table[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_JdS_20 = JdS_table20[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_JdS_21 = JdS_table21[["Date", "Winning leg darts"]].drop_duplicates()

leg_darts_DvD_total = DvD_table[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_DvD_20 = DvD_table20[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_DvD_21 = DvD_table21[["Date", "Winning leg darts"]].drop_duplicates()

leg_darts_NA_total = NA_table[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_NA_18 = NA_table18[["Date", "Winning leg darts"]].drop_duplicates()
leg_darts_NA_19 = NA_table19[["Date", "Winning leg darts"]].drop_duplicates()


Avg_leg_darts_DVdB_total = avg_score(leg_darts_DVdB_total, "Winning leg darts")
Avg_leg_darts_DVdB_19 = avg_score(leg_darts_DVdB_19, "Winning leg darts")
Avg_leg_darts_DVdB_20 = avg_score(leg_darts_DVdB_20, "Winning leg darts")
Avg_leg_darts_DVdB_21 = avg_score(leg_darts_DVdB_21, "Winning leg darts")

Avg_leg_darts_AL_total = avg_score(leg_darts_AL_total, "Winning leg darts")
Avg_leg_darts_AL_19 = avg_score(leg_darts_AL_19, "Winning leg darts")
Avg_leg_darts_AL_20 = avg_score(leg_darts_AL_20, "Winning leg darts")

Avg_leg_darts_JdS_total = avg_score(leg_darts_JdS_total, "Winning leg darts")
Avg_leg_darts_JdS_20 = avg_score(leg_darts_JdS_20, "Winning leg darts")
Avg_leg_darts_JdS_21 = avg_score(leg_darts_JdS_21, "Winning leg darts")

Avg_leg_darts_DvD_total = avg_score(leg_darts_DvD_total, "Winning leg darts")
Avg_leg_darts_DvD_20 = avg_score(leg_darts_DvD_20, "Winning leg darts")
Avg_leg_darts_DvD_21 = avg_score(leg_darts_DvD_21, "Winning leg darts")

Avg_leg_darts_NA_total = avg_score(leg_darts_NA_total, "Winning leg darts")
Avg_leg_darts_NA_18 = avg_score(leg_darts_NA_18, "Winning leg darts")
Avg_leg_darts_NA_19 = avg_score(leg_darts_NA_19, "Winning leg darts")
# ---------------------------------------------------------------------------------------------------------------
#                           Average remaining score of opponent when player finished the leg:

def rem_avg_score_opp(table, col_player, col_opp):
    counter = 0
    player_c = 0
    for cell in range(len(table.index.tolist())):
        if table[col_player].values[cell] == 0:
            player_c += table[col_opp].values[cell]
            counter += 1
        else:
            continue

    return player_c / counter

Rem_avg_score_DVdB_total = rem_avg_score_opp(DVdB_table, "Player finishing score", "Opponent score")
Rem_avg_score_DVdB_19 = rem_avg_score_opp(DVdB_table19, "Player finishing score", "Opponent score")
Rem_avg_score_DVdB_20 = rem_avg_score_opp(DVdB_table20, "Player finishing score", "Opponent score")
Rem_avg_score_DVdB_21 = rem_avg_score_opp(DVdB_table21, "Player finishing score", "Opponent score")
Rem_avg_score_AL_total = rem_avg_score_opp(AL_table, "Player finishing score", "Opponent score")
Rem_avg_score_AL_19 = rem_avg_score_opp(AL_table19, "Player finishing score", "Opponent score")
Rem_avg_score_AL_20 = rem_avg_score_opp(AL_table20, "Player finishing score", "Opponent score")
Rem_avg_score_JdS_total = rem_avg_score_opp(JdS_table, "Player finishing score", "Opponent score")
Rem_avg_score_JdS_20 = rem_avg_score_opp(JdS_table20, "Player finishing score", "Opponent score")
Rem_avg_score_JdS_21 = rem_avg_score_opp(JdS_table21, "Player finishing score", "Opponent score")
Rem_avg_score_DvD_total = rem_avg_score_opp(DvD_table, "Player finishing score", "Opponent score")
Rem_avg_score_DvD_20 = rem_avg_score_opp(DvD_table20, "Player finishing score", "Opponent score")
Rem_avg_score_DvD_21 = rem_avg_score_opp(DvD_table21, "Player finishing score", "Opponent score")
Rem_avg_score_NA_total = rem_avg_score_opp( NA_table, "Player finishing score", "Opponent score")
Rem_avg_score_NA_18 = rem_avg_score_opp(NA_table18, "Player finishing score", "Opponent score")
Rem_avg_score_NA_19 = rem_avg_score_opp(NA_table19, "Player finishing score", "Opponent score")
# -------------------------------------------------------------------------------------------------------------
# CREATING excel file with all the prepared information:
information = pd.DataFrame({"Player": ["van den Bergh", "van den Bergh", "van den Bergh", "van den Bergh",
                                       "Lewis", "Lewis", "Lewis",
                                       "de Sousa", "de Sousa", "de Sousa",
                                       "van Duijvenbode", "van Duijvenbode", "van Duijvenbode",
                                       "Aspinall", "Aspinall", "Aspinall"],
                            "Years": ["2019,2020,2021", "2019", "2020", "2021",
                                      "2019,2020", "2019", "2020",
                                      "2020,2021", "2020", "2021",
                                      "2020,2021", "2020", "2021",
                                      "2018,2019", "2018", "2019"],
                            "Total nr. of turns": [nr_turns_DVdB_total, nr_turns_DVdB_19, nr_turns_DVdB_20, nr_turns_DVdB_21,
                                                   nr_turns_AL_total, nr_turns_AL_19, nr_turns_AL_20,
                                                   nr_turns_JdS_total, nr_turns_JdS_20, nr_turns_JdS_21,
                                                   nr_turns_DvD_total, nr_turns_DvD_20, nr_turns_DvD_21,
                                                   nr_turns_NA_total, nr_turns_NA_18, nr_turns_NA_19],
                            "Total nr. of throws": [nr_throws_DVdB_total, nr_throws_DVdB_19, nr_throws_DVdB_20,
                                                    nr_throws_DVdB_21,
                                                    nr_throws_AL_total, nr_throws_AL_19, nr_throws_AL_20,
                                                    nr_throws_JdS_total, nr_throws_JdS_20, nr_throws_JdS_21,
                                                    nr_throws_DvD_total, nr_throws_DvD_20, nr_throws_DvD_21,
                                                    nr_throws_NA_total, nr_throws_NA_18, nr_throws_NA_19],
                            "Total nr. of games played": [nr_games_DVdB_total, nr_games_DVdB_19, nr_games_DVdB_20,
                                                          nr_games_DVdB_21,
                                                          nr_games_AL_total, nr_games_AL_19, nr_games_AL_20,
                                                          nr_games_JdS_total, nr_games_JdS_20, nr_games_JdS_21,
                                                          nr_games_DvD_total, nr_games_DvD_20, nr_games_DvD_21,
                                                          nr_games_NA_total, nr_games_NA_18, nr_games_NA_19],
                            "Total nr. of tournaments played": [nr_tournaments_DVdB_total, nr_tournaments_DVdB_19,
                                                                nr_tournaments_DVdB_20, nr_tournaments_DVdB_21,
                                                                nr_tournaments_AL_total, nr_tournaments_AL_19,
                                                                nr_tournaments_AL_20,
                                                                nr_tournaments_JdS_total, nr_tournaments_JdS_20,
                                                                nr_tournaments_JdS_21,
                                                                nr_tournaments_DvD_total, nr_tournaments_DvD_20,
                                                                nr_tournaments_DvD_21,
                                                                nr_tournaments_NA_total, nr_tournaments_NA_18,
                                                                nr_tournaments_NA_19],
                            "Average Score with all 3 darts": [np.nan, Avg_score_all_DVdB_19, Avg_score_all_DVdB_20,
                                                               Avg_score_all_DVdB_21,
                                                               np.nan, Avg_score_all_AL_19, Avg_score_all_AL_20,
                                                               np.nan, Avg_score_all_JdS_20, Avg_score_all_JdS_21,
                                                               np.nan, Avg_score_all_DvD_20, Avg_score_all_DvD_21,
                                                               np.nan, Avg_score_all_NA_18, Avg_score_all_NA_19],
                            "Average Score with first dart": [np.nan, Avg_score_dart1_DVdB_19, Avg_score_dart1_DVdB_20,
                                                              Avg_score_dart1_DVdB_21,
                                                              np.nan, Avg_score_dart1_AL_19, Avg_score_dart1_AL_20,
                                                              np.nan, Avg_score_dart1_JdS_20, Avg_score_dart1_JdS_21,
                                                              np.nan, Avg_score_dart1_DvD_20, Avg_score_dart1_DvD_21,
                                                              np.nan, Avg_score_dart1_NA_18, Avg_score_dart1_NA_19],
                            "Average Score with second dart": [np.nan, Avg_score_dart2_DVdB_19, Avg_score_dart2_DVdB_20,
                                                               Avg_score_dart2_DVdB_21,
                                                               np.nan, Avg_score_dart2_AL_19, Avg_score_dart2_AL_20,
                                                               np.nan, Avg_score_dart2_JdS_20, Avg_score_dart2_JdS_21,
                                                               np.nan, Avg_score_dart2_DvD_20, Avg_score_dart2_DvD_21,
                                                               np.nan, Avg_score_dart2_NA_18, Avg_score_dart2_NA_19],
                            "Average Score with third dart": [np.nan, Avg_score_dart3_DVdB_19, Avg_score_dart3_DVdB_20,
                                                              Avg_score_dart3_DVdB_21,
                                                              np.nan, Avg_score_dart3_AL_19, Avg_score_dart3_AL_20,
                                                              np.nan, Avg_score_dart3_JdS_20, Avg_score_dart3_JdS_21,
                                                              np.nan, Avg_score_dart3_DvD_20, Avg_score_dart3_DvD_21,
                                                              np.nan, Avg_score_dart3_NA_18, Avg_score_dart3_NA_19],
                            "Average nr. of darts to finish a leg": [Avg_leg_darts_DVdB_total, Avg_leg_darts_DVdB_19,
                                                                     Avg_leg_darts_DVdB_20, Avg_leg_darts_DVdB_21,
                                                                     Avg_leg_darts_AL_total, Avg_leg_darts_AL_19,
                                                                     Avg_leg_darts_AL_20,
                                                                     Avg_leg_darts_JdS_total, Avg_leg_darts_JdS_20,
                                                                     Avg_leg_darts_JdS_21,
                                                                     Avg_leg_darts_DvD_total, Avg_leg_darts_DvD_20,
                                                                     Avg_leg_darts_DvD_21,
                                                                     Avg_leg_darts_NA_total, Avg_leg_darts_NA_18,
                                                                     Avg_leg_darts_NA_19],
                            "Average score remaining of opponent when player finished leg":
                                [Rem_avg_score_DVdB_total, Rem_avg_score_DVdB_19, Rem_avg_score_DVdB_20,
                                 Rem_avg_score_DVdB_21,
                                 Rem_avg_score_AL_total, Rem_avg_score_AL_19, Rem_avg_score_AL_20,
                                 Rem_avg_score_JdS_total, Rem_avg_score_JdS_20, Rem_avg_score_JdS_21,
                                 Rem_avg_score_DvD_total, Rem_avg_score_DvD_20, Rem_avg_score_DvD_21,
                                 Rem_avg_score_NA_total, Rem_avg_score_NA_18, Rem_avg_score_NA_19]
                            })

# TRANSFORM dataframe to an excel file with the following command:

# information.to_excel(my_directory + '/created_sub_tables/constructed_information_table.xlsx')

# ================================================================================================================
# datatable.info()
# datatable.describe()

