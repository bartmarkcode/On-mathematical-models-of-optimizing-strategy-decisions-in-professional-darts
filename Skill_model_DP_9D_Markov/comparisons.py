import os
import pandas as pd
import matplotlib.pyplot as plt
import nine_d as nd

my_directory = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/created_sub_tables'
os.chdir(my_directory)
# ==================================================================================================================
# READ IN all recommendations of all created methods and create corresponding variables:
rec_9D = '/9D_recommendations.xlsx'
item1_9D = pd.read_excel(rec_9D, sheet_name="DVdB19")
item2_9D = pd.read_excel(rec_9D, sheet_name="DVdB20")
item3_9D = pd.read_excel(rec_9D, sheet_name="DVdB21")
item4_9D = pd.read_excel(rec_9D, sheet_name="AL19")
item5_9D = pd.read_excel(rec_9D, sheet_name="AL20")
item6_9D = pd.read_excel(rec_9D, sheet_name="JdS20")
item7_9D = pd.read_excel(rec_9D, sheet_name="JdS21")
item8_9D = pd.read_excel(rec_9D, sheet_name="DvD20")
item9_9D = pd.read_excel(rec_9D, sheet_name="DvD21")
item10_9D = pd.read_excel(rec_9D, sheet_name="NA18")
item11_9D = pd.read_excel(rec_9D, sheet_name="NA19")

rec_DP_turn = '/DP_turn_feature_recommendations.xlsx'
item1_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="DVdB19")
item2_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="DVdB20")
item3_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="DVdB21")
item4_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="AL19")
item5_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="AL20")
item6_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="JdS20")
item7_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="JdS21")
item8_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="DvD20")
item9_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="DvD21")
item10_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="NA18")
item11_DP_turn = pd.read_excel(rec_DP_turn, sheet_name="NA19")

rec_DP_noturn = '/DP_noturn_recommendations.xlsx'
item1_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="DVdB19")
item2_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="DVdB20")
item3_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="DVdB21")
item4_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="AL19")
item5_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="AL20")
item6_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="JdS20")
item7_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="JdS21")
item8_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="DvD20")
item9_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="DvD21")
item10_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="NA18")
item11_DP_noturn = pd.read_excel(rec_DP_noturn, sheet_name="NA19")

rec_Markov = '/Markov_recommendations.xlsx'
item1_Markov = pd.read_excel(rec_Markov, sheet_name="DVdB19")
item2_Markov = pd.read_excel(rec_Markov, sheet_name="DVdB20")
item3_Markov = pd.read_excel(rec_Markov, sheet_name="DVdB21")
item4_Markov = pd.read_excel(rec_Markov, sheet_name="AL19")
item5_Markov = pd.read_excel(rec_Markov, sheet_name="AL20")
item6_Markov = pd.read_excel(rec_Markov, sheet_name="JdS20")
item7_Markov = pd.read_excel(rec_Markov, sheet_name="JdS21")
item8_Markov = pd.read_excel(rec_Markov, sheet_name="DvD20")
item9_Markov = pd.read_excel(rec_Markov, sheet_name="DvD21")
item10_Markov = pd.read_excel(rec_Markov, sheet_name="NA18")
item11_Markov = pd.read_excel(rec_Markov, sheet_name="NA19")
# ====================================================================================================================
# ANALYZE dependency of a DP recommendation (with turn feature (S,i)) to the number of aivailable darts 'i':
def analyze_DP_turn_feature():
    s = 0
    coinc = 0
    coinc_f = 0
    different = [i for i in range(2,502)]
    while s!= 1500:
        if item11_DP_turn.loc[s,"Optimal target location"] == item11_DP_turn.loc[s+1,"Optimal target location"] and item11_DP_turn.loc[s+2,"Optimal target location"] == item11_DP_turn.loc[s + 1, "Optimal target location"]:
            coinc += 1
            exclude = s // 3 + 2
            different.remove(exclude)
            if exclude < 171:
                coinc_f +=1
        s += 3

    return coinc, coinc_f, different

# print(analyze_DP_turn_feature())
# --------------------------------------------------------------------------------------------------------------------
# ACCORDANCE of 9D stratgey/track of the same player in different years:
def compare_9D(column):
    notc_list = []
    c_val = 0
    for score in range(500):
        if item10_9D.loc[score,column] == item11_9D.loc[score,column]:
            c_val += 1
        else:
            notc_list.append(score)
    print("percentage:", c_val/500,"len_differences:",len(notc_list))

# compare_9D("Track")
# compare_9D("Optimal strategy z*")
# ------------------------------------------------------------------------------------------------------------------
# ACCORDANCE of DP strategy/track/target location of the same player in different years:
def compare_DP(column):
    notc_list = []
    c_val = 0
    for score in range(500):
        if item6_DP_noturn.loc[score,column] == item7_DP_noturn.loc[score,column]:
            c_val += 1
        else:
            notc_list.append(score)
    print("percentage:", c_val/500, "len_differences:", len(notc_list))

# print(compare_DP("Optimal strategy z*"))
# print(compare_DP("Track"))
# print(compare_DP("Optimal target location"))
# --------------------------------------------------------------------------------------------------------------------
# ACCORDANCE of Markov strategy/track of the same player in different years:
def compare_Markov(column):
    notc_list = []
    c_val = 0
    for score in range(500):
        if item10_Markov.loc[score,column] == item11_Markov.loc[score,column]:
            c_val += 1
        else:
            notc_list.append(score)
    print("percentage:", c_val / 500, "len_differences:", len(notc_list))

# compare_Markov("Optimal numerical value s*")
# compare_Markov("Track of numerical scores")
# --------------------------------------------------------------------------------------------------------------------
# COMPARE strategies/tracks/length of tracks of all three approaches and analyze the accordance:
def compare_all(column):
    c_DP_9D = 0
    c_DP_Markov = 0
    c_9D_Markov = 0
    c_total = 0
    Markov_min_len = 0
    DP_min_len = 0
    for score in range(500):
        if item11_DP_noturn.loc[score, column] == item11_9D.loc[score, column]:
            c_DP_9D += 1
        if item11_DP_noturn.loc[score,column] == item11_Markov.loc[score,column]:
            c_DP_Markov += 1
        if item11_9D.loc[score, column] == item11_Markov.loc[score,column]:
            c_9D_Markov += 1
        if item11_DP_noturn.loc[score, column] == item11_9D.loc[score, column] and item11_DP_noturn.loc[score,column] == item11_Markov.loc[score,column] and item11_9D.loc[score, column] == item11_Markov.loc[score,column]:
            c_total += 1
        # Do the DP- and Markov Tracks also have minimum length:
        if len(item11_DP_noturn.loc[score,column]) == len(item11_9D.loc[score,column]):
            Markov_min_len += 1
        if len(item11_Markov.loc[score,column]) == len(item11_9D.loc[score,column]):
            DP_min_len += 1
    print("DP-9D:",c_DP_9D/500, "DP-Markov:", c_DP_Markov/500, "9D-Markov:", c_9D_Markov/500, "TOTAL:", "Min_DP:", DP_min_len/500, c_total/500, "Min_Markov:", Markov_min_len/500)

# compare_all("Optimal numerical value s*")
# compare_all("Track of numerical scores")
# ===================================================================================================================
# helper function for plot_track():
def aggregate_track(T):
    T2 = [0]
    for t in range(len(T)):
        T2.append(T2[t]+T[t])
    return T2

# print(aggregate_track([1,4,60,7,10]))
# -------------------------------------------------------------------------------------------------------------------
# Possible visualization of optimal DP/Markov/9D - tracks for a given information item and current total score:
def plot_track(T1,T2,T3):
    T1 = T1.strip('][').split(', ')
    for t in range(len(T1)):
        T1[t] = int(T1[t])
    T2 = T2.strip('][').split(', ')
    for t in range(len(T2)):
        T2[t] = int(T2[t])
    T3 = T3.strip('][').split(', ')
    for t in range(len(T3)):
        T3[t] = int(T3[t])
    T1 = aggregate_track(T1)
    T2 = aggregate_track(T2)
    T3 = aggregate_track(T3)
    nr_x = [i for i in range(len(max(T1,T2,T3)))]
    plt.plot(nr_x,T1, "o-", label="DP",)
    plt.plot(nr_x,T2,"o-", label="Markov")
    plt.plot(nr_x,T3,"o-", label="9D")
    for k, label in enumerate(T1):
        plt.annotate(223-label, (nr_x[k], T1[k]+2),size=14)
    for k, label in enumerate(T2):
        plt.annotate(223-label, (nr_x[k], T2[k]+2),size=14)
    for k, label in enumerate(T3):
        plt.annotate(223-label, (nr_x[k], T3[k]+2),size=14)
    plt.xlabel("Throw", size=20)
    plt.ylabel("Score achieved so far", size =20)
    plt.xticks(nr_x)
    plt.legend(fontsize=20)
    plt.title("Van den Bergh 2020; S=223", size=25)
    plt.margins(0.2)
    plt.show()

# plot_track(item2_DP_noturn.loc[221, "Track of numerical scores"],item2_Markov.loc[221,"Track of numerical scores"],item2_9D.loc[221,"Track of numerical scores"])
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================
# =====================================================================================================================

#              FINALLY; ANALYZE which method coincides the most with the real game behavior of the players:

# For that: read-in again the tables for each information item:

DVdB_table19 = pd.read_excel(my_directory + '/Player_specific_tables/vandenBergh19.xlsx')
DVdB_table20 = pd.read_excel(my_directory + '/Player_specific_tables/vandenBergh20.xlsx')
DVdB_table21 = pd.read_excel(my_directory + '/Player_specific_tables/vandenBergh21.xlsx')
AL_table19 = pd.read_excel(my_directory + '/Player_specific_tables/Lewis19.xlsx')
AL_table20 = pd.read_excel(my_directory + '/Player_specific_tables/Lewis20.xlsx')
JdS_table20 = pd.read_excel(my_directory + '/Player_specific_tables/deSousa20.xlsx')
JdS_table21 = pd.read_excel(my_directory + '/Player_specific_tables/deSousa21.xlsx')
DvD_table20 = pd.read_excel(my_directory + '/Player_specific_tables/vanDuijvenbode20.xlsx')
DvD_table21 = pd.read_excel(my_directory + '/Player_specific_tables/vanDuijvenbode21.xlsx')
NA_table18 = pd.read_excel(my_directory + '/Player_specific_tables/Aspinall19.xlsx')
NA_table19 = pd.read_excel(my_directory + '/Player_specific_tables/Aspinall18.xlsx')

num_z = nd.num_z

attempt_dict = {"one": ["1st dart attempt (T/S/D/DD/Bull)",
                        "1st dart attempt (segment 1-20)",
                        "sportradar_first"],
                "two": ["2nd dart attempt (T/S/D/DD/Bull)", "2nd dart attempt (segment 1-20)", "sportradar_second"],
                "three": ["3rd dart attempt (T/S/D/DD/Bull)", "3rd dart attempt (segment 1-20)", "sportradar_third"]}

def analyze_real_game(table,strategy):
    total_throws = 0
    accordance = 0
    for u in attempt_dict:
        for cell in range(len(table.index.tolist())):
            if pd.isna(table[attempt_dict[u][1]].values[cell]):
                continue
            elif pd.isna(table[attempt_dict[u][2]].values[cell]):
                continue
            else:
                if u == "one":
                    start_score = int(table["Player starting score"].values[cell])
                if u == "two":
                    start_score = int(table["Points after 1st dart thrown"].values[cell])
                if u == "three":
                    start_score = int(table["Points after 2nd dart thrown"].values[cell])
                TR = str(table[attempt_dict[u][0]].values[cell]) + str(int(table[attempt_dict[u][1]].values[cell]))
                if TR == "D25":
                    TR = "DB25"
                if TR not in num_z.keys():
                    continue
                points = num_z[TR]
                if strategy.loc[start_score - 2,"Optimal numerical value s*"] == points:
                    accordance += 1
                total_throws += 1
    return accordance, total_throws, accordance/total_throws


print("------------DVdB---------")
print(analyze_real_game(DVdB_table19,item1_DP_noturn))
print(analyze_real_game(DVdB_table19,item1_Markov))
print(analyze_real_game(DVdB_table19,item1_9D))
print("-------------------------")
print(analyze_real_game(DVdB_table20,item2_DP_noturn))
print(analyze_real_game(DVdB_table20,item2_Markov))
print(analyze_real_game(DVdB_table20,item2_9D))
print("------------------------------")
print(analyze_real_game(DVdB_table21,item3_DP_noturn))
print(analyze_real_game(DVdB_table21,item3_Markov))
print(analyze_real_game(DVdB_table21,item3_9D))
print("------------AL-----------------")
print(analyze_real_game(AL_table19,item4_DP_noturn))
print(analyze_real_game(AL_table19,item4_Markov))
print(analyze_real_game(AL_table19,item4_9D))
print("------------------------------")
print(analyze_real_game(AL_table20,item5_DP_noturn))
print(analyze_real_game(AL_table20,item5_Markov))
print(analyze_real_game(AL_table20,item5_9D))
print("---------------JdS--------------")
print(analyze_real_game(JdS_table20,item6_DP_noturn))
print(analyze_real_game(JdS_table20,item6_Markov))
print(analyze_real_game(JdS_table20,item6_9D))
print("-----------------------------")
print(analyze_real_game(JdS_table21,item7_DP_noturn))
print(analyze_real_game(JdS_table21,item7_Markov))
print(analyze_real_game(JdS_table21,item7_9D))
print("--------------DvD---------------")
print(analyze_real_game(DvD_table20,item8_DP_noturn))
print(analyze_real_game(DvD_table20,item8_Markov))
print(analyze_real_game(DvD_table20,item8_9D))
print("-----------------------------")
print(analyze_real_game(DvD_table21,item9_DP_noturn))
print(analyze_real_game(DvD_table21,item9_Markov))
print(analyze_real_game(DvD_table21,item9_9D))
print("------------NA----------------")
print(analyze_real_game(NA_table18,item10_DP_noturn))
print(analyze_real_game(NA_table18,item10_Markov))
print(analyze_real_game(NA_table18,item10_9D))
print("-----------------------------")
print(analyze_real_game(NA_table19,item11_DP_noturn))
print(analyze_real_game(NA_table19,item11_Markov))
print(analyze_real_game(NA_table19,item11_9D))