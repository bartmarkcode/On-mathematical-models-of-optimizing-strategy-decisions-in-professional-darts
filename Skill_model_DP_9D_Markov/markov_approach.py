import evaluate_policy_and_aming_grid as epag
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
import pandas as pd
import maximize_expected_score as mes
from itertools import permutations

directory_of_action_set = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/Programms for MA/result_grid'
my_directory ='C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/created_sub_tables'
# =====================================================================================================================
allowed_fields = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30,
                  32, 33, 34, 36, 38, 39, 40, 42, 45, 48, 50, 51, 54, 57, 60]
allowed_doubles = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 50]
bogey = [159, 162, 163, 165, 166, 168, 169]
# =====================================================================================================================
# output all possible checkouts (of length <= 3) that can finish a total score of 'S':
# if ways == True: return the checkouts
# if ways == False: return the set of all possible first strategies h of the checkouts
def poss_turn_strategy(S, ways=False):
    checkout_2 = [i for i in permutations(allowed_fields * 2, 2) if sum(i) == S]
    checkout_3 = [j for j in permutations(allowed_fields * 3, 3) if sum(j) == S]
    checkouts = []
    H = []
    if S in allowed_doubles and S not in H:
        H.append(S)
        checkouts.append(S)
    # print(checkout_3)
    for c in checkout_2:
        # print(c,c[0]+c[1])
        if c[1] in allowed_doubles and c not in checkouts:
            checkouts.append(c)
    for cc in checkout_3:
        if cc[2] in allowed_doubles and cc not in checkouts:
            checkouts.append(cc)
    if ways == True:
        return checkouts

    for ccc in checkouts:
        if not isinstance(ccc, int) and ccc[0] not in H:
            H.append(ccc[0])
    return H

# print(poss_turn_strategy(5,ways=True))
# ====================================================================================================================
# CREATE the TRANSITION MATRIX for a given information item:
def transition_matrix(item):
    file_it = directory_of_action_set + '/v2_item{}'.format(item) + "_gaussin_prob_grid.pkl"
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
     prob_grid_bullscore] = epag.load_aiming_grid(file_it)
    # prob_grid_normalscore: 984x61
    # prob_grid_double_score: 984x 20
    # prob_grid_bullscore: 984x2
    time1 = time.time()
    print("go:", time1)

    abs_dict = {0: 0, 1: 0}
    M = np.zeros((171, 171))

    for u in range(len(M)):
        if u not in bogey and u != 0 and u != 1:
            H = sorted(poss_turn_strategy(u))
            abs_list = {}
            p_dict = {}
            for h in H:
                mu_j = np.empty(0)
                p_ij = np.empty(0)
                for l in range(len(H)):
                    mu_j = np.append(mu_j, abs_dict[u - H[l]])
                    max_val = 0
                    if h == u and h in allowed_doubles and h != 50:
                        for k in range(len(prob_grid_doublescore)):
                            if prob_grid_doublescore[k][int(h / 2 - 1)] > max_val:
                                tempmax = k
                                max_val = prob_grid_doublescore[k][int(h / 2 - 1)]
                    else:
                        for k in range(len(prob_grid_normalscore)):
                            if prob_grid_normalscore[k][h] > max_val:
                                tempmax = k
                                max_val = prob_grid_normalscore[k][h]
                    p_ij = np.append(p_ij, prob_grid_normalscore[tempmax][H[l]])
                p_dict[h] = p_ij
                abs_list[h] = 1 + np.sum(mu_j * p_ij)
            abs_dict[u] = min(abs_list.values())
            arg_min_p = [k for k, g in abs_list.items() if g == abs_dict[u]]
            p_for_T = list(p_dict[arg_min_p[0]])
            q_for_T = 1 - sum(p_for_T)

            for v in range(len(M)):
                if v < u and v != 1 and u - v in H:
                    M[u, v] = p_for_T.pop()
                elif u == v:
                    M[u, v] = q_for_T

    time2 = time.time()
    print("computation time:", time2 - time1)
    return M

# print(transition_matrix(1))
# ===================================================================================================================
# BUILD the transition maps out of the transition matrix of an information item:
def transition_map(item):
    M = transition_matrix(item)
    sns.heatmap(M, annot=False, cmap="PuRd", linewidths=0.2, linecolor='black',
                cbar_kws={'label': 'Transition Probability', "location": "left"}).xaxis.tick_top()
    plt.show()

# transition_map(7)
# ==================================================================================================================
# CHOOSE the optimal checkout given total score 'S' and player-specific transition matrix 'M':
def markov_strategy(S, M, max_exp_score):
    C = poss_turn_strategy(S, ways=True)
    recall_S = S
    if S == 0 or S == 1:
        return None
    elif S > 170 or S in bogey:
        # max_exp score
        return [max_exp_score]
    else:
        opt_checkout = None
        opt_checkout_prob = 0
        for c in C:
            prob_c = 1
            if isinstance(c, int):
                prob_c *= M[S, S - c]
            else:
                for cc in c:
                    prob_c *= M[S, S-cc]
                    S = S-cc
            if opt_checkout is None or prob_c > opt_checkout_prob:
                if isinstance(c, int):
                    opt_checkout = [c]
                else:
                    opt_checkout = list(c)
                opt_checkout_prob = prob_c
            S = recall_S
        return opt_checkout

# print(markov_strategy(60, transition_matrix(1),mes.max_exp_score(1)[2]))
# ===============================================================================================================
# For a given information item, compute the optimal checkouts for S=2,...,501 and store them in a dictionary:
def markov_result(item):
    M = transition_matrix(item)
    max_exp_score = mes.max_exp_score(item)[2]
    dict = {}
    for s in range(2,502):
        val = markov_strategy(s, M, max_exp_score)
        dict[s] = val
    return dict

# print(markov_result(1))
# ------------------------------------------------------------------------------------------------------------------
# Store the Markov strategies in an excel file:
def create_checkouts(item):
    # for i in range(len(item_list)):
    checkout = markov_result(item)
    strat_list = []
    for suc in checkout.keys():
        strat_list.append(checkout[suc][0])
        if suc in bogey or suc > 170:
            eps = suc - sum(checkout[suc])
            while eps != 0:
                checkout[suc] = checkout[suc] + checkout[eps]
                eps = suc - sum(checkout[suc])

    output = pd.DataFrame({"Score": list(checkout.keys()),
                           "Track of numerical scores": list(checkout.values()),
                           "Optimal numerical value s*": strat_list})
    return output

def write_markov():
    DVdB19_Markov = create_checkouts(1)
    DVdB20_Markov = create_checkouts(2)
    DVdB21_Markov = create_checkouts(3)
    AL19_Markov = create_checkouts(4)
    AL20_Markov = create_checkouts(5)
    JdS20_Markov = create_checkouts(6)
    JdS21_Markov = create_checkouts(7)
    DvD20_Markov = create_checkouts(8)
    DvD21_Markov = create_checkouts(9)
    NA18_Markov = create_checkouts(10)
    NA19_Markov = create_checkouts(11)

    with pd.ExcelWriter(my_directory + '/Markov_recommendations.xlsx') as writer:
        DVdB19_Markov.to_excel(writer, sheet_name="DVdB19")
        DVdB20_Markov.to_excel(writer, sheet_name="DVdB20")
        DVdB21_Markov.to_excel(writer, sheet_name="DVdB21")
        AL19_Markov.to_excel(writer, sheet_name="AL19")
        AL20_Markov.to_excel(writer, sheet_name="AL20")
        JdS20_Markov.to_excel(writer, sheet_name="JdS20")
        JdS21_Markov.to_excel(writer, sheet_name="JdS21")
        DvD20_Markov.to_excel(writer, sheet_name="DvD20")
        DvD21_Markov.to_excel(writer, sheet_name="DvD21")
        NA18_Markov.to_excel(writer, sheet_name="NA18")
        NA19_Markov.to_excel(writer, sheet_name="NA19")
# write_markov()


