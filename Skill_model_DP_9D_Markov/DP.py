import os
import time
import pandas as pd
import numpy as np

import function_tools as ft
import dartboard as onb
import evaluate_policy_and_aming_grid as epag

np.set_printoptions(precision=4)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=300)

import torch

torch.set_printoptions(precision=4)
torch.set_printoptions(linewidth=300)
torch.set_printoptions(threshold=300)

directory_of_action_set = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/Programms for MA/result_grid'
my_directory = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/created_sub_tables'
# ====================================================================================================================
# single player game without the turn feature
def solve_dp_noturn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore=None, prob_grid_bullscore=None,
                    prob_grid_doublescore_dic=None):
    """
    Solve the single player game without the turn feature. Find the optimal policy to minimize the expected number of
    throws for reaching zero score.
    Args:
        the action set and the hitting probability associated with the skill model

    Returns:
        optimal_value[score_state]: the expected number of throws for reaching zero from score_state=2,...,501.
        optimal_action_index[score_state]: the index of the aiming location used for score_state=2,...,501.
    """

    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore_1tosmax_dic = {}
    prob_normalscore_1tosmaxsum_dic = {}
    for score_max in range(0, 61):
        score_max_plus1 = score_max + 1
        prob_normalscore_1tosmax_dic[score_max] = np.array(prob_grid_normalscore[:, 1:score_max_plus1])
        prob_normalscore_1tosmaxsum_dic[score_max] = prob_normalscore_1tosmax_dic[score_max].sum(axis=1)
    if prob_grid_doublescore_dic is None:
        prob_doublescore_dic = {}
        for doublescore_index in range(20):
            doublescore = 2 * (doublescore_index + 1)
            prob_doublescore_dic[doublescore] = np.array(prob_grid_doublescore[:, doublescore_index])
    else:
        prob_doublescore_dic = prob_grid_doublescore_dic
    prob_DB = np.array(prob_grid_bullscore[:, 1])

    # possible state: s = 0,1(not possible),2,...,501
    optimal_value = np.zeros(502)
    # optimal_value[1] = np.nan
    optimal_action_index = np.zeros(502, np.int32)
    optimal_action_index[0] = -1
    optimal_action_index[1] = -1

    for score_state in range(2, 502):
        # use matrix operation to search all aiming locations

        # transit to less score state
        # s1 = min(score_state-2, 60)
        # p[z=1]*v[score_state-1] + p[z=2]*v[score_state-2] + ... + p[z=s1]*v[score_state-s1]
        score_max = min(score_state - 2, 60)
        score_max_plus1 = score_max + 1
        # transit to next state
        num_tothrow = 1.0 + prob_normalscore_1tosmax_dic[score_max].dot(
            optimal_value[score_state - 1:score_state - score_max - 1:-1])
        # probability of transition to state other than s itself
        prob_otherstate = prob_normalscore_1tosmaxsum_dic[score_max]

        # transit to the end of game
        if score_state == onb.score_DB:  # hit double bull
            prob_otherstate += prob_DB
        elif (score_state <= 40 and score_state % 2 == 0):  # hit double
            prob_otherstate += prob_doublescore_dic[score_state]
        else:  # game does not end
            pass

        # expected number of throw for all aiming locations
        prob_otherstate = np.maximum(prob_otherstate, 0)
        num_tothrow = num_tothrow / prob_otherstate

        # searching
        optimal_value[score_state] = num_tothrow.min()
        optimal_action_index[score_state] = num_tothrow.argmin()

    return [optimal_value, optimal_action_index]

# ================================================================================================================
# Solve the DP method for an information item and store the gathered results in an excel file:
def solve_noturn(item):
    file_it = directory_of_action_set + '/v2_item{}'.format(item) + "_gaussin_prob_grid.pkl"
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
     prob_grid_bullscore] = epag.load_aiming_grid(file_it)
    result_dic = solve_dp_noturn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    scores = [s for s in range(2, 502)]
    opt_strat = []
    opt_strat_value = []
    exp_list = []
    target_location_list = []
    track_dict_numerical = {}
    track_dict = {}
    for j in scores:
        track_dict_numerical[j] = []
        track_dict[j] = []
        a = result_dic[0][j]
        exp_list.append(a)
        b = result_dic[1][j]
        xj, yj = aiming_grid[b][0] - 170, aiming_grid[b][1] - 170
        target_location_list.append((xj,yj))
        if yj == 99:
            yj = 100
        strat = onb.get_score_and_multiplier(xj, yj)
        opt_strat_value.append(strat[0])
        if strat[0] == 25:
            opt_strat.append("SB25")
        elif strat[0] == 50:
            opt_strat.append("DB25")
        else:
            if strat[1] == 1:
                opt_strat.append("S" + str(int(strat[0])))
            elif strat[1] == 2:
                opt_strat.append("D" + str(int(strat[0] / 2)))
            elif strat[1] == 3:
                opt_strat.append(("T" + str(int(strat[0] / 3))))

    for d in scores:
        eps = d
        while eps != 0:
            track_dict[d].append(opt_strat[eps-2])
            track_dict_numerical[d].append(opt_strat_value[eps-2])
            eps -= opt_strat_value[eps-2]

    output = pd.DataFrame({"Score": scores,
                           "Optimal target location": target_location_list,
                           "Track": list(track_dict.values()),
                          "Optimal strategy z*": opt_strat,
                           "Optimal numerical value s*": opt_strat_value,
                           "Track of numerical scores": list(track_dict_numerical.values()),
                           "Expected number of throws": exp_list})

    return output

# print(solve_noturn(1))
# --------------------------------------------------------------------------------------------------------------
# solve DP for each information item and store the results afterwards
def write_noturn():
    DVdB19_DP = solve_noturn(1)
    DVdB20_DP = solve_noturn(2)
    DVdB21_DP = solve_noturn(3)
    AL19_DP = solve_noturn(4)
    AL20_DP = solve_noturn(5)
    JdS20_DP = solve_noturn(6)
    JdS21_DP = solve_noturn(7)
    DvD20_DP = solve_noturn(8)
    DvD21_DP = solve_noturn(9)
    NA18_DP = solve_noturn(10)
    NA19_DP = solve_noturn(11)

    with pd.ExcelWriter(my_directory + '/DP_noturn_recommendations.xlsx') as writer:
        DVdB19_DP.to_excel(writer, sheet_name="DVdB19")
        DVdB20_DP.to_excel(writer, sheet_name="DVdB20")
        DVdB21_DP.to_excel(writer, sheet_name="DVdB21")
        AL19_DP.to_excel(writer, sheet_name="AL19")
        AL20_DP.to_excel(writer, sheet_name="AL20")
        JdS20_DP.to_excel(writer, sheet_name="JdS20")
        JdS21_DP.to_excel(writer, sheet_name="JdS21")
        DvD20_DP.to_excel(writer, sheet_name="DvD20")
        DvD21_DP.to_excel(writer, sheet_name="DvD21")
        NA18_DP.to_excel(writer, sheet_name="NA18")
        NA19_DP.to_excel(writer, sheet_name="NA19")

# write_noturn()
# ==================================================================================================================
# single player game with the turn feature:
def solve_dp_turn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore):
    """
        Solve the single player game with the turn feature.
        Args:
            the action set and the hitting probability associated with the skill model

        Returns:
            optimal values and the corresponding aiming locations for each state (s,i,u)
        """

    # aiming_grid
    num_aiming_location = aiming_grid.shape[0]
    prob_normalscore = prob_grid_normalscore
    prob_doublescore_dic = {}
    for doublescore_index in range(20):
        doublescore = 2 * (doublescore_index + 1)
        prob_doublescore_dic[doublescore] = np.array(prob_grid_doublescore[:, doublescore_index])
    prob_DB = np.array(prob_grid_bullscore[:, 1])

    # the probability of not bust for each action given score_max=i (score_remain=i+2)
    prob_bust_dic = {}
    prob_notbust_dic = {}
    for score_max in range(60):
        # transit to next throw or turn
        prob_notbust = prob_grid_normalscore[:, 0:score_max + 1].sum(axis=1)
        # transit to the end of game
        score_remain = score_max + 2
        if score_remain == onb.score_DB:
            prob_notbust += prob_DB
        elif score_remain <= 40 and score_remain % 2 == 0:
            prob_notbust += prob_doublescore_dic[score_remain]
        ##
        prob_notbust = np.minimum(np.maximum(prob_notbust, 0), 1)
        prob_notbust_dic[score_max] = prob_notbust
        prob_bust_dic[score_max] = 1 - prob_notbust_dic[score_max]

    prob_normalscore_tensor = torch.from_numpy(prob_normalscore)

    iteration_round_limit = 20
    iteration_relerror_limit = 10 ** -9

    #### state space example of (SB=25 DB=50) ####
    ## rt: the number of remaining throws in a turn
    ## state_infeasible_rt2 = [23, 29, 31, 35, 37, 41, 43, 44, 46, 47, 49, 52, 53, 55, 56, 58, 59]
    ## state_infeasible_rt1 = [103, 106, 109, 112, 113, 115, 116, 118, 119]

    optimal_value_rt3 = np.zeros(502)  # vector: optimal value for the beginning state of each turn (rt=3)
    optimal_value_dic = {}  # first key: score=0,2,...,501, second key: remaining throws=3,2,1
    optimal_action_index_dic = {}
    num_iteration_record = np.zeros(502, dtype=np.int32)

    state_len_vector = np.zeros(4, dtype=np.int32)
    state_value = [
        None]  # optimal value (expected # of turns to finish the game) for each state in the current playing turn
    state_action = [None]  # aiming locations for each state in the current playing turn
    action_diff = [None]
    value_relerror = np.zeros(4)
    for rt in [1, 2, 3]:
        # for rt=3: possible score_gained = 0
        # for rt=2: possible score_gained = 0,1,...,60
        # for rt=1: possible score_gained = 0,1,...,120
        this_throw_state_len = onb.maxhitscore * (3 - rt) + 1
        state_value.append(np.ones(this_throw_state_len) * onb.largenumber)
        state_action.append(np.ones(this_throw_state_len, np.int32) * onb.infeasible_marker)
        action_diff.append(np.ones(this_throw_state_len))
    state_value_update = ft.copy_numberarray_container(state_value)
    state_action_update = ft.copy_numberarray_container(state_action)

    # use no_turn policy as the initial policy
    [noturn_optimal_value, noturn_optimal_action_index] = solve_dp_noturn(aiming_grid, prob_grid_normalscore,
                                                                          prob_grid_doublescore,
                                                                          prob_grid_bullscore)

    t1 = time.time()
    for score_state in range(2, 502):
        # print('#### solve_dp_turn score_state={} ####'.format(score_state))

        # initialization
        for rt in [1, 2, 3]:
            # for rt=3: score_gained = 0
            # for rt=2: score_gained = 0,1,...,min(s-2,60)
            # for rt=1: score_gained = 0,1,...,min(s-2,120)
            this_throw_state_len = min(score_state - 2, onb.maxhitscore * (3 - rt)) + 1
            state_len_vector[rt] = this_throw_state_len

            # initialize the starting policy:
            # use no_turn action in (s, i, u=0)
            # use turn action (s-1, i, u-1) in (s, i, u!=0) if (s-1, i, u-1) is feasible state
            state_action[rt][0] = noturn_optimal_action_index[score_state]
            for score_gained in range(1, this_throw_state_len):
                if onb.state_feasible_array[rt, score_gained]:  # if True
                    if onb.state_feasible_array[rt, score_gained - 1]:
                        state_action[rt][score_gained] = optimal_action_index_dic[score_state - 1][rt][
                            score_gained - 1]
                    else:
                        state_action[rt][score_gained] = noturn_optimal_action_index[score_state - score_gained]
                else:
                    state_action[rt][score_gained] = onb.infeasible_marker

        # policy iteration
        for round_index in range(iteration_round_limit):

            # policy evaluation
            rt = 3
            score_gained = 0
            score_max_turn = min(score_state - 2, 3 * onb.maxhitscore)
            prob_turn_transit = epag.solve_turn_transit_probability_fast(score_state, state_action,
                                                                         prob_grid_normalscore,
                                                                         prob_grid_doublescore, prob_grid_bullscore,
                                                                         prob_bust_dic)
            prob_turn_zeroscore = prob_turn_transit['bust'] + prob_turn_transit['score'][0]
            new_value_rt3 = (1 + np.dot(prob_turn_transit['score'][1:],
                                        optimal_value_rt3[score_state - 1:score_state - score_max_turn - 1:-1])) / (
                                    1 - prob_turn_zeroscore)
            state_value_update[rt][score_gained] = new_value_rt3
            optimal_value_rt3[score_state] = new_value_rt3
            # print('evaluate rt3 value= {}'.format(new_value_rt3)

            # policy improvement
            for rt in [1, 2, 3]:
                this_throw_state_len = state_len_vector[rt]

                # state which can not bust.  score_state-score_gained>=62
                state_notbust_len = max(min(score_state - 61, this_throw_state_len), 0)
                if state_notbust_len > 0:
                    if rt == 1 and round_index == 0:
                        # combine all non-bust states together
                        state_notbust_update_index = state_notbust_len
                        next_state_value_array = np.zeros((61, state_notbust_len))
                        for score_gained in range(state_notbust_len):
                            # skip infeasible state
                            if not onb.state_feasible_array[rt, score_gained]:
                                continue
                            score_remain = score_state - score_gained
                            score_max = 60  # always 60 here
                            score_max_plus1 = score_max + 1
                            next_state_value_array[:, score_gained] = optimal_value_rt3[
                                                                      score_remain:score_remain - score_max_plus1:-1]
                    elif rt == 2 and (round_index == 0 or score_state < 182):
                        # combine all non-bust states together
                        state_notbust_update_index = state_notbust_len
                        next_state_value_array = np.zeros((61, state_notbust_len))
                        for score_gained in range(state_notbust_len):
                            # skip infeasible state
                            if not onb.state_feasible_array[rt, score_gained]:
                                continue
                            score_remain = score_state - score_gained
                            score_max = 60  # always 60 here
                            score_max_plus1 = score_max + 1
                            next_state_value_array[:, score_gained] = state_value_update[rt - 1][
                                                                      score_gained:score_gained + score_max_plus1]
                    else:  # (rt==1 and round_index>0) or (rt==2 and round_index>0 and score_state>=182) or (rt==3)
                        # only update state of score_gained = 0
                        state_notbust_update_index = 1
                        next_state_value_array = np.zeros(61)
                        score_gained = 0
                        score_remain = score_state - score_gained
                        score_max = 60  # always 60 here
                        score_max_plus1 = score_max + 1
                        # make a copy
                        if (rt > 1):
                            next_state_value_array[:] = state_value_update[rt - 1][
                                                        score_gained:score_gained + score_max_plus1]
                        # transit to next turn when rt=1
                        else:
                            next_state_value_array[:] = optimal_value_rt3[
                                                        score_remain:score_remain - score_max_plus1:-1]

                    # matrix product to compute all together
                    next_state_value_tensor = torch.from_numpy(next_state_value_array)
                    # transit to next throw in the same turn when rt=3,2
                    if rt > 1:
                        num_turns_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)
                    # transit to next turn when rt=1
                    else:
                        num_turns_tensor = 1 + prob_normalscore_tensor.matmul(next_state_value_tensor)

                    # searching
                    temp1 = num_turns_tensor.min(axis=0)
                    state_action_update[rt][0:state_notbust_update_index] = temp1.indices.numpy()
                    state_value_update[rt][0:state_notbust_update_index] = temp1.values.numpy()

                    # state which possibly bust.  score_state-score_gained<62
                if state_notbust_len < this_throw_state_len:
                    # combine all bust states together
                    state_bust_len = this_throw_state_len - state_notbust_len
                    next_state_value_array = np.zeros((61, state_bust_len))
                    for score_gained in range(state_notbust_len, this_throw_state_len):
                        # skip infeasible state
                        if not onb.state_feasible_array[rt, score_gained]:
                            continue
                        score_remain = score_state - score_gained
                        # score_max = min(score_remain-2, 60)
                        score_max = score_remain - 2  # less than 60 here
                        score_max_plus1 = score_max + 1
                        score_gained_index = score_gained - state_notbust_len  # index off set
                        if rt > 1:
                            next_state_value_array[0:score_max_plus1, score_gained_index] = state_value_update[rt - 1][
                                                                                            score_gained:score_gained + score_max_plus1]
                        # transit to next turn when rt=1
                        else:
                            next_state_value_array[0:score_max_plus1, score_gained_index] = optimal_value_rt3[
                                                                                            score_remain:score_remain - score_max_plus1:-1]

                    next_state_value_tensor = torch.from_numpy(next_state_value_array)
                    # transit to next throw in the same turn when rt=3,2
                    if rt > 1:
                        num_turns_tensor = prob_normalscore_tensor.matmul(next_state_value_tensor)
                    # transit to next turn when rt=1
                    else:
                        num_turns_tensor = 1 + prob_normalscore_tensor.matmul(next_state_value_tensor)

                        # consider bust/finishing for each bust state separately
                    num_turns_array = num_turns_tensor.numpy()
                    for score_gained in range(state_notbust_len, this_throw_state_len):
                        # skip infeasible state
                        if not onb.state_feasible_array[rt, score_gained]:
                            continue
                        score_remain = score_state - score_gained
                        # score_max = min(score_remain-2, 60)
                        score_max = score_remain - 2  # less than 60 here
                        score_max_plus1 = score_max + 1
                        score_gained_index = score_gained - state_notbust_len

                        # transit to the end of game
                        if (rt > 1):
                            if (score_remain == onb.score_DB):
                                num_turns_array[:, score_gained_index] += prob_DB
                            elif (score_remain <= 40 and score_remain % 2 == 0):
                                num_turns_array[:, score_gained_index] += prob_doublescore_dic[score_remain]
                            else:
                                pass

                        # transit to bust
                        if rt == 3:
                            num_turns_array[:, score_gained_index] += prob_bust_dic[score_max]
                            # solve an equation other than using the policy evaluation value (s,i=3,u=0)
                            num_turns_array[:, score_gained_index] = num_turns_array[:, score_gained_index] / \
                                                                     prob_notbust_dic[score_max]
                        elif rt == 2:
                            num_turns_array[:, score_gained_index] += prob_bust_dic[score_max] * (1 + new_value_rt3)
                        else:
                            num_turns_array[:, score_gained_index] += prob_bust_dic[score_max] * (
                                new_value_rt3)  # 1 turn is already counted before

                    # searching
                    temp1 = num_turns_tensor.min(axis=0)
                    state_action_update[rt][state_notbust_len:this_throw_state_len] = temp1.indices.numpy()
                    state_value_update[rt][state_notbust_len:this_throw_state_len] = temp1.values.numpy()

                    # finish rt=1,2,3. check improvement
                action_diff[rt][:] = np.abs(state_action_update[rt] - state_action[rt])
                value_relerror[rt] = np.abs(
                    (state_value_update[rt] - state_value[rt]) / state_value_update[rt]).max()
                state_action[rt][:] = state_action_update[rt][:]
                state_value[rt][:] = state_value_update[rt][:]

            max_action_diff = max([action_diff[1].max(), action_diff[2].max(), action_diff[3].max()])
            max_value_relerror = value_relerror.max()

            if max_action_diff < 1:
                # if max_value_relerror < iteration_relerror_limit:
                num_iteration_record[score_state] = round_index + 1
                break

        for rt in [1, 2, 3]:
            state_value_update[rt][onb.state_infeasible[rt]] = onb.largenumber
            state_action_update[rt][onb.state_infeasible[rt]] = onb.infeasible_marker
        optimal_action_index_dic[score_state] = ft.copy_numberarray_container(state_action_update)
        optimal_value_dic[score_state] = ft.copy_numberarray_container(state_value_update,
                                                                       new_dtype=onb.result_float_dytpe)
        optimal_value_rt3[score_state] = state_value[3][0]
        # done:V(s,i=3/2/1,u)

    #
    prob_scorestate_transit = {}
    prob_scorestate_transit = epag.solve_policy_transit_probability(optimal_action_index_dic, prob_grid_normalscore,
                                                                    prob_grid_doublescore, prob_grid_bullscore)
    t2 = time.time()
    print('solve dp_turn_policyiter in {} seconds'.format(t2 - t1))

    #print(optimal_value_rt3)
    result_dic = {'optimal_value_dic': optimal_value_dic, 'optimal_action_index_dic': optimal_action_index_dic,
                  'optimal_value_rt3': optimal_value_rt3, 'prob_scorestate_transit': prob_scorestate_transit}

    return result_dic


# Solve single player game with the turn feature and store the results in an excel file:
# create an extended version for game states (S,i,u) or DP_turn only with game states (S,i):
def solve_singlegame(name_it, grid_version=onb.grid_version, postfix='', gpu_device=None, extended=False):
    """
    Solve the single player game with the turn feature. Find the optimal policy to minimize the expected number of turns for reaching zero score.
    Args:
        name_it: information item
        data_parameter_dir=onb.data_parameter_dir
        grid_version: the action set and the hitting probability associated with the skill model.
            use 'v2' for the small action set of 984 aiming locations.
            use 'circleboard'  for the action set of the 1mm gird on the entire dartboard, 90,785 aiming locations.
        result_dir: folder to store the result
        postfix='':
        gpu_device: None for CPU computation, otherwise use the gpu device ID defined in the system (default 0).
    Returns:
        a dict or save it.
    """
    # data_parameter_dir=onb.data_parameter_dir,
    # result_dir = None

    file_it = directory_of_action_set + '/v2_item{}'.format(name_it) + "_gaussin_prob_grid.pkl"
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
     prob_grid_bullscore] = epag.load_aiming_grid(file_it)

    if gpu_device is None:
        print('running solve_dp_turn')
        result_dic = solve_dp_turn(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore)
    # else:
    # print('runing gpusolve_dp_turn')
    # result_dic = solve_dp_turn_gpu(aiming_grid, prob_grid_normalscore, prob_grid_doublescore, prob_grid_bullscore,
    # gpu_device=gpu_device)

    # result_dic.keys() :  ['optimal_value_dic', 'optimal_action_index_dic', 'optimal_value_rt3', 'prob_scorestate_transit']

    if not extended:
        scores = [s for s in range(2, 502)]
        rt = [1,2,3]
        game_state = [(i,j) for i in scores for j in rt]
        game_state_dict = {}
        for x in game_state:
            opt_target = result_dic["optimal_action_index_dic"][x[0]][x[1]][0]
            xj, yj = aiming_grid[opt_target][0] - 170, aiming_grid[opt_target][1] - 170
            game_state_dict[x] = (xj,yj)

        output = pd.DataFrame({"Game State (S,i)": game_state,
                               "Optimal target location": list(game_state_dict.values())})

        return output
    else:
        scores = [s for s in range(2, 502)]
        rt = [1, 2, 3]
        state_feasible_rt1 = onb.state_feasible_rt1
        state_feasible_rt2 = onb.state_feasible_rt2
        game_state = [(i,j,l) for i in scores for j in rt for l in state_feasible_rt1 if j == 1]
        game_state = game_state + [(i,j,l) for i in scores for j in rt for l in state_feasible_rt2 if j == 2]
        game_state = game_state + [(i,3,0) for i in scores]
        game_state_dict = {}
        for x in game_state:
            opt_target = result_dic["optimal_action_index_dic"][x[0]][x[1]][x[2]]
            xj, yj = aiming_grid[opt_target][0] - 170, aiming_grid[opt_target][1] - 170
            game_state_dict[x] = (xj, yj)

        output = pd.DataFrame({"Game State (S,i,u)": game_state,
                               "Optimal target location": list(game_state_dict.values())})
        return output

# print(solve_singlegame(1))
# ---------------------------------------------------------------------------------------------------------------
def write_turn():
    DVdB19_DP = solve_singlegame(1)
    DVdB20_DP = solve_singlegame(2)
    DVdB21_DP = solve_singlegame(3)
    AL19_DP = solve_singlegame(4)
    AL20_DP = solve_singlegame(5)
    JdS20_DP = solve_singlegame(6)
    JdS21_DP = solve_singlegame(7)
    DvD20_DP = solve_singlegame(8)
    DvD21_DP = solve_singlegame(9)
    NA18_DP = solve_singlegame(10)
    NA19_DP = solve_singlegame(11)

    with pd.ExcelWriter(my_directory + '/DP_turn_feature_recommendations.xlsx') as writer:
        DVdB19_DP.to_excel(writer, sheet_name="DVdB19")
        DVdB20_DP.to_excel(writer, sheet_name="DVdB20")
        DVdB21_DP.to_excel(writer, sheet_name="DVdB21")
        AL19_DP.to_excel(writer, sheet_name="AL19")
        AL20_DP.to_excel(writer, sheet_name="AL20")
        JdS20_DP.to_excel(writer, sheet_name="JdS20")
        JdS21_DP.to_excel(writer, sheet_name="JdS21")
        DvD20_DP.to_excel(writer, sheet_name="DvD20")
        DvD21_DP.to_excel(writer, sheet_name="DvD21")
        NA18_DP.to_excel(writer, sheet_name="NA18")
        NA19_DP.to_excel(writer, sheet_name="NA19")

# write_turn()

def write_turn_extended():
    DVdB19_DP = solve_singlegame(1,extended=True)
    DVdB20_DP = solve_singlegame(2,extended=True)
    DVdB21_DP = solve_singlegame(3,extended=True)
    AL19_DP = solve_singlegame(4,extended=True)
    AL20_DP = solve_singlegame(5,extended=True)
    JdS20_DP = solve_singlegame(6,extended=True)
    JdS21_DP = solve_singlegame(7,extended=True)
    DvD20_DP = solve_singlegame(8,extended=True)
    DvD21_DP = solve_singlegame(9,extended=True)
    NA18_DP = solve_singlegame(10,extended=True)
    NA19_DP = solve_singlegame(11,extended=True)

    with pd.ExcelWriter(my_directory + '/DP_turn_feature_extended_recommendations.xlsx') as writer:
        DVdB19_DP.to_excel(writer, sheet_name="DVdB19")
        DVdB20_DP.to_excel(writer, sheet_name="DVdB20")
        DVdB21_DP.to_excel(writer, sheet_name="DVdB21")
        AL19_DP.to_excel(writer, sheet_name="AL19")
        AL20_DP.to_excel(writer, sheet_name="AL20")
        JdS20_DP.to_excel(writer, sheet_name="JdS20")
        JdS21_DP.to_excel(writer, sheet_name="JdS21")
        DvD20_DP.to_excel(writer, sheet_name="DvD20")
        DvD21_DP.to_excel(writer, sheet_name="DvD21")
        NA18_DP.to_excel(writer, sheet_name="NA18")
        NA19_DP.to_excel(writer, sheet_name="NA19")

# write_turn_extended()