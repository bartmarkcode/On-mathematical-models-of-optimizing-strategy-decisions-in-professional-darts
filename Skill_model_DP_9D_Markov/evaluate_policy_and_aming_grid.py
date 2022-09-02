import os
import numpy as np
import time

import dartboard as onb
import function_tools as ft

# ============================================================================================================
state_infeasible = onb.state_infeasible
R = onb.R  # radius of the dartboard 170
grid_num = onb.grid_num  # 341

directory_of_score_prob = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/Programms for MA/Score_probabilities_grid'
os.chdir(directory_of_score_prob)
result_dir = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/Programms for MA/result_grid'
# ===========================================================================================================
# ===========================================================================================================
# ALL necessary functions to solve turn transit probabilities.
# These ones are used for the DP approach later on.
def solve_turn_transit_probability(score_state, state_action, prob_normalscore, prob_doublescore, prob_bullscore):
    """
    Solve the state transition probability after a turn playing with a specified aiming policy

    Args:
        score_state: score at the beginning of the turn, e.g., 2,3,...,501
        state_action: a dict of aiming locations (actions in the policy) for each state (s,i,u) in this turn
        prob_normalscore, prob_doublescore, prob_bullscore: the skill model

    Returns: A dict
        result_dict['finish']: probability of finishing the game (reach zero by making a double)
        result_dict['bust']: probability of busting the game (transit to next turn of (s=score_state,i=3,u=0))
        result_dict['score']: probability of achieving a cumulative score_gained in this turn (transit to next turn of
        (s=score_state-score_gained,i=3,u=0))
    """

    result_dict = {}
    prob_finish = 0  # probability of finishing the game
    prob_bust = 0  # probability of busting the game
    # initialize for (s, rt=3, score_gained=0)
    next_throw_state_len = 1
    prob_transit_next_throw_state = np.ones(next_throw_state_len)

    for rt in [3, 2, 1]:
        prob_this_throw_state = prob_transit_next_throw_state
        this_throw_state_len = next_throw_state_len
        next_throw_state_len = min(score_state - 2, onb.maxhitscore * (4 - rt)) + 1
        # probability vector of total score_gained after this throw
        prob_transit_next_throw_state = np.zeros(next_throw_state_len)

        for score_gained in range(this_throw_state_len):
            # skip infeasible state
            if not onb.state_feasible_array[rt, score_gained]:
                continue

            # aiming location of the policy at this state
            aiming_location_index = state_action[rt][score_gained]
            prob_this_state = prob_this_throw_state[score_gained]

            # largest possible normal score to make in the next throw without busting
            score_remain = score_state - score_gained
            score_max = min(score_remain - 2, 60)
            score_max_plus1 = score_max + 1

            # transit to next throw or turn with normal scores
            prob_transit_next_throw_state[score_gained:score_gained + score_max_plus1] += \
                prob_normalscore[aiming_location_index, 0:score_max_plus1] * prob_this_state
            # game can not bust or end when score_max = 60, i.e.,  prob_notbust = 1
            if score_max < 60:
                prob_notbust_this_state = prob_normalscore[aiming_location_index, 0:score_max + 1].sum()
                # transit to the end of game
                if score_remain == onb.score_DB:
                    prob_finish += prob_bullscore[aiming_location_index, 1] * prob_this_state
                    prob_notbust_this_state += prob_bullscore[aiming_location_index, 1]
                elif score_remain <= 40 and score_remain % 2 == 0:
                    doublescore_index = (score_remain // 2) - 1
                    prob_finish += prob_doublescore[aiming_location_index, doublescore_index] * prob_this_state
                    prob_notbust_this_state += prob_doublescore[aiming_location_index, doublescore_index]
                else:
                    pass

                # transit to bust
                prob_bust += (max(1 - prob_notbust_this_state, 0)) * prob_this_state

    result_dict['finish'] = prob_finish
    result_dict['bust'] = prob_bust
    result_dict['score'] = prob_transit_next_throw_state

    return result_dict


# ===================================================================================================================
def solve_turn_transit_probability_fast(score_state, state_action, prob_normalscore, prob_doublescore, prob_bullscore,
                                        prob_bust_dic):
    """
    A fast way of implementing solve_turn_transit_probability by using pre-stored prob_bust_dic
    """

    result_dict = {}
    prob_finish = 0  # probability of finishing the game
    prob_bust_total = 0  # probability of busting the game
    # initialize for (s, rt=3, score_gained=0)
    next_throw_state_len = 1
    prob_transit_next_throw_state = np.ones(next_throw_state_len)

    for rt in [3, 2, 1]:
        prob_this_throw_state = prob_transit_next_throw_state
        this_throw_state_len = next_throw_state_len
        next_throw_state_len = min(score_state - 2, onb.maxhitscore * (4 - rt)) + 1
        prob_transit_next_throw_state = np.zeros(
            next_throw_state_len)  # probability vector of total score_gained after this throw

        prob_normalscore_transit = prob_normalscore[
                                       state_action[rt][0:this_throw_state_len]] * prob_this_throw_state.reshape(
            (this_throw_state_len, 1))

        for score_gained in range(this_throw_state_len):  # loop through score already gained
            # skip infeasible state
            if not onb.state_feasible_array[rt, score_gained]:
                continue

                # aiming location of the policy at this state
            aiming_location_index = state_action[rt][score_gained]
            prob_this_state = prob_this_throw_state[score_gained]

            # largest possible normal score to make in the next throw without busting
            score_remain = score_state - score_gained
            score_max = min(score_remain - 2, 60)
            score_max_plus1 = score_max + 1

            # transit to next throw or turn with normal scores
            prob_transit_next_throw_state[score_gained:score_gained + score_max_plus1] += prob_normalscore_transit[
                                                                                          score_gained,
                                                                                          0:score_max_plus1]
            # game can not bust or end when score_max = 60, i.e.,  prob_notbust = 1
            if score_max < 60:
                # transit to the end of game
                if score_remain == onb.score_DB:
                    prob_finish += prob_bullscore[aiming_location_index, 1] * prob_this_state
                elif score_remain <= 40 and score_remain % 2 == 0:
                    doublescore_index = (score_remain // 2) - 1
                    prob_finish += prob_doublescore[aiming_location_index, doublescore_index] * prob_this_state
                else:
                    pass

                # transit to bust
                prob_bust_total += prob_bust_dic[score_max][aiming_location_index] * prob_this_state

    result_dict['finish'] = prob_finish
    result_dict['bust'] = prob_bust_total
    result_dict['score'] = prob_transit_next_throw_state

    return result_dict


# =============================================================================================================
def solve_policy_transit_probability(policy_action_index_dic, prob_grid_normalscore, prob_grid_doublescore,
                                     prob_grid_bullscore):
    """
    For each turn, solve the state transition probability for a specified aiming policy

    Args:
        policy_action_index_dic: a dict of aiming locations (actions in the policy) for each state (s,i,u) of
        each turn s=2,...,501
        prob_normalscore, prob_doublescore, prob_bullscore: the skill model

    Returns: A dict
    """

    prob_policy_transit_dict = {}
    t1 = time.time()
    for score_state in range(2, 502):
        prob_policy_transit_dict[score_state] = solve_turn_transit_probability(score_state,
                                                                               policy_action_index_dic[score_state],
                                                                               prob_grid_normalscore,
                                                                               prob_grid_doublescore,
                                                                               prob_grid_bullscore)

    t2 = time.time()
    print('solve prob_policy_transit in {} seconds'.format(t2 - t1))

    return prob_policy_transit_dict

# ===================================================================================================================
# ===================================================================================================================
# LOADING probability grid from 'score_prob_player.py'
# CREATING the adapted action set
# STORING the adapted action set

# 3-dimension probability grid
def load_prob_grid(item_filename):  # p(x,y | mu)
    """
    Load 3-dimensional numpy arrays of size 341*341*si (the 340mmX340mm square grid enclosing the dartboard).
    Generate prob_grid_normalscore which has 61 columns and contains the aggregated hitting probability
    of score 0,1...,60. For example, P(score 18) = P(S18) + P(D9) + P(T6)
    """

    prob_grid_dict = ft.load_pickle(item_filename, printflag=True)
    prob_grid_singlescore = prob_grid_dict['prob_grid_singlescore']
    prob_grid_doublescore = prob_grid_dict['prob_grid_doublescore']
    prob_grid_triplescore = prob_grid_dict['prob_grid_triplescore']
    prob_grid_bullscore = prob_grid_dict['prob_grid_bullscore']

    # prob_grid_singlescore has 61 columns and contains the aggregated hitting probability of score 0,1...,60.
    # For example, P(score 18) = P(S18) + P(D9) + P(T6)
    # Bull score is NOT included yet !!
    prob_grid_normalscore = np.zeros((grid_num, grid_num, 61))
    for temp_s in range(1, 61):
        if temp_s <= 20:
            prob_grid_normalscore[:, :, temp_s] = prob_grid_singlescore[:, :, temp_s - 1]
        if temp_s % 2 == 0 and temp_s <= 40:
            prob_grid_normalscore[:, :, temp_s] = prob_grid_normalscore[:, :, temp_s] + \
                                                  prob_grid_doublescore[:, :, temp_s // 2 - 1]
        if temp_s % 3 == 0:
            prob_grid_normalscore[:, :, temp_s] = prob_grid_normalscore[:, :, temp_s] + \
                                                  prob_grid_triplescore[:, :, temp_s // 3 - 1]
    # prob of hitting zero
    prob_grid_normalscore[:, :, 0] = np.maximum(0, 1 - prob_grid_normalscore[:, :, 1:].sum(axis=2) -
                                                prob_grid_bullscore.sum(axis=2))

    return [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore,
            prob_grid_triplescore, prob_grid_bullscore]


# print(load_prob_grid("item1_gaussin_prob_grid.pkl"))
# ===============================================================================================================
# generate adapted action set 'v2' (984 target points)
def get_aiming_grid_v2(item_filename):
    """
    Select 984 aiming locations to build a small action set.
    Eighteen targets in each of the double and treble regions are permitted for a total of 720 double / treble targets.
    A small square enclosing the SB and DB regions contains 9 × 9 = 81 additional targets.
    Three targets located in each of the inner and outer and single regions are permitted.
    This leads to a total of 720+81+120 = 921 common targets for each of the players.
    For each particular player, we also include the target within each region, e.g. T20, S12 etc.,
     that has the highest probability of being hit for that player.
    Finally we also include the single point on the board that has the highest expected score which is generally in T20.
    This means an additional 63 targets for each player.
    Totally 921+63=984 points.
    """

    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
     prob_grid_bullscore] = load_prob_grid(item_filename)
    temp_num = 1000
    aiming_grid = np.zeros((temp_num, 2), dtype=np.int32)

    # x,y location in [-170,170]. x,y index in [0,340].
    # center of DB is the first one
    temp_index = 0
    aiming_grid[temp_index, 0] = R
    aiming_grid[temp_index, 1] = R

    # square enclosing SB. 4mm*4mm grid
    for temp_x in range(-16, 16 + 1, 4):
        for temp_y in range(-16, 16 + 1, 4):
            if temp_x == 0 and temp_y == 0:
                continue
            else:
                temp_index += 1
                aiming_grid[temp_index] = [temp_x + R, temp_y + R]

    # inner single area and outer single area
    rgrid = [58, 135]
    theta_num = 60
    for theta in range(theta_num):
        theta = np.pi * (2.0 * theta / theta_num)
        for temp_r in rgrid:
            temp_index += 1
            temp_x = int(np.round(np.cos(theta) * temp_r))
            temp_y = int(np.round(np.sin(theta) * temp_r))
            aiming_grid[temp_index] = [temp_x + R, temp_y + R]

    # double and triple area
    rgrid = [100, 103, 106, 163, 166, 169]
    theta_num = 120
    for theta in range(theta_num):
        theta = np.pi * (2.0 * theta / theta_num)
        for temp_r in rgrid:
            temp_index += 1
            temp_x = int(np.round(np.cos(theta) * temp_r))
            temp_y = int(np.round(np.sin(theta) * temp_r))
            aiming_grid[temp_index] = [temp_x + R, temp_y + R]

    # all players share the above common points.
    # the following points are different among players

    # maximize probability point for each score region
    # single area
    for temp_s in range(onb.singlescorelist_len):
        temp_index += 1
        temp_s_argmax = np.argmax(prob_grid_singlescore[:, :, temp_s])
        temp_x = temp_s_argmax / grid_num
        temp_y = temp_s_argmax % grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]
    # double area
    for temp_s in range(onb.doublescorelist_len):
        temp_index += 1
        temp_s_argmax = np.argmax(prob_grid_doublescore[:, :, temp_s])
        temp_x = temp_s_argmax / grid_num
        temp_y = temp_s_argmax % grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]
    # triple area
    for temp_s in range(onb.triplescorelist_len):
        temp_index += 1
        temp_s_argmax = np.argmax(prob_grid_triplescore[:, :, temp_s])
        temp_x = temp_s_argmax / grid_num
        temp_y = temp_s_argmax % grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]
    # bull
    for temp_s in range(onb.bullscorelist_len):
        temp_index += 1
        temp_s_argmax = np.argmax(prob_grid_bullscore[:, :, temp_s])
        temp_x = temp_s_argmax / grid_num
        temp_y = temp_s_argmax % grid_num
        aiming_grid[temp_index] = [temp_x, temp_y]

    # max expected score
    temp_index += 1
    e_score = prob_grid_normalscore.dot(np.arange(61)) + prob_grid_bullscore.dot(np.array([onb.score_SB, onb.score_DB]))
    max_e_score = np.max(e_score)
    temp_argmax = np.argmax(e_score)
    temp_x = temp_argmax / grid_num
    temp_y = temp_argmax % grid_num
    aiming_grid[temp_index] = [temp_x, temp_y]
    print('max_e_score={}, max_e_score_index={}'.format(max_e_score, aiming_grid[temp_index]))

    # [0, 340]
    aiming_grid_num = temp_index + 1  # 984
    # cutting aiming grid of length 1000 (set above) to len = aiming_grid_num
    aiming_grid = aiming_grid[:aiming_grid_num, :]
    aiming_grid = np.maximum(aiming_grid, 0)
    aiming_grid = np.minimum(aiming_grid, grid_num - 1)

    # return probability
    prob_grid_normalscore_new = np.zeros((aiming_grid_num, 61))
    prob_grid_singlescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_doublescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_triplescore_new = np.zeros((aiming_grid_num, 20))
    prob_grid_bullscore_new = np.zeros((aiming_grid_num, 2))
    for temp_index in range(aiming_grid_num):
        prob_grid_normalscore_new[temp_index, :] = prob_grid_normalscore[aiming_grid[temp_index, 0],
                                                   aiming_grid[temp_index, 1], :]
        prob_grid_singlescore_new[temp_index, :] = prob_grid_singlescore[aiming_grid[temp_index, 0],
                                                   aiming_grid[temp_index, 1], :]
        prob_grid_doublescore_new[temp_index, :] = prob_grid_doublescore[aiming_grid[temp_index, 0],
                                                   aiming_grid[temp_index, 1], :]
        prob_grid_triplescore_new[temp_index, :] = prob_grid_triplescore[aiming_grid[temp_index, 0],
                                                   aiming_grid[temp_index, 1], :]
        prob_grid_bullscore_new[temp_index, :] = prob_grid_bullscore[aiming_grid[temp_index, 0],
                                                 aiming_grid[temp_index, 1], :]

    return [aiming_grid, prob_grid_normalscore_new, prob_grid_singlescore_new, prob_grid_doublescore_new,
            prob_grid_triplescore_new, prob_grid_bullscore_new]


# print(get_aiming_grid_v2("item1_gaussin_prob_grid.pkl"))
# ===============================================================================================================
# save action set v2
def save_aiming_grid_v2(information_item):
    grid_version_result = 'v2'
    print('generate and save action set grid_version={}'.format(grid_version_result))
    for item in information_item:
        name_it = 'item{}'.format(item) + "_gaussin_prob_grid.pkl"
        [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
         prob_grid_bullscore] = get_aiming_grid_v2(name_it)

        postfix = ''
        info = 'SB={} DB={} R1={} postfix={} skillmodel=full grid_version={}'.format(onb.score_SB, onb.score_DB, onb.R1,
                                                                                     postfix, grid_version_result)
        result_dic = {}
        result_dic['info'] = info
        result_dic['aiming_grid'] = aiming_grid
        result_dic['prob_grid_normalscore'] = prob_grid_normalscore
        result_dic['prob_grid_singlescore'] = prob_grid_singlescore
        result_dic['prob_grid_doublescore'] = prob_grid_doublescore
        result_dic['prob_grid_triplescore'] = prob_grid_triplescore
        result_dic['prob_grid_bullscore'] = prob_grid_bullscore

        # result_dir = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/Programms for MA/result_grid'
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        result_filename = result_dir + '/{}_{}'.format(grid_version_result, name_it)
        ft.dump_pickle(result_filename, result_dic, printflag=True)

    print()
    return

# save_aiming_grid_v2([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# save_aiming_grid_v2([1])
# ================================================================================================================

# 2-dimension probability grid
def load_aiming_grid(item_filename, count_bull=True):
    """
    Load 2-dimensional numpy arrays of hitting probability from files
    Each row of aiming_grid is the (x-index, y-index) of an aiming location.
    For each aiming location, the corresponding row in prob_grid_singlescore (same row index as that in aiming_grid)
     contains the hitting probability of score S1,...,S20.
    (prob_grid_doublescore for D1,...,D20, prob_grid_triplescore for T1,...,T20,, prob_grid_bullscore for SB,DB)
    prob_grid_normalscore has 61 columns and contains the aggregated hitting probability of score 0,1...,60.
     For example, P(score 18) = P(S18) + P(D9) + P(T6)

    """

    os.chdir(result_dir)
    result_dic = ft.load_pickle(item_filename, printflag=True)
    aiming_grid = result_dic['aiming_grid']
    prob_grid_normalscore = result_dic['prob_grid_normalscore']
    prob_grid_singlescore = result_dic['prob_grid_singlescore']
    prob_grid_doublescore = result_dic['prob_grid_doublescore']
    prob_grid_triplescore = result_dic['prob_grid_triplescore']
    prob_grid_bullscore = result_dic['prob_grid_bullscore']

    # default setting counts bull score
    if count_bull:
        prob_grid_normalscore[:, onb.score_SB] += prob_grid_bullscore[:, 0]
        prob_grid_normalscore[:, onb.score_DB] += prob_grid_bullscore[:, 1]
    else:
        print('bull score in NOT counted in prob_grid_normalscore')

    return [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
            prob_grid_bullscore]
