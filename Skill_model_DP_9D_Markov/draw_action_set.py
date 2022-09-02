from matplotlib import pyplot as plt
import numpy as np
import evaluate_policy_and_aming_grid as epag
import math
import nine_d as nd
import DP


directory_of_action_set = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/Programms for MA/result_grid'
# ============================================================================================================
# DRAW the adapted action set with the 984 feasible target points for a given information item in a dartboard:
def draw_action_set(item):
    file_it = directory_of_action_set + '/v2_item{}'.format(item) + "_gaussin_prob_grid.pkl"
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
     prob_grid_bullscore] = epag.load_aiming_grid(file_it)

    theta = []
    theta2 = []
    r = []
    r2 = []
    for j in range(len(aiming_grid) - 63):
        xj, yj = aiming_grid[j][0] - 170, aiming_grid[j][1] - 170
        theta_j = math.atan2(yj, xj)
        r_j = math.sqrt(xj * xj + yj * yj)
        theta.append(theta_j)
        r.append(r_j)
    for jj in range(921, len(aiming_grid)):
        xj, yj = aiming_grid[jj][0] - 170, aiming_grid[jj][1] - 170
        theta_j = math.atan2(yj, xj)
        r_j = math.sqrt(xj * xj + yj * yj)
        theta2.append(theta_j)
        r2.append(r_j)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")

    ax.scatter(theta, r, s=4, c="c")
    ax.scatter(theta2, r2, s=4, c="b")

    ax.set_xticks(np.arange(np.pi / 20, 2 * np.pi + np.pi / 20, np.pi / 10))
    ax.set_yticks(np.array([6.4, 15.9, 99, 107, 162, 170, 190]))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    plt.show()


# draw_action_set(1)
# ================================================================================================================
# DRAW the points in a dartboard that are part of the optimal track
# for the information item 'item' when he is on total score 'score'


def draw_track(item, score, dp=False):
    file_it = directory_of_action_set + '/v2_item{}'.format(item) + "_gaussin_prob_grid.pkl"
    [aiming_grid, prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore,
     prob_grid_bullscore] = epag.load_aiming_grid(file_it)

    # Output the optimal track from 9D
    if not dp:
        track = nd.algo_9D(score,nd.allowed_fields,nd.allowed_doubles,nd.skill_dict(nd.allowed_fields,item),nd.num_z,memo={})
    else: # Output the optimal track from DP
        track = DP.solve_noturn(item)["Track"][score-2]
    theta = []
    r = []
    for t in track:
        if t == "SB25":
            temp_s_argmax = np.argmax(prob_grid_bullscore[:, 0])
            xj, yj = aiming_grid[temp_s_argmax][0] - 170, aiming_grid[temp_s_argmax][1] - 170
            theta_j = math.atan2(yj, xj)
            r_j = math.sqrt(xj * xj + yj * yj)
            theta.append(theta_j)
            r.append(r_j)
        elif t == "DB25":
            temp_s_argmax = np.argmax(prob_grid_bullscore[:, 1])
            xj, yj = aiming_grid[temp_s_argmax][0] - 170, aiming_grid[temp_s_argmax][1] - 170
            theta_j = math.atan2(yj, xj)
            r_j = math.sqrt(xj * xj + yj * yj)
            theta.append(theta_j)
            r.append(r_j)
        else:
            st = t[0]
            nr = int(t.removeprefix(st))
            if st == "S":
                temp_s_argmax = np.argmax(prob_grid_singlescore[:, nr - 1])
                xj, yj = aiming_grid[temp_s_argmax][0] - 170, aiming_grid[temp_s_argmax][1] - 170
                theta_j = math.atan2(yj, xj)
                r_j = math.sqrt(xj * xj + yj * yj)
                theta.append(theta_j)
                r.append(r_j)
            elif st == "D":
                temp_s_argmax = np.argmax(prob_grid_doublescore[:, nr - 1])
                xj, yj = aiming_grid[temp_s_argmax][0] - 170, aiming_grid[temp_s_argmax][1] - 170
                theta_j = math.atan2(yj, xj)
                r_j = math.sqrt(xj * xj + yj * yj)
                theta.append(theta_j)
                r.append(r_j)
            elif st == "T":
                temp_s_argmax = np.argmax(prob_grid_triplescore[:, nr - 1])
                xj, yj = aiming_grid[temp_s_argmax][0] - 170, aiming_grid[temp_s_argmax][1] - 170
                theta_j = math.atan2(yj, xj)
                r_j = math.sqrt(xj * xj + yj * yj)
                theta.append(theta_j)
                r.append(r_j)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    if not dp:
        color = "brown"
    else:
        color = "orange"
    ax.scatter(theta, r, s=8, c=color)
    plt.title(track, fontsize=15)
    for i, label in enumerate(track):
        plt.annotate(label, (theta[i], r[i]))

    ax.set_xticks(np.arange(np.pi / 20, 2 * np.pi + np.pi / 20, np.pi / 10))
    ax.set_yticks(np.array([6.4, 15.9, 99, 107, 162, 170, 190]))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.show()

# draw_track(6,199,dp=True)
# draw_track(6,199)
