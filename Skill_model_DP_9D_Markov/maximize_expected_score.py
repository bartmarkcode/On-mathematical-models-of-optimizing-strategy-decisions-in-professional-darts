import function_tools as ft
import dartboard as onb
import evaluate_policy_and_aming_grid as epag
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

directory_of_score_prob = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/Programms for MA/Score_probabilities_grid'
# =====================================================================================================================
# Compute expected scores over the whole 1mm grid:
def exp_score_grid(item):
    item_filename = directory_of_score_prob + '/item{}_gaussin_prob_grid.pkl'.format(item)
    [prob_grid_normalscore, prob_grid_singlescore, prob_grid_doublescore, prob_grid_triplescore, prob_grid_bullscore] = \
        epag.load_prob_grid(item_filename)
    # prob_grid_normalscore: 341x341x61
    e_score = prob_grid_normalscore.dot(np.arange(61)) + prob_grid_bullscore.dot(np.array([onb.score_SB, onb.score_DB]))
    #print(len(prob_grid_normalscore[0][0]))
    return e_score   # 341x341

#print(exp_score_grid(1))
# ===================================================================================================================
# Compute the maximum expected score, the corresponding field, its corresponding numerical score and the (x,y) position:
def max_exp_score(item):
    grid = exp_score_grid(item)

    max_e_score = np.max(grid)
    argmax_e_score = np.argmax(grid)
    temp_x = argmax_e_score // 341
    temp_y = argmax_e_score % 341

    target_score = int(onb.score_matrix()[temp_x, temp_y])
    # It's trivial that a maximum expected score can only be achieved by aiming for a treble and not for a single
    # Using our adapted action, it is possible that optimal aiming locations for T20 lie on the wire of its field and
    # 'dartboard.py' assigns them to S20. Hence, we have to correct this little issue, if it happens.
    if target_score in onb.singlescorelist:
        target_score = target_score*3
    xy_grid_pos = temp_x -170, temp_y-170

    field = onb.get_score_and_multiplier(temp_x-170, temp_y-170)
    if field[1] == 1:
        field = "S" + str(int(field[0]))
    elif field[1] == 2:
        field = "D" + str(int(field[0]/2))
    elif field[1] == 3:
        field = "T" + str(int(field[0]/3))
    return max_e_score, field, target_score, xy_grid_pos

#for i in range(1,12):
    #print("i:", i)
    #print(max_exp_score(i))

#-------------------------------------------------------------------------------------------------------------
# CREATE the heat maps for a given information item
def heatmap(item):
    val = exp_score_grid(item)
    # fig = plt.figure()
    ax = sns.heatmap(np.rot90(val, 1), xticklabels=False, yticklabels=False, cmap="flare",
                     cbar_kws={'label': 'Expected Score',"location":"left"}, vmin=0, vmax=np.max(val))

    # ax.invert_yaxis()
    # ax2 = fig.add_subplot(111, projection="polar")
    # ax2.set_xticks(np.arange(np.pi/20,2*np.pi+np.pi/20,np.pi/10))
    # ax2.set_yticks(np.array([6.4,15.9,99,107,162,170]))
    # ax2.xaxis.set_ticklabels([])
    # ax2.yaxis.set_ticklabels([])

    plt.show()

# heatmap(1)