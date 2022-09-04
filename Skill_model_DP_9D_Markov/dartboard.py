import math
import numpy as np


# ===============================================================================================================

grid_version = 'v2'  # adapted action set with 984 feasible targets
result_float_dytpe = None  # store results in default float type (usually np.float64)

# ===============================================================================================================
# Dartboard layout
R1 = 6.35  # distance to double bullseye
R2 = 15.9  # distance to single bullseye
R3 = 99  # distance to inside triple ring
R4 = 107  # distance to outside triple ring
R5 = 162  # distance to inside double ring
R = 170  # distance to outside double ring

# ordering of the scores from top clock-wise
d = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# score list
score_SB = 25
score_DB = 2 * score_SB
singlescorelist = [i for i in range(1, 21)]
doublescorelist = [i * 2 for i in singlescorelist]
triplescorelist = [i * 3 for i in singlescorelist]
bullscorelist = [score_SB, score_DB]
singlescorelist_len = 20
doublescorelist_len = 20
triplescorelist_len = 20
bullscorelist_len = 2
# create list of all scores without multiples:
allscorelist = list(set([0] + singlescorelist + doublescorelist + triplescorelist + bullscorelist))
allscorelist.sort()
allscorelist_len = len(allscorelist)  # 44

maxhitscore = max(allscorelist)  # 60

#  states in turn
# rt: number of remaining throws in a turn, i.e., 3,2,1
# sg: score already gained in a turn
largenumber = 1000
infeasible_marker = -1

state_feasible_rt3 = [0]
state_infeasible_rt3 = []

state_feasible_rt2 = [i for i in allscorelist]
state_infeasible_rt2 = [i for i in range(maxhitscore * 1) if i not in state_feasible_rt2]

state_feasible_rt1 = []
for score_gained in state_feasible_rt2:
    for temp_score in allscorelist:
        state_feasible_rt1.append(score_gained + temp_score)
state_feasible_rt1 = list(set(state_feasible_rt1))
state_feasible_rt1.sort()  # [0,...120] with some exceptions e.g. 109,112
state_infeasible_rt1 = [i for i in range(maxhitscore * 2) if i not in state_feasible_rt1]

state_feasible = [None, state_feasible_rt1, state_feasible_rt2, state_feasible_rt3]
# print(len(state_feasible[1])+len(state_feasible[2])+len(state_feasible[3]))
# ----> [None, [0,...,120],[0,...,60],[0]]
state_infeasible = [None, state_infeasible_rt1, state_infeasible_rt2, state_infeasible_rt3]
# ----> [None,[Exc.],[Exc.],[]]

state_feasible_array = np.zeros((4, 2 * maxhitscore + 1), dtype=bool)
for rt in [1, 2, 3]:
    for score_gained in state_feasible[rt]:
        state_feasible_array[rt, score_gained] = True


# --> [rt = 1: [False False ....


def check_state_feasiblility_turngame(rt, score_gained):
    return state_feasible_array[rt, score_gained]


# ============================================================================================================

# 1mm grid on a square enclosing the circle dartboard
# grid for 341*341 aiming location
# x is the horizon axis (column index) and y is the vertical axis (row index).
# Taking the center of the DB region to be the origin (x=0,y=0)
# x, y location values (xgrid, ygrid): -170, -169, ..., -1, 0, 1, ..., 169, 170
# x, y index values (starting from 0): 0, 1, ..., 169, 170, 171, 339, 340

pixel_per_mm = 1  # 1mm grid
grid_num = int(2 * R * pixel_per_mm) + 1  # (341)
grid_width = 1.0 / pixel_per_mm  # (1)
# x coordinate left to right increasing
xindex = range(grid_num)  # (0-341)
xgrid = np.arange(grid_num) * grid_width - R  # [-170,-169,...,169,170]
# y coordinate top to bottom increasing
yindex = xindex[:]  # (0-341)
ygrid = xgrid[:]  # [-170,-169,...,169,170]

# default 1mm grid
def get_1mm_grid():
    return [xindex, yindex, xgrid, ygrid, grid_num]

# ============================================================================================================
# get index of aiming locations inside or outside of the circle board

def get_index_circle(inside_or_outside):
    radius_matrix = np.zeros((grid_num, grid_num))
    radius_matrix = radius_matrix + xgrid.reshape((1, grid_num)) ** 2
    radius_matrix = radius_matrix + ygrid[::-1].reshape((grid_num, 1)) ** 2
    print(radius_matrix)
    if inside_or_outside.startswith('out'):
        index_circle = np.where(radius_matrix > R ** 2)
    if inside_or_outside.startswith('in'):
        index_circle = np.where(radius_matrix <= R ** 2)
    return index_circle


# =========================================================================================================
# score function
# input:  location (x,y)
# output: numerical score

def get_score(x, y):
    # compute the radius
    r = math.sqrt(x * x + y * y)
    # check if it's off the board (do this for speed)
    if r > R:
        return 0
    # check for a double bullseye
    if r <= R1:
        return score_DB
    # check for a single bullseye
    if r <= R2:
        return score_SB
    # get the angle
    th = math.atan2(y, x)
    phi = (math.pi / 2 + math.pi / 20 - th) % (2 * math.pi)
    if phi < 0:
        phi += 2 * math.pi

    # now get the number
    i = int(phi / (2 * math.pi) * 20)
    if i < 0:
        i = 0
    if i >= 19:
        i = 19

    n = d[i]
    # check for a single
    if r <= R3:
        return n
    # check for a triple
    if r <= R4:
        return 3 * n
    # check for a single
    if r <= R5:
        return n
    # if we got here, it must be a double
    return 2 * n

# print(get_score(212-170,301-170))
# print(get_score(166 * math.cos(72 / 180 * math.pi), 166 * math.sin(72 / 180 * math.pi)))
# ===========================================================================================================
# input:  location (x,y)
# output: return numerical score if (x,y) is in the single area, otherwise 0

def get_score_singleonly(x, y):
    r = math.sqrt(x * x + y * y)
    if r > R:
        return 0
    if r <= R2:
        return 0
    th = math.atan2(y, x)
    phi = (math.pi / 2 + math.pi / 20 - th) % (2 * math.pi)
    if phi < 0:
        phi += 2 * math.pi
    i = int(phi / (2 * math.pi) * 20)
    if i < 0:
        i = 0
    if i >= 19:
        i = 19
    n = d[i]
    if r <= R3:
        return n
    if r <= R4:
        return 0
    if r <= R5:
        return n
    return 0


# ===========================================================================================================
# input:  location (x,y)
# output: return numerical score if (x,y) is in the double ring, otherwise 0

def get_score_doubleonly(x, y):
    r = math.sqrt(x * x + y * y)
    if r <= R5:
        return 0
    if r > R:
        return 0
    th = math.atan2(y, x)
    phi = (math.pi / 2 + math.pi / 20 - th) % (2 * math.pi)
    if phi < 0:
        phi += 2 * math.pi
    i = int(phi / (2 * math.pi) * 20)
    if i < 0:
        i = 0
    if i >= 19:
        i = 19
    n = d[i]
    return 2 * n


# ==============================================================================================================
# input:  location (x,y)
# output: return numerical score if (x,y) is in the triple ring, otherwise 0

def get_score_tripleonly(x, y):
    r = math.sqrt(x * x + y * y)
    if r <= R3:
        return 0
    if r > R4:
        return 0
    th = math.atan2(y, x)
    phi = (math.pi / 2 + math.pi / 20 - th) % (2 * math.pi)
    if phi < 0:
        phi += 2 * math.pi
    i = int(phi / (2 * math.pi) * 20)
    if i < 0:
        i = 0
    if i >= 19:
        i = 19
    n = d[i]
    return 3 * n


# ===============================================================================================================
# input:  location (x,y)
# output: return numerical score if (x,y) is in the bull area, otherwise 0

def get_score_bullonly(x, y):
    r = math.sqrt(x * x + y * y)
    if r <= R1:
        return score_DB
    if r <= R2:
        return score_SB
    return 0


# ========================================================================================================
# input:  location (x,y)
# output: return numerical score as well as the corresponding multiplier (e.g. [60,3] for T20)

def get_score_and_multiplier(x_input, y_input=None):
    if y_input is None:
        # input is an array or a list containing (x,y) at x_input location
        x = x_input[0]
        y = x_input[1]
    else:
        # input is (x,y)
        x = x_input
        y = y_input

    r = math.sqrt(x * x + y * y)
    if r > R:
        return [0, 1]
    if r <= R1:
        return [score_DB, 2]
    if r <= R2:
        return [score_SB, 1]

    th = math.atan2(y, x)
    phi = (math.pi / 2 + math.pi / 20 - th) % (2 * math.pi)
    if phi < 0:
        phi += 2 * math.pi
    i = int(phi / (2 * math.pi) * 20)
    if i < 0:
        i = 0
    if i >= 19:
        i = 19
    n = d[i]
    if r <= R3:
        multiplier = 1
        return [n * multiplier, multiplier]
    if r <= R4:
        multiplier = 3
        return [n * multiplier, multiplier]
    if r <= R5:
        multiplier = 1
        return [n * multiplier, multiplier]
    # if we got here, it must be a double
    multiplier = 2
    return [n * multiplier, multiplier]

# print(get_score_and_multiplier(-1,100))

# ================================================================================================================
def get_score_and_multiplier_fromindex(x_input, y_input=None):
    if y_input is None:
        # input is an array or a list containing (x,y) at x_input location
        x = x_input[0]
        y = x_input[1]
    else:
        # input is (x,y)
        x = x_input
        y = y_input
    # index range [0,340], coordinate range [-170,170]. index - 170 = coordinate
    return get_score_and_multiplier(x - R, y - R)


# ============================================================================================================
# defining a grid over the board 341mm x 341mm and output which score is achieved at a specific point of the grid.

def score_matrix():
    M = np.zeros((grid_num, grid_num))
    for i in range(grid_num):
        for j in range(grid_num):
            M[i, j] = get_score(i - 170, j - 170)

    return M

# print(score_matrix())
