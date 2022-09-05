import os
import numpy as np
import time
from scipy.stats import multivariate_normal

import dartboard as onb
import function_tools as ft
import read_in_sigma as ris

# ================================================================================================================
my_directory = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit'
result_folder = 'Programms for MA/Score_probabilities_grid'
# ================================================================================================================
# CREATE the distributions p(x,y | mu) for an information item
def evaluate_score_probability(information_item):
    """
    This function conducts a numerical integration to evaluate the hitting probability of each score
    (each score segment in the dartboard) associated with each aiming location on the 1mm-grid.

    Args:
        A list of information items to evaluate, e.g., [1,2] for information items 1 and 2.

    Returns:
            A dict of four numpy arrays: prob_grid_singlescore, prob_grid_doublescore,
            prob_grid_triplescore, prob_grid_bullscore.
            prob_grid_singlescore[xi,yi,si] is of size 341*341*20.
            (prob_grid_doublescore and prob_grid_triplescore have the similar structure.)
            xi and yi are the x-axis and y-axis indexes (starting from 0) of the square 1mm grid enclosing
            the circle dartboard.
            si = 0,1,...,19 for score S1,S2,...,S20
            prob_grid_bullscore[xi,yi,si] is of size 341*341*2, where si=0 represents SB and si=1 represents DB
            For example, when aiming at the center of the dartboard,
            prob_grid_singlescore[xi=170,yi=170,si=9] is the probability of hitting S10
            prob_grid_doublescore[xi=170,yi=170,si=0] is the probability of hitting D1
            prob_grid_triplescore[xi=170,yi=170,si=7] is the probability of hitting T8
            prob_grid_bullscore[xi=170,yi=170,si=0] is the probability of hitting SB
            prob_grid_bullscore[xi=170,yi=170,si=1] is the probability of hitting DB

            Results are stored in the folder .Programms for MA/Score_probabilities_grid
    """
    os.chdir(my_directory)
    result_dir = my_directory + '/' + result_folder
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    # 1mm-width grid of 341*341 aiming locations (a square enclosing the circle dart board)
    [xindex, yindex, xgrid, ygrid, grid_num] = onb.get_1mm_grid()
    f_density_grid_pixel_per_mm = 1
    f_density_grid_num = int(2 * onb.R * f_density_grid_pixel_per_mm) + 1  # (341)
    # print(f_density_grid_num)
    f_density_grid_width = 1.0 / f_density_grid_pixel_per_mm  # (1)
    f_density_constant = f_density_grid_width * f_density_grid_width  # (1)

    print('f_density_grid_num={} f_density_grid_width={}'.format(f_density_grid_num, f_density_grid_width))

    # f_density_grid x coordinate left to right increasing
    f_density_xindex = range(f_density_grid_num)
    f_density_xgrid = np.arange(f_density_grid_num) * f_density_grid_width - onb.R  # ([-170,-160,...,0,...,160,170])
    # y coordinate top to bottom increasing
    f_density_yindex = f_density_xindex[:]
    f_density_ygrid = f_density_xgrid[:]

    # build f_density_grid, x is the horizon axis (column index) and y is the vertical axis (row index).
    # Hence, y is at first
    y, x = np.mgrid[-onb.R:onb.R + 0.1 * f_density_grid_width:f_density_grid_width,
           -onb.R:onb.R + 0.1 * f_density_grid_width:f_density_grid_width]
    pos = np.dstack((x, y))  # [[[-170 -170] [-160 -170]...[170 -170]] ... [[-170 170] ...[170 170]]]

    # score information on the f_density_grid
    singlescore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    doublescore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    triplescore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    bullscore_grid = np.zeros((f_density_grid_num, f_density_grid_num), dtype=np.int8)
    for xi in f_density_xindex:
        for yi in f_density_yindex:
            singlescore_grid[yi, xi] = onb.get_score_singleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            doublescore_grid[yi, xi] = onb.get_score_doubleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            triplescore_grid[yi, xi] = onb.get_score_tripleonly(f_density_xgrid[xi], f_density_ygrid[yi])
            bullscore_grid[yi, xi] = onb.get_score_bullonly(f_density_xgrid[xi], f_density_ygrid[yi])
    singlescore_coordinate_dic = {}
    doublescore_coordinate_dic = {}
    triplescore_coordinate_dic = {}
    bullscore_coordinate_dic = {}

    # coordinate for each score
    for si in range(20):
        singlescore_coordinate_dic[si] = np.where(singlescore_grid == onb.singlescorelist[si])
        doublescore_coordinate_dic[si] = np.where(doublescore_grid == onb.doublescorelist[si])
        triplescore_coordinate_dic[si] = np.where(triplescore_grid == onb.triplescorelist[si])
    bullscore_coordinate_dic[0] = np.where(bullscore_grid == onb.bullscorelist[0])
    bullscore_coordinate_dic[1] = np.where(bullscore_grid == onb.bullscorelist[1])

    for item in information_item:
        name_it = 'item{}'.format(item)
        result_filename = result_dir + '/' + '{}_gaussin_prob_grid.pkl'.format(name_it)
        print('\ncomputing {}'.format(result_filename))
        item_index = item - 1

        # new result grid
        prob_grid_singlescore = np.zeros((grid_num, grid_num, onb.singlescorelist_len))
        prob_grid_doublescore = np.zeros((grid_num, grid_num, onb.doublescorelist_len))
        prob_grid_triplescore = np.zeros((grid_num, grid_num, onb.triplescorelist_len))
        prob_grid_bullscore = np.zeros((grid_num, grid_num, onb.bullscorelist_len))

        # conduct a numerical integration
        # to evaluate the hitting probability for each score associated with the given aiming location
        time1 = time.time()
        for xi in xindex:  # (0,341) ----> (xi=170,yi=170) is the bullseye-center
            for yi in yindex:
                # select the proper Gaussian distribution according to the area to which the aiming location belongs
                mu = [xgrid[xi], ygrid[yi]]
                score, multiplier = onb.get_score_and_multiplier(mu)
                if score == 60 and multiplier == 3:  # triple 20
                    covariance_matrix = ris.T20_covariance_matrix_list[item_index]
                elif score == 57 and multiplier == 3:  # triple 19
                    covariance_matrix = ris.T19_covariance_matrix_list[item_index]
                elif score == 54 and multiplier == 3:  # triple 18
                    covariance_matrix = ris.T18_covariance_matrix_list[item_index]
                elif score == 51 and multiplier == 3:  # triple 17
                    covariance_matrix = ris.T17_covariance_matrix_list[item_index]
                elif score == 50 and multiplier == 2:  # double bull
                    covariance_matrix = ris.DB_covariance_matrix_list[item_index]
                else:
                    covariance_matrix = ris.double_covariance_matrix_list[item_index]
                # f_density_grid is the PDF of the fitted Gaussian distribution
                rv = multivariate_normal(mu, covariance_matrix)
                f_density_grid = rv.pdf(pos)

                # check score and integrate density
                for si in range(20):
                    print(si)
                    prob_grid_singlescore[xi, yi, si] = f_density_grid[
                                                            singlescore_coordinate_dic[si]].sum() * f_density_constant
                    prob_grid_doublescore[xi, yi, si] = f_density_grid[
                                                            doublescore_coordinate_dic[si]].sum() * f_density_constant
                    prob_grid_triplescore[xi, yi, si] = f_density_grid[
                                                            triplescore_coordinate_dic[si]].sum() * f_density_constant
                prob_grid_bullscore[xi, yi, 0] = f_density_grid[bullscore_coordinate_dic[0]].sum() * f_density_constant
                prob_grid_bullscore[xi, yi, 1] = f_density_grid[bullscore_coordinate_dic[1]].sum() * f_density_constant

        result_dic = {'prob_grid_singlescore': prob_grid_singlescore, 'prob_grid_doublescore': prob_grid_doublescore,
                      'prob_grid_triplescore': prob_grid_triplescore, 'prob_grid_bullscore': prob_grid_bullscore}
        time2 = time.time()

        print('computation is done in {} seconds'.format(time2 - time1))
        ft.dump_pickle(result_filename, result_dic, printflag=True)

    return

# evaluate_score_probability([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
