# On mathematical models of optimizing strategy decisions in professional darts
## Python implementation for darts master thesis
Markus Barth

This gitHub repository comprises some implementations in python to work with the game of darts/ with the given data set.

*draw_board_polar.py*:  Code that draws the board together with polar coordinate labels.

*read_in_data.py*:  Contains the first steps of reading in the data set/clearing the data/transforming the data in wanted shape/renaming colomns/removing unnecessary information from the data. In addition, we created sub-tables for each information item (i.e. for each player-year combination) --> See the folder **Player_specific_tables**.
Furthermore, it contains the function 'combi_file( )' that creates tuples (TR,z,n) that will be used as input for the upcoming EM-algorithm. The resulting combinations can be found in the repository under the name **combinations_all_players.xlsx**. Further, we deduce several in-game informations of the players (and their perfomance) and summarize them in the table **constructed_information_table.xlsx**.

The folder 'Skill_model_DP_9D_Markov' contains the necessary code for working with the models described in the thesis:

function_tools.py:  Code that we take over from Haugh and Wang. It provides some file operation functions like reading and saving.

dartboard.py: Code that we take over from Haugh and Wang. It provides the layout of the dart board and the 1mm grid of aiming locations. (We added the function 'score_matrix()'

read_in_combinations.py:  reads in all combinations for an information item from 'combinations_all_players.xlsx' to use it in:
em_algorithm.py:  we adjusted and extended the R-implementation from Tibshirani et al. to compute the estimated covariance matrix for each information and then store it in 'Estimated_variances.xlsx'.

read_in_sigma.py: read in the values from 'Estimated_variances.xlsx', create the covariance matrices and then use them in:

score_prob_player.py:   Code that we take over from Haugh and Wang which we adjusted at some points. It conducts a numerical integration to evaluate the hitting probability of each score (each score segment in the dartboard and the corresponding the numerical score) associated with each aiming location on the 1mm-grid.

evaluate_policy_and_aiming_grid.py: Part of the code that we take over from Haugh and Wang. It provides functions to generate the action set containing 984 aiming points as well as solving the state transition (turn to turn) probability associated with a specified aiming policy in the single player game.

maximize_expected_score.py: Functions that output the maximum expected score of a player as well as the corresponding point in the aiming grid. In addtion, it outputs a heatmap.

DP.py:  Code that we take over from Haugh and Wang. It solves the single player dart game via the dynamic programming formulation (approach 1).

Markov.py:  Creates the transition matrix for the closing process and solves the game of darts as a markovian decision process (approach 2).

9D.py:  Outputs a naive recommendation for a player at each specific point in the game (approach 3).
