# On mathematical models of optimizing strategy decisions in professional darts
## Python implementation for darts master thesis
Markus Barth

This gitHub repository comprises some implementations in python to work with the game of darts/ with the given data set.

*draw_board_polar.py*:  Code that draws the board together with polar coordinate labels.

*read_in_data.py*:  Contains the first steps of reading in the data set/clearing the data/transforming the data in wanted shape/renaming colomns/removing unnecessary information from the data. In addition, we created sub-tables for each information item (i.e. for each player-year combination) --> See the folder **Player_specific_tables**.
Furthermore, it contains the function 'combi_file( )' that creates tuples (TR,z,n) that will be used as input for the upcoming EM-algorithm. The resulting combinations can be found in the repository under the name **combinations_all_players.xlsx**. Further, we deduce several in-game informations of the players (and their perfomance) and summarize them in the table **constructed_information_table.xlsx**.


The folder **Skill_model_DP_9D_Markov** contains the necessary code for working with the models described in the thesis:

*function_tools.py*:  Code that we take over from Haugh and Wang. It provides some file operation functions like reading and saving.

*dartboard.py*: Code that we take over from Haugh and Wang. It provides the layout of the dart board and the 1mm grid of aiming locations. (We added the function 'score_matrix( )'

*read_in_combinations.py*: Reads in all combinations for an information item from **combinations_all_players.xlsx** to use it in:

*em_algorithm.py*:  We adjusted and extended the R-implementation from Tibshirani et al. to compute the estimated covariance matrix for each information and then store it in **estimated_variances.xlsx**. 

*read_in_sigma.py*: read in the values from **estimated_variances.xlsx**, create the covariance matrices and then use them in:

*score_prob_player.py*: Code that we take over from Haugh and Wang and which we adjusted at some points. It conducts a numerical integration to evaluate the hitting probability of each score (each score segment in the dartboard and the corresponding the numerical score) associated with each aiming location on the 1mm-grid.

*evaluate_policy_and_aiming_grid.py*: Part of the code that we take over from Haugh and Wang. It provides functions to generate the action set containing 984 aiming points as well as solving the state transition (turn to turn) probability associated with a specified aiming policy in the single player game.

*maximize_expected_score.py*: Outputs the expected score over the whole 1mm grid and contains functions that output the maximum expected score of a player as well as the corresponding point in the aiming grid. In addtion, it outputs a heatmap for a given player.

*draw_action_set.py*: Visualize the constructed adapted action set in a dartboard. (Additionally, we provide another function visualize recommended tracks in a dartboard).

*DP.py*:  Code that we take over from Haugh and Wang. It solves the single player dart game via the dynamic programming formulation. We added the function 'solve_noturn( )' to receive the output of the DP version without turn feature in our desired form. For this reason, we also adjusted the function 'solve_singlegame( )'. As a result, we created three recommendation tables: **DP_noturn_recommendations.xlsx**, **DP_turn_feature_recommendations.xlsx** and **DP_turn_feature_extended_recommendations.xlsx**. (approach 1)

*markov_approach.py*: Create the set of all possible checkouts for a given score as well as the first target score of such checkouts, create the transition using the minimum expected time to absorption property (transform them in transition maps) and compute the final optimal Markov strategy for a given total score such that we get an output table **Markov_recommendations.xlsx**. (approach 2).

*nined.py*:  Creating player-specific skill sets K using the adapted action set (also the alternative version using the 1mm grid is provided), creating skill maps and produce 9D-strategies to receive an output table **9D_recommendations.xlsx**. (approach 3).

*comparison.py*: All functions that were necessary to provide us with the findings which are stated in the Results-section of the thesis. (In addition, we provide the function 'plot_track( )' such that we can visualize recommended tracks and compare them on the fly)
