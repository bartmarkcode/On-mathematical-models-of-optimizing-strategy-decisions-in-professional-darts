import os
import pandas as pd

# =======================================================================================================
# Read in the table which contains all (TR,z,n) combinations and create variables for each information item:
# They are then used for the EM-algorithm

directory_of_combinations = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/created_sub_tables'
os.chdir(directory_of_combinations)

file_name = 'combinations_all_players.xlsx'

DVdB19_combitable = pd.read_excel(file_name, sheet_name="DVdB19")
DVdB20_combitable = pd.read_excel(file_name, sheet_name="DVdB20")
DVdB21_combitable = pd.read_excel(file_name, sheet_name="DVdB21")
AL19_combitable = pd.read_excel(file_name, sheet_name="AL19")
AL20_combitable = pd.read_excel(file_name, sheet_name="AL20")
JdS20_combitable = pd.read_excel(file_name, sheet_name="JdS20")
JdS21_combitable = pd.read_excel(file_name, sheet_name="JdS21")
DvD20_combitable = pd.read_excel(file_name, sheet_name="DvD20")
DvD21_combitable = pd.read_excel(file_name, sheet_name="DvD21")
NA18_combitable = pd.read_excel(file_name, sheet_name="NA18")
NA19_combitable = pd.read_excel(file_name, sheet_name="NA19")
