import os
import pandas as pd

# ========================================================================================================
# Read in the table with all covariance matrices and create lists for each of the 6 components:

directory_of_sigma = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/created_sub_tables'
os.chdir(directory_of_sigma)

file_name = 'estimated_variances.xlsx'
datatable = pd.read_excel(file_name)
# datatable.set_index('#', inplace=True)

T20_covariance_matrix_list = []
T19_covariance_matrix_list = []
T18_covariance_matrix_list = []
T17_covariance_matrix_list = []
DB_covariance_matrix_list = []
double_covariance_matrix_list = []

for i in datatable.index.tolist():
    T20_covariance_matrix_list.append([[datatable.loc[i, "T20_Sigma_x"], datatable.loc[i, "T20_Sigma_xy"]], [
        datatable.loc[i, "T20_Sigma_xy"], datatable.loc[i, "T20_Sigma_y"]]])
    T19_covariance_matrix_list.append([[datatable.loc[i, "T19_Sigma_x"], datatable.loc[i, "T19_Sigma_xy"]], [
        datatable.loc[i, "T19_Sigma_xy"], datatable.loc[i, "T19_Sigma_y"]]])
    T18_covariance_matrix_list.append([[datatable.loc[i, "T18_Sigma_x"], datatable.loc[i, "T18_Sigma_xy"]], [
        datatable.loc[i, "T18_Sigma_xy"], datatable.loc[i, "T18_Sigma_y"]]])
    T17_covariance_matrix_list.append([[datatable.loc[i, "T17_Sigma_x"], datatable.loc[i, "T17_Sigma_xy"]], [
        datatable.loc[i, "T17_Sigma_xy"], datatable.loc[i, "T17_Sigma_y"]]])
    DB_covariance_matrix_list.append([[datatable.loc[i, "DB_Sigma_x"], datatable.loc[i, "DB_Sigma_xy"]], [
        datatable.loc[i, "DB_Sigma_xy"], datatable.loc[i, "DB_Sigma_y"]]])
    double_covariance_matrix_list.append([[datatable.loc[i, "Double_Sigma_x"], datatable.loc[i, "Double_Sigma_xy"]], [
        datatable.loc[i, "Double_Sigma_xy"], datatable.loc[i, "Double_Sigma_y"]]])

# print(T20_covariance_matrix_list)
