import numpy as np
import math
import pandas as pd
import dartboard as onb
import read_in_combinations as ric
import time


store_dictionary = 'C:/Users/lolbr/Documents/Uni/#Master/#Masterarbeit/created_sub_tables'
# ===============================================================================================================
# defining distance from every wire to the board center
Rl = [onb.R1, onb.R2, onb.R3, onb.R4, onb.R5, onb.R, 180]
d = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

# index ordering of the scores:
ii = np.argsort(d)
# list of all double fields, excluding DB:
d_list = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16",
          "D17", "D18", "D19", "D20"]

# ================================================================================================================
# select a radius between r1 and r2 uniformly at random
def randomR(r1, r2):
    u = np.random.uniform(0, 1)
    return math.sqrt(r1 * r1 + (r2 * r2 - r1 * r1) * u)

# ----------------------------------------------------------------------------------------------------------------
# select a random point from SB or DB (uniformly at random):
def randomCirclePt(r1, r2):
    u = np.random.uniform(0, 1)
    theta = 2 * math.pi * u
    r = randomR(r1, r2)
    z = [r * math.cos(theta), r * math.sin(theta)]
    return z

# ------------------------------------------------------------------------------------------------------------------
# select a random point from a given segment 'x'
# single, double or treble segments are defined by using the corresponding wire distances r1 and r2:
def randomSlicePt(x, r1, r2, ii):
    u = np.random.uniform(0, 1)
    k = int(ii[int(x - 1)])
    theta = -math.pi / 20 + k * math.pi / 10 + math.pi / 10 * u
    theta = math.pi / 2 - theta
    r = randomR(r1, r2)
    z = [r * math.cos(theta), r * math.sin(theta)]
    return z

# print(randomSlicePt(11,Rl[3],Rl[4],ii))
# --------------------------------------------------------------------------------------------------------------
# select a random point outside of the board
# a miss can only land next to the aimed TR (e.g. D20) and its 2 neighbors (e.g. D1 and D5)
# and at most 10mm beyond the double wire (i.e. Rl[6] = 180)
# a further deviation is unrealistic for professional players
def randomMissPt(x, r1, r2, ii):
    u = np.random.uniform(-0.5, 1)
    k = int(ii[int(x - 1)])
    theta = -math.pi / 20 + k * math.pi / 10 + 2 * math.pi / 10 * u
    theta = math.pi / 2 - theta
    # print("th:",theta,"----",theta*180/math.pi)
    r = randomR(r1, r2)
    z = [r * math.cos(theta), r * math.sin(theta)]
    return z


# ==============================================================================================================
# select a random point in dependence of a certain field 'x'
# Target region 'TR' can be an additional input if we want to define the neighbors for a miss throw
# i.e. 'TR' is needed for the function randomMissPt
def randomPt2(x, Rl, ii, *TR):
    u = np.random.uniform(0, 1)
    if x != "SB25" and x != "DB25":
        st = x[0]
        nr = int(x.removeprefix(st))
        if st == "T":
            return randomSlicePt(nr, Rl[2], Rl[3], ii)
        elif st == "D":
            return randomSlicePt(nr, Rl[4], Rl[5], ii)
        elif st == "S":
            if TR[0] == "DB25":
                return randomSlicePt(nr, Rl[1], Rl[2], ii)
            elif TR[0].startswith("D") and TR[0] != "DB25":
                return randomSlicePt(nr, Rl[3], Rl[4], ii)
            else:
                if u <= 0.5:
                    return randomSlicePt(nr, Rl[1], Rl[2], ii)
                else:
                    return randomSlicePt(nr, Rl[3], Rl[4], ii)
        elif st == "M":
            return randomMissPt(int(TR[0].removeprefix(TR[0][0])), Rl[5], Rl[6], ii)
    elif x == "DB25":
        return randomCirclePt(0, Rl[0])
    elif x == "SB25":
        return randomCirclePt(Rl[0], Rl[1])

# print(randomPt2("M0",Rl,ii,("D2")))
# =============================================================================================================
# Transform the string 'TR' to a two-dimensional point 'mu=(mu_x,mu_y)' on the board:
# I.E. the mid-point of the polar coordinates of the region:
def TR_to_mu(TR):
    if TR == "T20":
        mu = [0, 103]
    elif TR == "T19":
        mu = [103 * math.cos(252 / 180 * math.pi), 103 * math.sin(252 / 180 * math.pi)]
    elif TR == "T18":
        mu = [103 * math.cos(54 / 180 * math.pi), 103 * math.sin(54 / 180 * math.pi)]
    elif TR == "T17":
        mu = [103 * math.cos(288 / 180 * math.pi), 103 * math.sin(288 / 180 * math.pi)]
    elif TR == "DB25":
        mu = [0, 0]
    elif TR == "D20":
        mu = [0, 166]
    elif TR == "D19":
        mu = [166 * math.cos(252 / 180 * math.pi), 166 * math.sin(252 / 180 * math.pi)]
    elif TR == "D18":
        mu = [166 * math.cos(54 / 180 * math.pi), 166 * math.sin(54 / 180 * math.pi)]
    elif TR == "D17":
        mu = [166 * math.cos(288 / 180 * math.pi), 166 * math.sin(288 / 180 * math.pi)]
    elif TR == "D16":
        mu = [166 * math.cos(216 / 180 * math.pi), 166 * math.sin(216 / 180 * math.pi)]
    elif TR == "D15":
        mu = [166 * math.cos(324 / 180 * math.pi), 166 * math.sin(324 / 180 * math.pi)]
    elif TR == "D14":
        mu = [166 * math.cos(162 / 180 * math.pi), 166 * math.sin(162 / 180 * math.pi)]
    elif TR == "D13":
        mu = [166 * math.cos(18 / 180 * math.pi), 166 * math.sin(18 / 180 * math.pi)]
    elif TR == "D12":
        mu = [166 * math.cos(126 / 180 * math.pi), 166 * math.sin(126 / 180 * math.pi)]
    elif TR == "D11":
        mu = [-166, 0]
    elif TR == "D10":
        mu = [166 * math.cos(342 / 180 * math.pi), 166 * math.sin(342 / 180 * math.pi)]
    elif TR == "D9":
        mu = [166 * math.cos(144 / 180 * math.pi), 166 * math.sin(144 / 180 * math.pi)]
    elif TR == "D8":
        mu = [166 * math.cos(198 / 180 * math.pi), 166 * math.sin(198 / 180 * math.pi)]
    elif TR == "D7":
        mu = [166 * math.cos(234 / 180 * math.pi), 166 * math.sin(234 / 180 * math.pi)]
    elif TR == "D6":
        mu = [166, 0]
    elif TR == "D5":
        mu = [166 * math.cos(108 / 180 * math.pi), 166 * math.sin(108 / 180 * math.pi)]
    elif TR == "D4":
        mu = [166 * math.cos(36 / 180 * math.pi), 166 * math.sin(36 / 180 * math.pi)]
    elif TR == "D3":
        mu = [0, -166]
    elif TR == "D2":
        mu = [166 * math.cos(306 / 180 * math.pi), 166 * math.sin(306 / 180 * math.pi)]
    elif TR == "D1":
        mu = [166 * math.cos(72 / 180 * math.pi), 166 * math.sin(72 / 180 * math.pi)]
    return mu


def simulateExp(TR, x, Sigma, Rl, ii):
    det = Sigma[0] * Sigma[1] - Sigma[2] ** 2
    w, W = 0, 0
    B = [0, 0, 0]
    # Importance Sampling with sample size: m = 5000
    for i in range(0, 5000):
        z = randomPt2(x, Rl, ii, TR)
        mu = TR_to_mu(TR)
        # delta = (mu-v)
        delta = [mu[0] - z[0], mu[1] - z[1]]
        w = math.exp(-(Sigma[1] * delta[0] ** 2 - 2 * Sigma[2] * delta[0] * delta[1] + Sigma[0] * delta[1] ** 2) /
                     (2 * det))
        B[0] += delta[0] ** 2 * w
        B[1] += delta[1] ** 2 * w
        B[2] += delta[0] * delta[1] * w
        W += w

    B[0] /= W
    B[1] /= W
    B[2] /= W

    return B

# ===============================================================================================================
def EMCov(TR, x_list, Sigma, Rl, ii):
    C = [0, 0, 0]
    # n = n_1+...+n_p (For example, p = 6 for any treble component/ n belongs to the same TR).
    if type(TR) == list:
        m = 0
        N = len(TR)
        for l in range(N):
            x_list_l = x_list[l]
            n = len(x_list_l)
            for i in range(0, n):
                print("substep:", i, "TR[l]:",TR[l],"x_list_l[i]:",x_list_l[i])
                B = simulateExp(TR[l], x_list_l[i], Sigma, Rl, ii)
                # print("B=", B)
                C[0] += B[0]
                C[1] += B[1]
                C[2] += B[2]
                # print("C=", C)
            m += n

        A = [C[0] / m, C[1] / m, C[2] / m]

    else:
        n = len(x_list)
        # print("len(x_list):", n)
        for i in range(0, n):
            print("substep:", i)
            B = simulateExp(TR, x_list[i], Sigma, Rl, ii)
            # print("B=", B)
            C[0] += B[0]
            C[1] += B[1]
            C[2] += B[2]
            # print("C=", C)

        A = [C[0] / n, C[1] / n, C[2] / n]

    return A

# print(EMCov(["D20","D3"],[["S20","S20","M0","M0","D20"],["S3","S3","S19","S3","S17","M0","M0","M0","M0"]],[100,100,10],Rl,ii))
# print(EMCov("T20",["S20","S20","T20","S20","T20","T5","T5"],[100,100,10],Rl,ii))
# ===============================================================================================================
# function that gives 'x_list' (i.e. a list of all z scores hit when targeting 'TR') for a given TR (for each table):
# For example, for TR=T20 only 6 possible z scores are possible but 'x_list' contains them
# as often as they were hit. Similarly, if a z score was never hit even if it is a probable score for the target region,
# it will not be inside 'x_list'.
def score_list(TR, combi_table):
    x_list = []
    if type(TR) == list:
        for i in range(len(TR)):
            x_list.append([])
            for j in range(0, len(combi_table.index.tolist())):
                if combi_table["TR"].values[j] == TR[i]:
                    x_list[i] += [(combi_table["z"].values[j]) for w in range(combi_table["n"].values[j])]
    else:
        for j in range(0, len(combi_table.index.tolist())):
            if combi_table["TR"].values[j] == TR:
                x_list += [(combi_table["z"].values[j]) for w in range(combi_table["n"].values[j])]

    return x_list

# print(score_list(d_list,ric.DVdB19_combitable))
# =============================================================================================================
# We first used a fix number of iteration 'niter' = 50 to get first good results in appropriate time
# Later we substituted the function EM() by the final function EM_while() that iterates as long as the
# convergence criterion is not fulfilled
def EM(TR, x_list, niter, Rl, ii):
    S1 = [0 for k in range(niter)]
    S2 = [0 for k in range(niter)]
    S3 = [0 for k in range(niter)]
    Scur = [100, 100, 10]  # = initial Sigma
    for i in range(0, niter):
        # print("iteration:", i)
        # print("TR:",TR)
        A = EMCov(TR, x_list, Scur, Rl, ii)
        S1[i] = A[0]
        S2[i] = A[1]
        S3[i] = A[2]
        Scur[0] = S1[i]
        Scur[1] = S2[i]
        Scur[2] = S3[i]

    return Scur, S1,S2,S3
# ==================================================================================================================
# STOPPING CRITERIA: We decided to stop the iterations when sigma_x, sigma_y and sigma_x,y do not change more than
# the value of 1 at the same time.
# This should lead to very good results and is sometimes faster than the previously fixed number of 50 iterations
def EM_while(TR, x_list, Rl, ii):
    S_current = [0, 0, 0]
    Scur = [100, 100, 10]
    time1 = time.time()
    iteration = 0
    while abs(Scur[0] - S_current[0]) > 1 or abs(Scur[1] - S_current[1]) > 1 or abs(Scur[2] - S_current[2]) > 1:
        iteration += 1
        print("iteration:", iteration)
        A = EMCov(TR, x_list, Scur, Rl, ii)
        S_current = Scur
        Scur = A
    time2 = time.time()
    print("time:", time2-time1, "iterations_total:", iteration)
    return Scur

# =====================================================================================================================
# NOW: Compute all 66 covariance matrices (for each information item and each component) and store them in an excel-file:
DVdB19_T20_score_list = score_list("T20", ric.DVdB19_combitable)
DVdB19_T20_Cov = EM_while("T20", DVdB19_T20_score_list, Rl, ii)
DVdB19_T19_score_list = score_list("T19", ric.DVdB19_combitable)
DVdB19_T19_Cov = EM_while("T19", DVdB19_T19_score_list, Rl, ii)
DVdB19_T18_score_list = score_list("T18", ric.DVdB19_combitable)
DVdB19_T18_Cov = EM_while("T18",DVdB19_T18_score_list,Rl,ii)
DVdB19_T17_score_list = score_list("T17", ric.DVdB19_combitable)
DVdB19_T17_Cov = EM_while("T17", DVdB19_T17_score_list, Rl, ii)
DVdB19_DB_score_list = score_list("DB25", ric.DVdB19_combitable)
DVdB19_DB_Cov = EM_while("DB25", DVdB19_DB_score_list, Rl, ii)
DVdB19_d_score_list = score_list(d_list, ric.DVdB19_combitable)
DVdB19_d_Cov = EM_while(d_list, DVdB19_d_score_list, Rl, ii)

DVdB20_T20_score_list = score_list("T20", ric.DVdB20_combitable)
DVdB20_T20_Cov = EM_while("T20", DVdB20_T20_score_list, Rl, ii)
DVdB20_T19_score_list = score_list("T19", ric.DVdB20_combitable)
DVdB20_T19_Cov = EM_while("T19", DVdB20_T19_score_list, Rl, ii)
DVdB20_T18_score_list = score_list("T18", ric.DVdB20_combitable)
DVdB20_T18_Cov = EM_while("T18", DVdB20_T18_score_list, Rl, ii)
DVdB20_T17_score_list = score_list("T17", ric.DVdB20_combitable)
DVdB20_T17_Cov = EM_while("T17", DVdB20_T17_score_list, Rl, ii)
DVdB20_DB_score_list = score_list("DB25", ric.DVdB20_combitable)
DVdB20_DB_Cov = EM_while("DB25", DVdB20_DB_score_list, Rl, ii)
DVdB20_d_score_list = score_list(d_list, ric.DVdB20_combitable)
DVdB20_d_Cov = EM_while(d_list, DVdB20_d_score_list, Rl, ii)

DVdB21_T20_score_list = score_list("T20", ric.DVdB21_combitable)
DVdB21_T20_Cov = EM_while("T20", DVdB21_T20_score_list, Rl, ii)
DVdB21_T19_score_list = score_list("T19", ric.DVdB21_combitable)
DVdB21_T19_Cov = EM_while("T19", DVdB21_T19_score_list, Rl, ii)
DVdB21_T18_score_list = score_list("T18", ric.DVdB21_combitable)
DVdB21_T18_Cov = EM_while("T18", DVdB21_T18_score_list, Rl, ii)
DVdB21_T17_score_list = score_list("T17", ric.DVdB21_combitable)
DVdB21_T17_Cov = EM_while("T17", DVdB21_T17_score_list, Rl, ii)
DVdB21_DB_score_list = score_list("DB25", ric.DVdB21_combitable)
DVdB21_DB_Cov = EM_while("DB25", DVdB21_DB_score_list, Rl, ii)
DVdB21_d_score_list = score_list(d_list, ric.DVdB21_combitable)
DVdB21_d_Cov = EM_while(d_list, DVdB21_d_score_list, Rl, ii)

AL19_T20_score_list = score_list("T20", ric.AL19_combitable)
AL19_T20_Cov = EM_while("T20", AL19_T20_score_list, Rl, ii)
AL19_T19_score_list = score_list("T19", ric.AL19_combitable)
AL19_T19_Cov = EM_while("T19", AL19_T19_score_list, Rl, ii)
AL19_T18_score_list = score_list("T18", ric.AL19_combitable)
AL19_T18_Cov = EM_while("T18", AL19_T18_score_list, Rl, ii)
AL19_T17_score_list = score_list("T17", ric.AL19_combitable)
AL19_T17_Cov = EM_while("T17", AL19_T17_score_list, Rl, ii)
AL19_DB_score_list = score_list("DB25", ric.AL19_combitable)
AL19_DB_Cov = EM_while("DB25", AL19_DB_score_list, Rl, ii)
AL19_d_score_list = score_list(d_list, ric.AL19_combitable)
AL19_d_Cov = EM_while(d_list, AL19_d_score_list, Rl, ii)

AL20_T20_score_list = score_list("T20", ric.AL20_combitable)
AL20_T20_Cov = EM_while("T20", AL20_T20_score_list, Rl, ii)
AL20_T19_score_list = score_list("T19", ric.AL20_combitable)
AL20_T19_Cov = EM_while("T19", AL20_T19_score_list, Rl, ii)
AL20_T18_score_list = score_list("T18", ric.AL20_combitable)
AL20_T18_Cov = EM_while("T18", AL20_T18_score_list, Rl, ii)
AL20_T17_score_list = score_list("T17", ric.AL20_combitable)
AL20_T17_Cov = EM_while("T17", AL20_T17_score_list, Rl, ii)
AL20_DB_score_list = score_list("DB25", ric.AL20_combitable)
AL20_DB_Cov = EM_while("DB25", AL20_DB_score_list, Rl, ii)
AL20_d_score_list = score_list(d_list, ric.AL20_combitable)
AL20_d_Cov = EM_while(d_list, AL20_d_score_list, Rl, ii)

JdS20_T20_score_list = score_list("T20",ric.JdS20_combitable)
JdS20_T20_Cov = EM_while("T20",JdS20_T20_score_list, Rl, ii)
JdS20_T19_score_list = score_list("T19", ric.JdS20_combitable)
JdS20_T19_Cov = EM_while("T19", JdS20_T19_score_list, Rl, ii)
JdS20_T18_score_list = score_list("T18", ric.JdS20_combitable)
JdS20_T18_Cov = EM_while("T18", JdS20_T18_score_list, Rl, ii)
JdS20_T17_score_list = score_list("T17", ric.JdS20_combitable)
JdS20_T17_Cov = EM_while("T17", JdS20_T17_score_list, Rl, ii)
JdS20_DB_score_list = score_list("DB25", ric.JdS20_combitable)
JdS20_DB_Cov = EM_while("DB25", JdS20_DB_score_list, Rl, ii)
JdS20_d_score_list = score_list(d_list, ric.JdS20_combitable)
JdS20_d_Cov = EM_while(d_list, JdS20_d_score_list, Rl, ii)

JdS21_T20_score_list = score_list("T20", ric.JdS21_combitable)
JdS21_T20_Cov = EM_while("T20", JdS21_T20_score_list, Rl, ii)
JdS21_T19_score_list = score_list("T19", ric.JdS21_combitable)
JdS21_T19_Cov = EM_while("T19", JdS21_T19_score_list, Rl, ii)
JdS21_T18_score_list = score_list("T18", ric.JdS21_combitable)
JdS21_T18_Cov = EM_while("T18", JdS21_T18_score_list, Rl, ii)
JdS21_T17_score_list = score_list("T17", ric.JdS21_combitable)
JdS21_T17_Cov = EM_while("T17", JdS21_T17_score_list, Rl, ii)
JdS21_DB_score_list = score_list("DB25", ric.JdS21_combitable)
JdS21_DB_Cov = EM_while("DB25", JdS21_DB_score_list, Rl, ii)
JdS21_d_score_list = score_list(d_list, ric.JdS21_combitable)
JdS21_d_Cov = EM_while(d_list, JdS21_d_score_list, Rl, ii)

DvD20_T20_score_list = score_list("T20", ric.DvD20_combitable)
DvD20_T20_Cov = EM_while("T20", DvD20_T20_score_list, Rl, ii)
DvD20_T19_score_list = score_list("T19", ric.DvD20_combitable)
DvD20_T19_Cov = EM_while("T19", DvD20_T19_score_list, Rl, ii)
DvD20_T18_score_list = score_list("T18", ric.DvD20_combitable)
DvD20_T18_Cov = EM_while("T18", DvD20_T18_score_list, Rl, ii)
DvD20_T17_score_list = score_list("T17", ric.DvD20_combitable)
DvD20_T17_Cov = EM_while("T17", DvD20_T17_score_list, Rl, ii)
DvD20_DB_score_list = score_list("DB25", ric.DvD20_combitable)
DvD20_DB_Cov = EM_while("DB25", DvD20_DB_score_list, Rl, ii)
DvD20_d_score_list = score_list(d_list, ric.DvD20_combitable)
DvD20_d_Cov = EM_while(d_list, DvD20_d_score_list, Rl, ii)

DvD21_T20_score_list = score_list("T20", ric.DvD21_combitable)
DvD21_T20_Cov = EM_while("T20", DvD21_T20_score_list, Rl, ii)
DvD21_T19_score_list = score_list("T19", ric.DvD21_combitable)
DvD21_T19_Cov = EM_while("T19", DvD21_T19_score_list, Rl, ii)
DvD21_T18_score_list = score_list("T18", ric.DvD21_combitable)
DvD21_T18_Cov = EM_while("T18", DvD21_T18_score_list, Rl, ii)
DvD21_T17_score_list = score_list("T17", ric.DvD21_combitable)
DvD21_T17_Cov = EM_while("T17", DvD21_T17_score_list, Rl, ii)
DvD21_DB_score_list = score_list("DB25", ric.DvD21_combitable)
DvD21_DB_Cov = EM_while("DB25", DvD21_DB_score_list, Rl, ii)
DvD21_d_score_list = score_list(d_list, ric.DvD21_combitable)
DvD21_d_Cov = EM_while(d_list, DvD21_d_score_list, Rl, ii)

NA18_T20_score_list = score_list("T20", ric.NA18_combitable)
NA18_T20_Cov = EM_while("T20", NA18_T20_score_list, Rl, ii)
NA18_T19_score_list = score_list("T19", ric.NA18_combitable)
NA18_T19_Cov = EM_while("T19", NA18_T19_score_list, Rl, ii)
NA18_T18_score_list = score_list("T18", ric.NA18_combitable)
NA18_T18_Cov = EM_while("T18", NA18_T18_score_list, Rl, ii)
NA18_T17_score_list = score_list("T17", ric.NA18_combitable)
NA18_T17_Cov = EM_while("T17", NA18_T17_score_list, Rl, ii)
NA18_DB_score_list = score_list("DB25", ric.NA18_combitable)
NA18_DB_Cov = EM_while("DB25", NA18_DB_score_list, Rl, ii)
NA18_d_score_list = score_list(d_list, ric.NA18_combitable)
NA18_d_Cov = EM_while(d_list, NA18_d_score_list, Rl, ii)

NA19_T20_score_list = score_list("T20", ric.NA19_combitable)
NA19_T20_Cov = EM_while("T20", NA19_T20_score_list, Rl, ii)
NA19_T19_score_list = score_list("T19", ric.NA19_combitable)
NA19_T19_Cov = EM_while("T19", NA19_T19_score_list, Rl, ii)
NA19_T18_score_list = score_list("T18", ric.NA19_combitable)
NA19_T18_Cov = EM_while("T18", NA19_T18_score_list, Rl, ii)
NA19_T17_score_list = score_list("T17", ric.NA19_combitable)
NA19_T17_Cov = EM_while("T17", NA19_T17_score_list, Rl, ii)
NA19_DB_score_list = score_list("DB25", ric.NA19_combitable)
NA19_DB_Cov = EM_while("DB25", NA19_DB_score_list, Rl, ii)
NA19_d_score_list = score_list(d_list, ric.NA19_combitable)
NA19_d_Cov = EM_while(d_list, NA19_d_score_list, Rl, ii)

estimated_covariances = pd.DataFrame({"Player": ["van den Bergh", "van den Bergh", "van den Bergh",
                                                 "Lewis", "Lewis",
                                                 "de Sousa", "de Sousa",
                                                 "van Duijvenbode", "van Duijvenbode",
                                                 "Aspinall", "Aspinall"],
                                      "Years": [ "2019", "2020", "2021",
                                                 "2019", "2020",
                                                 "2020", "2021",
                                                 "2020", "2021",
                                                 "2018", "2019"],
                                      "T20_Sigma_x": [DVdB19_T20_Cov[0], DVdB20_T20_Cov[0], DVdB21_T20_Cov[0],
                                                      AL19_T20_Cov[0], AL20_T20_Cov[0],
                                                      JdS20_T20_Cov[0], JdS21_T20_Cov[0],
                                                      DvD20_T20_Cov[0], DvD21_T20_Cov[0],
                                                      NA18_T20_Cov[0], NA19_T20_Cov[0]],
                                      "T20_Sigma_y": [DVdB19_T20_Cov[1], DVdB20_T20_Cov[1], DVdB21_T20_Cov[1],
                                                      AL19_T20_Cov[1], AL20_T20_Cov[1],
                                                      JdS20_T20_Cov[1], JdS21_T20_Cov[1],
                                                      DvD20_T20_Cov[1], DvD21_T20_Cov[1],
                                                      NA18_T20_Cov[1], NA19_T20_Cov[1]],
                                      "T20_Sigma_xy": [DVdB19_T20_Cov[2], DVdB20_T20_Cov[2], DVdB21_T20_Cov[2],
                                                      AL19_T20_Cov[2], AL20_T20_Cov[2],
                                                      JdS20_T20_Cov[2], JdS21_T20_Cov[2],
                                                      DvD20_T20_Cov[2], DvD21_T20_Cov[2],
                                                      NA18_T20_Cov[2], NA19_T20_Cov[2]],
                                      "T19_Sigma_x": [DVdB19_T19_Cov[0], DVdB20_T19_Cov[0], DVdB21_T19_Cov[0],
                                                      AL19_T19_Cov[0], AL20_T19_Cov[0],
                                                      JdS20_T19_Cov[0], JdS21_T19_Cov[0],
                                                      DvD20_T19_Cov[0], DvD21_T19_Cov[0],
                                                      NA18_T19_Cov[0], NA19_T19_Cov[0]],
                                      "T19_Sigma_y": [DVdB19_T19_Cov[1], DVdB20_T19_Cov[1], DVdB21_T19_Cov[1],
                                                      AL19_T19_Cov[1], AL20_T19_Cov[1],
                                                      JdS20_T19_Cov[1], JdS21_T19_Cov[1],
                                                      DvD20_T19_Cov[1], DvD21_T19_Cov[1],
                                                      NA18_T19_Cov[1], NA19_T19_Cov[1]],
                                      "T19_Sigma_xy": [DVdB19_T19_Cov[2], DVdB20_T19_Cov[2], DVdB21_T19_Cov[2],
                                                      AL19_T19_Cov[2], AL20_T19_Cov[2],
                                                      JdS20_T19_Cov[2], JdS21_T19_Cov[2],
                                                      DvD20_T19_Cov[2], DvD21_T19_Cov[2],
                                                      NA18_T19_Cov[2], NA19_T19_Cov[2]],
                                      "T18_Sigma_x": [DVdB19_T18_Cov[0], DVdB20_T18_Cov[0], DVdB21_T18_Cov[0],
                                                      AL19_T18_Cov[0], AL20_T18_Cov[0],
                                                      JdS20_T18_Cov[0], JdS21_T18_Cov[0],
                                                      DvD20_T18_Cov[0], DvD21_T18_Cov[0],
                                                      NA18_T18_Cov[0], NA19_T18_Cov[0]],
                                      "T18_Sigma_y": [DVdB19_T18_Cov[1], DVdB20_T18_Cov[1], DVdB21_T18_Cov[1],
                                                      AL19_T18_Cov[1], AL20_T18_Cov[1],
                                                      JdS20_T18_Cov[1], JdS21_T18_Cov[1],
                                                      DvD20_T18_Cov[1], DvD21_T18_Cov[1],
                                                      NA18_T18_Cov[1], NA19_T18_Cov[1]],
                                      "T18_Sigma_xy": [DVdB19_T18_Cov[2], DVdB20_T18_Cov[2], DVdB21_T18_Cov[2],
                                                      AL19_T18_Cov[2], AL20_T18_Cov[2],
                                                      JdS20_T18_Cov[2], JdS21_T18_Cov[2],
                                                      DvD20_T18_Cov[2], DvD21_T18_Cov[2],
                                                      NA18_T18_Cov[2], NA19_T18_Cov[2]],
                                      "T17_Sigma_x": [DVdB19_T17_Cov[0], DVdB20_T17_Cov[0], DVdB21_T17_Cov[0],
                                                      AL19_T17_Cov[0], AL20_T17_Cov[0],
                                                      JdS20_T17_Cov[0], JdS21_T17_Cov[0],
                                                      DvD20_T17_Cov[0], DvD21_T17_Cov[0],
                                                      NA18_T17_Cov[0], NA19_T17_Cov[0]],
                                      "T17_Sigma_y": [DVdB19_T17_Cov[1], DVdB20_T17_Cov[1], DVdB21_T17_Cov[1],
                                                      AL19_T17_Cov[1], AL20_T17_Cov[1],
                                                      JdS20_T17_Cov[1], JdS21_T17_Cov[1],
                                                      DvD20_T17_Cov[1], DvD21_T17_Cov[1],
                                                      NA18_T17_Cov[1], NA19_T17_Cov[1]],
                                      "T17_Sigma_xy": [DVdB19_T17_Cov[2], DVdB20_T17_Cov[2], DVdB21_T17_Cov[2],
                                                      AL19_T17_Cov[2], AL20_T17_Cov[2],
                                                      JdS20_T17_Cov[2], JdS21_T17_Cov[2],
                                                      DvD20_T17_Cov[2], DvD21_T17_Cov[2],
                                                      NA18_T17_Cov[2], NA19_T17_Cov[2]],
                                      "Double_Sigma_x": [DVdB19_d_Cov[0], DVdB20_d_Cov[0], DVdB21_d_Cov[0],
                                                      AL19_d_Cov[0], AL20_d_Cov[0],
                                                      JdS20_d_Cov[0], JdS21_d_Cov[0],
                                                      DvD20_d_Cov[0], DvD21_d_Cov[0],
                                                      NA18_d_Cov[0], NA19_d_Cov[0]],
                                      "Double_Sigma_y": [DVdB19_d_Cov[1], DVdB20_d_Cov[1], DVdB21_d_Cov[1],
                                                      AL19_d_Cov[1], AL20_d_Cov[1],
                                                      JdS20_d_Cov[1], JdS21_d_Cov[1],
                                                      DvD20_d_Cov[1], DvD21_d_Cov[1],
                                                      NA18_d_Cov[1], NA19_d_Cov[1]],
                                      "Double_Sigma_xy": [DVdB19_d_Cov[2], DVdB20_d_Cov[2], DVdB21_d_Cov[2],
                                                      AL19_d_Cov[2], AL20_d_Cov[2],
                                                      JdS20_d_Cov[2], JdS21_d_Cov[2],
                                                      DvD20_d_Cov[2], DvD21_d_Cov[2],
                                                      NA18_d_Cov[2], NA19_d_Cov[2]],
                                      "DB_Sigma_x": [DVdB19_DB_Cov[0], DVdB20_DB_Cov[0], DVdB21_DB_Cov[0],
                                                      AL19_DB_Cov[0], AL20_DB_Cov[0],
                                                      JdS20_DB_Cov[0], JdS21_DB_Cov[0],
                                                      DvD20_DB_Cov[0], DvD21_DB_Cov[0],
                                                      NA18_DB_Cov[0], NA19_DB_Cov[0]],
                                      "DB_Sigma_y": [DVdB19_DB_Cov[1], DVdB20_DB_Cov[1], DVdB21_DB_Cov[1],
                                                      AL19_DB_Cov[1], AL20_DB_Cov[1],
                                                      JdS20_DB_Cov[1], JdS21_DB_Cov[1],
                                                      DvD20_DB_Cov[1], DvD21_DB_Cov[1],
                                                      NA18_DB_Cov[1], NA19_DB_Cov[1]],
                                      "DB_Sigma_xy": [DVdB19_DB_Cov[2], DVdB20_DB_Cov[2], DVdB21_DB_Cov[2],
                                                      AL19_DB_Cov[2], AL20_DB_Cov[2],
                                                      JdS20_DB_Cov[2], JdS21_DB_Cov[2],
                                                      DvD20_DB_Cov[2], DvD21_DB_Cov[2],
                                                      NA18_DB_Cov[2], NA19_DB_Cov[2]]})

# STORE COMMAND:
#estimated_covariances.to_excel(store_dictionary + '/estimated_variances.xlsx')