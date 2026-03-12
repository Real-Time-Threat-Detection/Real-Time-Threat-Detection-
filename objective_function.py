import numpy as np
from Global_Vars import Global_Vars
from Model_ADeepCRF import Model_ADeepCRF
def Objfun(Soln):
    Feat = Global_Vars.Feat
    Target  = Global_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        learnperc = round(Feat.shape[0] * 0.75)
        Train_Data = Feat[:learnperc, :]
        Train_Target = Target[:learnperc, :]
        Test_Data = Feat[learnperc:, :]
        Test_Target = Target[learnperc:, :]
        Eval= Model_ADeepCRF(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Fitn[i] =(1 /Eval[4] ) + Eval[13] # Maximization of Accuracy and Minimization of FOR
    return Fitn

