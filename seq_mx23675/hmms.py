import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pymc
from hmmlearn import hmm


#get the data
data = pd.read_csv(pymc.get_data("deaths_and_temps_england_wales.csv"))


#IMPORTANT Create an HMM for this data where the state of the HMM for each month is
# the temperature and the observation is the reported number of deaths. You will
# need discretise the temperature variable; how you do this is up to you. Also
# in order to make the task manageable you are to replace the original deaths
# variable with a discrete variable with only 3 values: low, medium and high,
# where it is up to you how to do this.




#USE QUANTILES
#SOURCE:

#discretise temperature (values chosen based on clustering of temperature)
temps_raw = data["temp"]
temps_d = []
for t in temps_raw:
    if t <= 3:
        temps_d.append(0)
    elif t <=7:
        temps_d.append(1)
    elif t <=12:
        temps_d.append(2)  
    elif t <=16:
        temps_d.append(3)   
    elif t > 16:
        temps_d.append(4)           



# #discretise deaths -where 0,1,2 corresponds to low, medium, high
deaths_raw = data.columns["Deaths"]
death_quantiles = deaths_raw.quantile([0.33, 0.66]).values
deaths_d = []
for de in deaths_raw:
    #low
    if de <= death_quantiles[0]:
        deaths_d.append(0)
    #medium
    elif de <= death_quantiles[1]:
        deaths_d.append(1)   
    #high    
    else:
        deaths_d.append(2)





# # IMPORTANT You should learn the parameters of the HMM in two ways:
# # 1. Use both the sequence of (discretised) temperatures and the sequence of
# # (low, medium or high) deaths as data. So this is supervised learning which
# # is not the normal sort of learning for HMMs. hmmlearn does not appear
# # to implement supervised learning so you will have to do it ‘by hand’. Call
# # the HMM with parameters learned in this way: HMM1.

# use both the hidden states(temperature) as well as the 
#its basically supervised
# def HMM1(temps, deaths):
    #hided

    # startprob = 

    # transmat 

    # emission matrix


    





# # IMPORTANT 2. Use just the sequence of (low, medium or high) deaths as data. This is
# # normal HMM training and you can use hmmlearn to do it. Call the HMM
# # with parameters learned in this way: HMM2

# #probability of siwtching states
# def HMM2(switch_prob, noise_level, startprob):

#ie just use the observations (deaths) for this hmm

#     #snippet from the lab as a possible "template"
#     n_components = 2
#     model = hmm.GaussianHMM(n_components=n_components, covariance_type="full")
#     model.startprob_ = startprob
#     model.transmat_ = np.array([[1. - switch_prob, switch_prob],
#                                 [switch_prob, 1. - switch_prob]])
#     model.means_ = np.array([[1.0], [-1.0]])
#     model.covars_ = np.ones((2, 1, 1)) * noise_level
#     return model
