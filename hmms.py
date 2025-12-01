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

#discretise temperature
temps_raw = data["temp"]

plt.figure(figsize=(8,4))
sns.histplot(temps_raw, bins=20, kde=True)
plt.title("Distribution of Temperature")
plt.xlabel("Temperature")
plt.ylabel("Count")
plt.show()
#maybe in order to do this you should check the distribution...see the spread
temps_d = []

##look at this graph and decide



# #discretise deaths
deaths_raw = data.columns["Deaths"]
death_quantiles = deaths_raw.quantile([0.33, 0.66]).values
deaths_d = []
for de in deaths_raw:
    if de <= death_quantiles[0]:
        deaths_d.append(0)
    elif de <= death_quantiles[1]:
        deaths_d.append(1)       
    else:
        deaths_d.append(2)

        

# deaths_possible_values = {"low", "medium", "high"} #again maybe split this based on
# #then convert all death values to their discrete equivalents





# # IMPORTANT You should learn the parameters of the HMM in two ways:
# # 1. Use both the sequence of (discretised) temperatures and the sequence of
# # (low, medium or high) deaths as data. So this is supervised learning which
# # is not the normal sort of learning for HMMs. hmmlearn does not appear
# # to implement supervised learning so you will have to do it ‘by hand’. Call
# # the HMM with parameters learned in this way: HMM1.

# #what are the main paremeters lol. But yes use deaths as y and temps as x?
# #split into trainign and testing set and maybe use some kind of training method? or is that ill advised
# def HMM1(temps, deaths):
    startprob 

    transmat 

    emission matrix


    





# # IMPORTANT 2. Use just the sequence of (low, medium or high) deaths as data. This is
# # normal HMM training and you can use hmmlearn to do it. Call the HMM
# # with parameters learned in this way: HMM2

# #probability of siwtching states
# def HMM2(switch_prob, noise_level, startprob):

#     #snippet from the lab as a possible "template"
#     n_components = 2
#     model = hmm.GaussianHMM(n_components=n_components, covariance_type="full")
#     model.startprob_ = startprob
#     model.transmat_ = np.array([[1. - switch_prob, switch_prob],
#                                 [switch_prob, 1. - switch_prob]])
#     model.means_ = np.array([[1.0], [-1.0]])
#     model.covars_ = np.ones((2, 1, 1)) * noise_level
#     return model
