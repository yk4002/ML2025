import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pymc
from hmmlearn import hmm


#get the data
data = pd.read_csv(pymc.get_data("deaths_and_temps_england_wales.csv"))


# Discretise temperature into 5 categories (based on clustering ranges)
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
    else:
        temps_d.append(4)           


# Discretise deaths into 3 categories: low, medium, high (using quantiles)
deaths_raw = data["deaths"]
death_quantiles = deaths_raw.quantile([0.33, 0.66]).values
deaths_d = []
for de in deaths_raw:
    if de <= death_quantiles[0]:
        deaths_d.append(0)  # low
    elif de <= death_quantiles[1]:
        deaths_d.append(1)  # medium
    else:
        deaths_d.append(2)  # high


# HMM1: Supervised learning of HMM parameters from known states and observations
def HMM1(deaths_arr, temps_arr, n_states, n_obs):

    ts = np.array(temps_arr)
    ds = np.array(deaths_arr)

    # Start probabilities: probability of starting in each hidden state (temperature)
    startprob = np.zeros(n_states)
    startprob[ts[0]] = 1.0  # only one sequence, so start is the first temp state

    # prob of moving from states
    transmat = np.zeros((n_states, n_states))

    # emitting observation k given state i
    emission_matrix = np.zeros((n_states, n_obs))

    # Count transitions and emissions along the sequence
    for i in range(len(ts) - 1):
        transmat[ts[i], ts[i+1]] += 1
        emission_matrix[ts[i], ds[i]] += 1

    # Count emission for the last observation
    emission_matrix[ts[-1], ds[-1]] += 1

    # Normalising to great prob
    transmat = np.divide(transmat, transmat.sum(axis=1, keepdims=True), where=transmat.sum(axis=1, keepdims=True)!=0)
    emission_matrix = np.divide(emission_matrix, emission_matrix.sum(axis=1, keepdims=True), where=emission_matrix.sum(axis=1, keepdims=True)!=0)

    return startprob, transmat, emission_matrix


# HMM2: HMM learned from only observations
#source: 
def HMM2(death_arr, n_states, iterations):
    deaths = np.array(death_arr).reshape(-1, 1)
    model = hmm.CategoricalHMM(n_components=n_states, n_iter=iterations, random_state=42)
    model.fit(deaths)
    return model


# Parameters
n_states = 4
n_obs = 3
iters = 500

# Learn supervised parameters from discretised seuqneces, then manually set learned params
startprob, transmat, emission_matrix = HMM1(deaths_d, temps_d, n_states, n_obs)
model1 = hmm.CategoricalHMM(n_components=n_states, init_params="")
model1.startprob_ = startprob
model1.transmat_ = transmat
model1.emissionprob_ = emission_matrix
# Sample from the supervised HMM (HMM1)
samples_hmm1, _ = model1.sample(len(deaths_d))


# Learn HMM2 from deaths
model2 = HMM2(deaths_d, n_states, iters)
# Sample from the unsupervised HMM (HMM2)
samples_hmm2, _ = model2.sample(len(deaths_d))




#plot other random stuff

# Plot actual deaths and samples from both HMMs for comparison
plt.figure(figsize=(15,5))


#use maybe 3 bar charts in one figure as a simple visualisation
plt.bar()

#then maybe some kind of matching matrix (ie which discrete values match the most? or the mean squared error is closest to 0?
#another graph type would be nice but discretised values are better

plt.plot(deaths_d, label="Actual Deaths (discretised)", alpha=0.7)
plt.plot(samples_hmm1.flatten(), label="HMM1 Sampled Deaths (supervised)", alpha=0.7)
plt.plot(samples_hmm2.flatten(), label="HMM2 Sampled Deaths (unsupervised)", alpha=0.7)

plt.legend()
plt.title("Comparison of Actual and HMM Sampled Death Sequences")
plt.xlabel("Time (months)")
plt.ylabel("Death category (0=low,1=medium,2=high)")
plt.show()


