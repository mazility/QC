from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from markowitz import Markowitz

file_prices = "data/prices.csv"
file_averages = "data/averages.csv"
file_covariance = "data/covariance.csv"
token_str = "" # Set D-Wave Access Token

ising_model = Markowitz(file_prices, file_averages, file_covariance, theta = [0.3, 0.3, 0.3], budget = 100.0)
c = {}
N = ising_model.n
for i in range (N):
    for j in range (i + 1, N, 1):
        c[i, j] = ising_model.G[i][j]
h = {}

for i in range (N):
    h[i] = ising_model.h[i][0]

sampler = EmbeddingComposite(DWaveSampler(token = token_str))
response = sampler.sample_ising(h, c)

for sample, energy in response.data(['sample', 'energy']):
    print("Samples: ", sample, "\nEnergies: ", energy)
