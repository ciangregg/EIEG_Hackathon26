import pypsa
import pandas as pd
import numpy as np

n = pypsa.Network(
    "/home/seba/Documents/Colleg/Hackathon_2/"
    "EIEG_Hackathon26/cian/EIEG_Hackathon26/"
    "data/SV2024_northern_ireland.nc"
)

buses = n.buses.index

A = pd.DataFrame(
    0,
    index=buses,
    columns=buses,
    dtype=int
)

for _, line in n.lines.iterrows():
    bus0 = line["bus0"]
    bus1 = line["bus1"]

    A.loc[bus0, bus1] = 1
    A.loc[bus1, bus0] = 1

print(A)

A.to_csv("Adjacency_Matrix")