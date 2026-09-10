import pypsa
import pandas as pd


n = pypsa.Network("/home/seba/Documents/Colleg/Hackathon_2/EIEG_Hackathon26/cian/EIEG_Hackathon26/data/SV2024_northern_ireland.nc")

print(n)
print(n.buses)
print(n.lines)
print(n.generators)
print(n.loads)
n_opt = n.optimize()

print(n_opt)

import pypsa

n = pypsa.Network(
    "/home/seba/Documents/Colleg/Hackathon_2/"
    "EIEG_Hackathon26/cian/EIEG_Hackathon26/"
    "data/SV2024_northern_ireland.nc"
)

# Keep a copy before optimization
n_before = n.copy()

# Optimize
status, condition = n.optimize()

print(status, condition)

# n itself now contains the optimization results
n_after = n

comparison = pd.DataFrame({
    "Carrier": n_after.generators["carrier"],
    "Capacity_MW": n_after.generators["p_nom"],
    "Optimized_Generation_MWh":
        n_after.generators_t.p.sum()
})

print(comparison)

snapshot = n_after.snapshots[0]

flow_before = n_before.lines_t.p0.loc[snapshot]
flow_after = n_after.lines_t.p0.loc[snapshot]

flow_change = flow_after - flow_before

print(flow_change.sort_values())

gen_before = n_before.generators_t.p.loc[snapshot]
gen_after = n_after.generators_t.p.loc[snapshot]

gen_change = gen_after - gen_before

n.export_to_netcdf("/home/seba/Documents/Colleg/Hackathon_2/"
    "EIEG_Hackathon26/cian/EIEG_Hackathon26/"
    "data/SV2024_NI_after_optimization.nc")