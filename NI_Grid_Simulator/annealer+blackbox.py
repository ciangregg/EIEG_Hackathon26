import numpy as np
import pandas as pd
from pathlib import Path
from ni_annealer_api import make_emulator
from numba import njit

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# ============================================================
# 1. ENERGY FUNCTION
# ============================================================

def energy(grouping, nodes, blackbox):
    """
    Energy of a grouping.

    Lower dispatch-down = lower energy 

    grouping:
        1D numpy array containing the group assigned to each node

    nodes:
        Original pandas DataFrame containing the grid nodes

    blackbox:
        Saoirses' dispatch_down() function
    """

    candidate = nodes.copy(deep=True)

    # Convert integer group labels into the tuple format
    # expected by the emulator.
    candidate["groups"] = [
        (int(g),)
        for g in grouping
    ]

    return blackbox(candidate)


# ============================================================
# 2. PROPOSE A NEW GROUPING
# ============================================================
@njit
def propose_swap(grouping):
    """
    Propose a new grouping by swapping the groups of two nodes.

    This preserves the number of nodes in each group.
    """

    candidate = grouping.copy()

    # Pick two different nodes
    i, j = np.random.choice(
        len(grouping),
        size=2,
        replace=False
    )

    # If they are already in the same group, try again.
    while candidate[i] == candidate[j]:

        i, j = np.random.choice(
            len(grouping),
            size=2,
            replace=False
        )

    # Swap their group memberships
    candidate[i], candidate[j] = (
        candidate[j],
        candidate[i]
    )

    return candidate


# ============================================================
# 3. ESTIMATE TEMPERATURE RANGE
# ============================================================

def estimate_temperature_range_blackbox(
    grouping,
    nodes,
    blackbox,
    n_samples=20,
    hot_accept_prob=0.5,
    cold_accept_prob=1e-3
):
    """
    Estimate sensible starting and ending temperatures.

    We probe the blackbox with random swaps and look at
    the typical change in dispatch-down.
    """

    E = energy(
        grouping,
        nodes,
        blackbox
    )

    dEs = []

    for _ in range(n_samples):

        # Propose a random swap
        candidate = propose_swap(grouping)

        # Evaluate candidate
        E_new = energy(
            candidate,
            nodes,
            blackbox
        )

        # Store magnitude of energy change
        dEs.append(
            abs(E_new - E)
        )

        # Random walk for probing
        grouping = candidate
        E = E_new

    dEs = np.array(dEs)

    # Remove zero-energy changes
    dEs = dEs[dEs > 0]

    if len(dEs) == 0:
        raise RuntimeError(
            "All sampled moves had dE = 0. "
            "Cannot estimate temperature range."
        )

    # Typical and small energy changes
    dE_typical = np.percentile(
        dEs,
        50
    )

    dE_small = np.percentile(
        dEs,
        5
    )

    # From
    #
    # P = exp(-dE/T)
    #
    # therefore
    #
    # T = -dE / ln(P)

    T_start = (
        -dE_typical
        / np.log(hot_accept_prob)
    )

    T_end = (
        -dE_small
        / np.log(cold_accept_prob)
    )

    return T_start, T_end


# ============================================================
# 4. SIMULATED ANNEALING
# ============================================================

def sim_annealing_grouping(
    nodes,
    blackbox,
    alpha=0.95,
    sweeps_per_temp=20,
    n_temperature_samples=20,
    hot_accept_prob=0.5,
    cold_accept_prob=1e-3,
    verbose=True
):
    """
    Simulated annealing over constraint-group assignments.

    The initial grouping is taken directly from the SONI
    configuration in `nodes`.

    A move consists of swapping the group assignments of
    two nodes.

    Returns:
        best_nodes
        best_energy
        history
    """

    # --------------------------------------------------------
    # Convert the existing SONI grouping into an integer array
    # --------------------------------------------------------

    grouping = np.array([
        groups[0]
        for groups in nodes["groups"]
    ], dtype=int)

    # --------------------------------------------------------
    # Calculate initial energy
    # --------------------------------------------------------

    E = energy(
        grouping,
        nodes,
        blackbox
    )

    if verbose:
        print()
        print("=" * 60)
        print("INITIAL CONFIGURATION")
        print("=" * 60)
        print(f"Initial dispatch-down: {E:.4f}%")

    # --------------------------------------------------------
    # Estimate temperature range
    # --------------------------------------------------------

    T_start, T_end = (
        estimate_temperature_range_blackbox(
            grouping=grouping.copy(),
            nodes=nodes,
            blackbox=blackbox,
            n_samples=n_temperature_samples,
            hot_accept_prob=hot_accept_prob,
            cold_accept_prob=cold_accept_prob
        )
    )

    if verbose:
        print()
        print("TEMPERATURE RANGE")
        print("=" * 60)
        print(f"T_start = {T_start:.6f}")
        print(f"T_end   = {T_end:.6f}")
        print(f"alpha   = {alpha}")
        print()

    # --------------------------------------------------------
    # Store best solution found
    # --------------------------------------------------------

    best_grouping = grouping.copy()
    best_E = E

    history = []

    # --------------------------------------------------------
    # Annealing loop
    # --------------------------------------------------------

    T = T_start

    iteration = 0

    while T > T_end:

        accepted = 0

        for _ in range(sweeps_per_temp):

            iteration += 1

            # -----------------------------------------------
            # Propose a new grouping
            # -----------------------------------------------

            candidate = propose_swap(
                grouping
            )

            # -----------------------------------------------
            # Evaluate new grouping
            # -----------------------------------------------

            E_new = energy(
                candidate,
                nodes,
                blackbox
            )

            # -----------------------------------------------
            # Energy difference
            # -----------------------------------------------

            dE = E_new - E

            # -----------------------------------------------
            # Metropolis acceptance criterion
            # -----------------------------------------------

            if (
                dE < 0
                or np.random.rand()
                < np.exp(-dE / T)
            ):

                grouping = candidate
                E = E_new

                accepted += 1

                # -------------------------------------------
                # Update best solution
                # -------------------------------------------

                if E < best_E:

                    best_grouping = (
                        grouping.copy()
                    )

                    best_E = E

        # ----------------------------------------------------
        # Record history
        # ----------------------------------------------------

        history.append({
            "iteration": iteration,
            "temperature": T,
            "energy": E,
            "best_energy": best_E,
            "acceptance_rate": (
                accepted / sweeps_per_temp
            )
        })

        if verbose:
            print(
                f"T = {T:.6f} | "
                f"E = {E:.4f}% | "
                f"best = {best_E:.4f}% | "
                f"accept = "
                f"{accepted / sweeps_per_temp:.2f}"
            )

        # ----------------------------------------------------
        # Cool system
        # ----------------------------------------------------

        T *= alpha

    # ========================================================
    # CONVERT BEST GROUPING BACK TO DATAFRAME
    # ========================================================

    best_nodes = nodes.copy(deep=True)

    best_nodes["groups"] = [
        (int(g),)
        for g in best_grouping
    ]

    return (
        best_nodes,
        best_E,
        history
    )


# ============================================================
# 5. START THE NORTHERN IRELAND GRID EMULATOR
# ============================================================

print("=" * 60)
print("STARTING NI GRID EMULATOR")
print("=" * 60)

dispatch_down, _ = make_emulator(
    runs=10_000,
    seed=42,
)


# ============================================================
# 6. LOAD CURRENT NODE / CONSTRAINT-GROUP CONFIGURATION
# ============================================================

nodes = pd.read_csv(
    OUTPUT_DIR / "annealer_nodes_current.csv"
)


# CSV stores tuples as strings, e.g.
#
# "(1,)"
# "(2,)"
# "(1, 3)"
#
# Convert them back into tuples.

nodes["groups"] = nodes["groups"].apply(
    lambda x: tuple(
        int(g)
        for g in x.strip("()").split(",")
        if g.strip()
    )
)


print()
print("=" * 60)
print("CURRENT NODE CONFIGURATION")
print("=" * 60)

print(
    nodes[
        [
            "node_id",
            "name",
            "bus",
            "groups"
        ]
    ]
)


# ============================================================
# 7. CALCULATE CURRENT SONI BASELINE
# ============================================================

baseline_dispatch_down = dispatch_down(
    nodes
)

print()
print("=" * 60)
print("SONI BASELINE")
print("=" * 60)

print(
    f"Baseline dispatch-down: "
    f"{baseline_dispatch_down:.4f}%"
)


# ============================================================
# 8. RUN SIMULATED ANNEALING
# ============================================================

best_nodes, best_dispatch_down, history = (
    sim_annealing_grouping(
        nodes=nodes,
        blackbox=dispatch_down,

        # Cooling schedule
        alpha=0.95,

        # Number of proposed swaps per temperature
        sweeps_per_temp=20,

        # Number of blackbox calls used to estimate T
        n_temperature_samples=20,

        # Desired acceptance probability at beginning
        hot_accept_prob=0.5,

        # Desired acceptance probability at end
        cold_accept_prob=1e-3,

        verbose=True
    )
)


# ============================================================
# 9. RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Baseline dispatch-down:  "
    f"{baseline_dispatch_down:.4f}%"
)

print(
    f"Optimised dispatch-down: "
    f"{best_dispatch_down:.4f}%"
)

print(
    f"Improvement: "
    f"{baseline_dispatch_down - best_dispatch_down:.4f} "
    f"percentage points"
)

if baseline_dispatch_down != 0:

    improvement_percent = (
        (
            baseline_dispatch_down
            - best_dispatch_down
        )
        / baseline_dispatch_down
        * 100
    )

    print(
        f"Relative improvement: "
        f"{improvement_percent:.2f}%"
    )


# ============================================================
# 10. SHOW OPTIMISED GROUPING
# ============================================================

print()
print("=" * 60)
print("OPTIMISED NODE CONFIGURATION")
print("=" * 60)

print(
    best_nodes[
        [
            "node_id",
            "name",
            "bus",
            "groups"
        ]
    ]
)


# ============================================================
# 11. SAVE OPTIMISED CONFIGURATION
# ============================================================

best_nodes.to_csv(
    OUTPUT_DIR / "optimised_constraint_groups.csv",
    index=False
)

print()
print(
    "Saved optimised configuration to:"
)

print(
    "outputs/optimised_constraint_groups.csv"
)


# ============================================================
# 12. SAVE ANNEALING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUTPUT_DIR / "annealing_history.csv",
    index=False
)

print(
    "Saved annealing history to:"
)

print(
    "outputs/annealing_history.csv"
)