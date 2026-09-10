import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

from ni_annealer_api import make_emulator


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# 1. NUMBA: PROPOSE A SWAP
# ============================================================

@njit
def propose_swap(grouping):
    """
    Pick two nodes belonging to different groups and swap
    their group memberships.

    Because this is a swap, the number of nodes in each
    group is preserved.
    """

    candidate = grouping.copy()

    n = len(grouping)

    # Pick two nodes
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)

    # Make sure they are different nodes and different groups
    while i == j or candidate[i] == candidate[j]:
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)

    # Swap their group assignments
    temp = candidate[i]
    candidate[i] = candidate[j]
    candidate[j] = temp

    return candidate


# ============================================================
# 2. CONVERT GROUPING -> NODES DATAFRAME
# ============================================================

def grouping_to_nodes(grouping, nodes):
    """
    Convert the NumPy grouping used by the annealer into
    the DataFrame format expected by dispatch_down().
    """

    candidate = nodes.copy(deep=True)

    candidate["groups"] = [
        (int(group),)
        for group in grouping
    ]

    return candidate


# ============================================================
# 3. ENERGY FUNCTION
# ============================================================

def energy(grouping, nodes, blackbox):
    """
    Calculate the energy of a grouping.

    Energy = dispatch-down percentage.

    Lower energy is better.
    """

    candidate = grouping_to_nodes(
        grouping,
        nodes
    )

    return blackbox(candidate)


# ============================================================
# 4. TEMPERATURE ESTIMATION
# ============================================================

def estimate_temperature_range_blackbox(
    grouping,
    nodes,
    blackbox,
    n_samples=20,
    hot_accept_prob=0.5,
    cold_accept_prob=1e-3,
    verbose=True
):
    """
    Probe the black box with random swaps to estimate
    sensible starting and ending temperatures.

    Uses:

        P = exp(-dE / T)

    therefore:

        T = -dE / ln(P)
    """

    E = energy(
        grouping,
        nodes,
        blackbox
    )

    dEs = []

    for sample in range(n_samples):

        # Propose a random swap
        candidate = propose_swap(grouping)

        # Evaluate it
        E_new = energy(
            candidate,
            nodes,
            blackbox
        )

        dE = abs(E_new - E)

        if dE > 0:
            dEs.append(dE)

        # Continue random walk
        grouping = candidate
        E = E_new

        if verbose:
            print(
                f"  Temperature probe "
                f"{sample + 1}/{n_samples} | "
                f"E = {E:.4f}%"
            )

    dEs = np.array(dEs)

    if len(dEs) == 0:
        raise RuntimeError(
            "No non-zero energy changes were found "
            "during temperature estimation."
        )

    # Typical change
    dE_typical = np.percentile(
        dEs,
        50
    )

    # Small change
    dE_small = np.percentile(
        dEs,
        5
    )

    # Initial temperature
    T_start = (
        -dE_typical
        / np.log(hot_accept_prob)
    )

    # Final temperature
    T_end = (
        -dE_small
        / np.log(cold_accept_prob)
    )

    return T_start, T_end


# ============================================================
# 5. SIMULATED ANNEALING
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

    Starts from the existing SONI grouping.

    Each move swaps the groups of two nodes, so group sizes
    remain fixed.
    """

    # --------------------------------------------------------
    # Convert DataFrame groups into NumPy array
    # --------------------------------------------------------

    grouping = np.array(
        [
            groups[0]
            for groups in nodes["groups"]
        ],
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Initial energy
    # --------------------------------------------------------

    E = energy(
        grouping,
        nodes,
        blackbox
    )

    best_grouping = grouping.copy()
    best_E = E

    # Count expensive black-box calls
    blackbox_calls = 1

    print()
    print("=" * 60)
    print("SIMULATED ANNEALING")
    print("=" * 60)

    print(
        f"Starting energy: "
        f"{E:.4f}%"
    )

    print(
        f"Number of nodes: "
        f"{len(grouping)}"
    )

    print(
        f"Groups present: "
        f"{np.unique(grouping)}"
    )

    # --------------------------------------------------------
    # Show group sizes
    # --------------------------------------------------------

    unique_groups, counts = np.unique(
        grouping,
        return_counts=True
    )

    print()
    print("Initial group sizes:")

    for group, count in zip(
        unique_groups,
        counts
    ):
        print(
            f"  Group {group}: "
            f"{count} nodes"
        )

    # --------------------------------------------------------
    # Estimate temperature range
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ESTIMATING TEMPERATURE RANGE")
    print("=" * 60)

    T_start, T_end = (
        estimate_temperature_range_blackbox(
            grouping=grouping.copy(),
            nodes=nodes,
            blackbox=blackbox,
            n_samples=n_temperature_samples,
            hot_accept_prob=hot_accept_prob,
            cold_accept_prob=cold_accept_prob,
            verbose=verbose
        )
    )

    # Add temperature-probe calls
    blackbox_calls += n_temperature_samples

    print()
    print(
        f"T_start = {T_start:.6f}"
    )

    print(
        f"T_end   = {T_end:.6f}"
    )

    print(
        f"alpha   = {alpha}"
    )

    # --------------------------------------------------------
    # Annealing
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("BEGINNING ANNEALING")
    print("=" * 60)

    T = T_start

    temperature_step = 0

    history = []

    while T > T_end:

        temperature_step += 1

        accepted = 0
        improved = 0

        # ----------------------------------------------------
        # Moves at this temperature
        # ----------------------------------------------------

        for _ in range(sweeps_per_temp):

            # Propose new grouping
            candidate = propose_swap(
                grouping
            )

            # Evaluate candidate
            E_new = energy(
                candidate,
                nodes,
                blackbox
            )

            blackbox_calls += 1

            # Energy difference
            dE = E_new - E

            # ------------------------------------------------
            # Metropolis criterion
            # ------------------------------------------------

            if (
                dE < 0
                or np.random.random()
                < np.exp(-dE / T)
            ):

                grouping = candidate
                E = E_new

                accepted += 1

                # ------------------------------------------------
                # New global best?
                # ------------------------------------------------

                if E < best_E:

                    best_grouping = (
                        grouping.copy()
                    )

                    best_E = E

                    improved += 1

                    print(
                        f"    NEW BEST -> "
                        f"{best_E:.4f}%"
                    )

        # ----------------------------------------------------
        # Acceptance rate
        # ----------------------------------------------------

        acceptance_rate = (
            accepted / sweeps_per_temp
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append(
            {
                "temperature_step":
                    temperature_step,

                "temperature":
                    T,

                "current_energy":
                    E,

                "best_energy":
                    best_E,

                "accepted":
                    accepted,

                "acceptance_rate":
                    acceptance_rate,

                "new_bests":
                    improved,

                "blackbox_calls":
                    blackbox_calls,
            }
        )

        # ----------------------------------------------------
        # PRINT PROGRESS
        # ----------------------------------------------------

        print(
            f"T step {temperature_step:3d} | "
            f"T = {T:.6f} | "
            f"current = {E:.4f}% | "
            f"best = {best_E:.4f}% | "
            f"accepted = {accepted}/{sweeps_per_temp} "
            f"({acceptance_rate:.0%}) | "
            f"calls = {blackbox_calls}"
        )

        # ----------------------------------------------------
        # Cool system
        # ----------------------------------------------------

        T *= alpha

    # ========================================================
    # CONVERT BEST GROUPING BACK TO DATAFRAME
    # ========================================================

    best_nodes = grouping_to_nodes(
        best_grouping,
        nodes
    )

    print()
    print("=" * 60)
    print("ANNEALING FINISHED")
    print("=" * 60)

    print(
    f"Initial energy: "
    f"{E:.4f}%"
)

    print(
        f"Best energy: "
        f"{best_E:.4f}%"
    )

    print(
        f"Black-box calls: "
        f"{blackbox_calls}"
    )

    return (
        best_nodes,
        best_E,
        history
    )


# ============================================================
# 6. START NORTHERN IRELAND GRID EMULATOR
# ============================================================

print("=" * 60)
print("STARTING NI GRID EMULATOR")
print("=" * 60)

dispatch_down, _ = make_emulator(
    runs=10_000,
    seed=42,
)


# ============================================================
# 7. LOAD CURRENT NODE / CONSTRAINT GROUP CONFIGURATION
# ============================================================

nodes = pd.read_csv(
    OUTPUT_DIR / "annealer_nodes_current.csv"
)


# ============================================================
# 8. CONVERT GROUP STRINGS INTO TUPLES
# ============================================================

nodes["groups"] = nodes["groups"].apply(
    lambda x: tuple(
        int(g)
        for g in x.strip("()").split(",")
        if g.strip()
    )
)


print()
print("=" * 60)
print("LOADED NODE CONFIGURATION")
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
# 9. CALCULATE SONI BASELINE
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
# 10. SMALL TEST
# ============================================================
#
# IMPORTANT:
#
# This is deliberately a SMALL test.
#
# Once this works, we can increase:
#
#   n_temperature_samples
#   sweeps_per_temp
#   alpha
#
# ============================================================

best_nodes, best_dispatch_down, history = (
    sim_annealing_grouping(

        nodes=nodes,

        blackbox=dispatch_down,

        # Fast cooling for test
        alpha=0.80,

        # Only 3 moves per temperature
        sweeps_per_temp=3,

        # Only 5 expensive calls to probe T
        n_temperature_samples=5,

        hot_accept_prob=0.5,

        cold_accept_prob=1e-3,

        verbose=True
    )
)


# ============================================================
# 11. FINAL COMPARISON
# ============================================================

print()
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    f"SONI baseline:       "
    f"{baseline_dispatch_down:.4f}%"
)

print(
    f"Annealed grouping:   "
    f"{best_dispatch_down:.4f}%"
)

print(
    f"Difference:          "
    f"{baseline_dispatch_down - best_dispatch_down:.4f} "
    f"percentage points"
)

if baseline_dispatch_down != 0:

    improvement = (
        (
            baseline_dispatch_down
            - best_dispatch_down
        )
        / baseline_dispatch_down
        * 100
    )

    print(
        f"Relative improvement:"
        f" {improvement:.2f}%"
    )


# ============================================================
# 12. SHOW BEST GROUPING
# ============================================================

print()
print("=" * 60)
print("BEST GROUPING FOUND")
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
# 13. SAVE TEST RESULT
# ============================================================

best_nodes.to_csv(
    OUTPUT_DIR / "optimised_constraint_groups.csv",
    index=False
)

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "annealing_history.csv",
    index=False
)

print()
print(
    "Saved best grouping to:"
)

print(
    OUTPUT_DIR / "optimised_constraint_groups.csv"
)

print(
    "Saved annealing history to:"
)

print(
    OUTPUT_DIR / "annealing_history.csv"
)