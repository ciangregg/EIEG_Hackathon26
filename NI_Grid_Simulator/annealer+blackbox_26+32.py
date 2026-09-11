import numpy as np
import pandas as pd
from pathlib import Path
import sys
from numba import njit

from all_island_annealer_api import make_emulator


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. NUMBA: PROPOSE A SWAP
# ============================================================

@njit
def propose_swap(grouping):
    """
    Pick two nodes belonging to different groups
    and swap their group assignments.

    The number of nodes in each group is therefore
    preserved.
    """

    candidate = grouping.copy()

    n = len(grouping)

    i = np.random.randint(0, n)
    j = np.random.randint(0, n)

    while i == j or candidate[i] == candidate[j]:

        i = np.random.randint(0, n)
        j = np.random.randint(0, n)

    temp = candidate[i]
    candidate[i] = candidate[j]
    candidate[j] = temp

    return candidate


# ============================================================
# 2. CONVERT INTEGER GROUPING -> NODES DATAFRAME
# ============================================================

def grouping_to_nodes(grouping, nodes):
    """
    Convert the integer grouping used by the annealer into
    the tuple representation expected by the emulator.

    Example:

        4 -> (4,)
        2 -> (2,)
        1 -> (1,)
    """

    # All non-group columns are read-only in the annealer. A shallow copy is
    # sufficient because assigning the groups column creates a new column and
    # leaves the input DataFrame unchanged.
    candidate = nodes.copy(deep=False)

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
    Evaluate the dispatch-down for a proposed grouping.

    This public helper keeps the original generic black-box contract.
    The simulated annealer prepares a faster evaluator once when the
    supplied blackbox exposes the optional tandem optimisation hook.
    """

    candidate = grouping_to_nodes(
        grouping,
        nodes
    )

    return blackbox(candidate)


def _prepare_energy_evaluator(nodes, blackbox):
    """Prepare the cheapest equivalent grouping -> energy callable available."""

    prepare = getattr(
        blackbox,
        "prepare_grouping_evaluator",
        None
    )

    if callable(prepare):
        return prepare(nodes)

    # Compatibility fallback for any other black-box function.
    return lambda grouping: energy(
        grouping,
        nodes,
        blackbox
    )


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
    verbose=True,
    initial_E=None,
    energy_evaluator=None
):
    """
    Estimate a useful starting and ending temperature
    from the energy scale of random swap moves.

    T_start is chosen so that a typical uphill move
    has approximately hot_accept_prob probability of
    being accepted.

    T_end is chosen so that a small uphill move has
    approximately cold_accept_prob probability of
    being accepted.
    """

    if energy_evaluator is None:
        energy_evaluator = _prepare_energy_evaluator(
            nodes,
            blackbox
        )

    # sim_annealing_grouping already knows this value. Reusing it removes one
    # full black-box evaluation without changing the temperature calculation.
    E = (
        energy_evaluator(grouping)
        if initial_E is None
        else initial_E
    )

    dEs = []

    for sample in range(n_samples):

        candidate = propose_swap(
            grouping
        )

        E_new = energy_evaluator(
            candidate
        )

        # security_guard can deliberately return +inf for a grouping that
        # creates a new security failure. Such a point is a rejected/forbidden
        # state, not an energy scale from which to estimate temperature.
        if np.isfinite(E_new):
            dE = abs(E_new - E)
            if dE > 0 and np.isfinite(dE):
                dEs.append(dE)

            # Random walk only through valid finite states.
            grouping = candidate
            E = E_new

        if verbose:
            shown = f"{E_new:.4f}%" if np.isfinite(E_new) else "REJECTED (security)"
            print(
                f"  Temperature probe "
                f"{sample + 1}/{n_samples} | "
                f"E = {shown}"
            )

    dEs = np.array(dEs)

    if len(dEs) == 0:
        # Flat local objective: use a very small finite schedule rather than
        # crashing. The annealer can still explore valid zero-delta swaps.
        if verbose:
            print("  No finite non-zero probe deltas found; using fallback temperatures.")
        return 1e-3, 1e-6

    dE_typical = np.percentile(
        dEs,
        50
    )

    dE_small = np.percentile(
        dEs,
        5
    )

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
# 5. SIMULATED ANNEALING
# ============================================================

def sim_annealing_grouping(
    nodes,
    blackbox,
    alpha=0.9,
    sweeps_per_temp=10,
    n_temperature_samples=20,
    hot_accept_prob=0.5,
    cold_accept_prob=1e-3,
    verbose=True
):
    """
    Simulated annealing over exclusive integer group
    assignments.

    Each node has exactly one integer group:

        grouping[i] = 1, 2, 3, ...

    A move swaps the groups of two nodes, preserving
    the number of nodes in each group.
    """

    # --------------------------------------------------------
    # INITIAL GROUPING
    # --------------------------------------------------------

    grouping = np.array(
        [
            int(groups[0])
            for groups in nodes["groups"]
        ],
        dtype=np.int64
    )

    # --------------------------------------------------------
    # INITIAL ENERGY
    # --------------------------------------------------------

    energy_evaluator = _prepare_energy_evaluator(
        nodes,
        blackbox
    )

    E = energy_evaluator(
        grouping
    )

    # Keep a separate copy so it never changes
    initial_E = E

    best_grouping = grouping.copy()
    best_E = E

    blackbox_calls = 1

    print()
    print("=" * 60)
    print("SIMULATED ANNEALING")
    print("=" * 60)

    print(
        f"Starting energy: "
        f"{initial_E:.4f}%"
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
    # INITIAL GROUP SIZES
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
    # TEMPERATURE ESTIMATION
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
            verbose=verbose,
            initial_E=E,
            energy_evaluator=energy_evaluator
        )
    )

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
    # BEGIN ANNEALING
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
        # SWAPS AT CURRENT TEMPERATURE
        # ----------------------------------------------------

        for _ in range(sweeps_per_temp):

            candidate = propose_swap(
                grouping
            )

            E_new = energy_evaluator(
                candidate
            )

            blackbox_calls += 1

            dE = E_new - E

            # ------------------------------------------------
            # METROPOLIS ACCEPTANCE
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
                # NEW BEST
                # ------------------------------------------------

                if E < best_E:

                    best_grouping = grouping.copy()

                    best_E = E

                    improved += 1

                    print(
                        f"    NEW BEST -> "
                        f"{best_E:.4f}%"
                    )

        # ----------------------------------------------------
        # ACCEPTANCE RATE
        # ----------------------------------------------------

        acceptance_rate = (
            accepted
            / sweeps_per_temp
        )

        # ----------------------------------------------------
        # SAVE HISTORY
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
        # PROGRESS OUTPUT
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
        # COOL
        # ----------------------------------------------------

        T *= alpha

    # ========================================================
    # CONVERT BEST INTEGER GROUPING BACK TO DATAFRAME
    # ========================================================

    best_nodes = grouping_to_nodes(
        best_grouping,
        nodes
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print("ANNEALING FINISHED")
    print("=" * 60)

    print(
        f"Initial energy: "
        f"{initial_E:.4f}%"
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
# 6. START 26- OR 32-COUNTY GRID EMULATOR
# ============================================================

# Usage:
#   python3 annealer+blackbox_26+32.py 32
#   python3 annealer+blackbox_26+32.py 26
# Defaults to the 32-county all-island case.
SCOPE = sys.argv[1].strip() if len(sys.argv) > 1 else "32"
if SCOPE not in {"26", "32"}:
    raise SystemExit("Usage: python3 annealer+blackbox_26+32.py [26|32]")

print("=" * 60)
print(f"STARTING {SCOPE}-COUNTY SV2024 GRID EMULATOR")
print("=" * 60)

# IMPORTANT: use the node table returned by make_emulator().  It is loaded from
# the matching SV2024 WDT CSV and is guaranteed to match the black-box template.
dispatch_down, nodes = make_emulator(
    scope=SCOPE,
    runs=10_000,
    seed=42,
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
# 9. CALCULATE CURRENT WDT BASELINE
# ============================================================

baseline_dispatch_down = dispatch_down(
    nodes
)

print()
print("=" * 60)
print("CURRENT WDT BASELINE")
print("=" * 60)

print(
    f"Baseline dispatch-down: "
    f"{baseline_dispatch_down:.4f}%"
)


# ============================================================
# 10. SMALL TEST
# ============================================================

best_nodes, best_dispatch_down, history = (
    sim_annealing_grouping(

        nodes=nodes,

        blackbox=dispatch_down,

        # Small test settings
        alpha=0.99,

        sweeps_per_temp=20,

        n_temperature_samples=10,

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
    f"Current WDT baseline:       "
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
    OUTPUT_DIR / f"optimised_constraint_groups_{SCOPE}_counties.csv",
    index=False
)

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / f"annealing_history_{SCOPE}_counties.csv",
    index=False
)

print()
print(
    "Saved best grouping to:"
)

print(
    OUTPUT_DIR / f"optimised_constraint_groups_{SCOPE}_counties.csv"
)

print(
    "Saved annealing history to:"
)

print(
    OUTPUT_DIR / f"annealing_history_{SCOPE}_counties.csv"
)