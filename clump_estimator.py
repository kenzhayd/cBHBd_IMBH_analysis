"""
clumpestimator.py

Simple estimator for the "Omega Cen assembled from many sub-clumps" situation.
How many separate IMBHs would be sitting in the nuclear
star cluster once the clumps have sunk in, and what are their masses?

Splits a progenitor cloud of mass M_total into N equal-mass sub-clumps, assigns
each the same birth density rhoh0, and uses CBHBD.get_imbh_seed_rantala2026 to 
get each clump's IMBH seed mass.
Then estimates how long each clump takes to sink to the centre of the assembling
complex via dynamical friction, using the dwarf galaxy potential.

If sink times for the clumps, is faster than the relevant time scale, 
The initial seeds can just be hard-coded directly into the initial conditions. 
Ex. CBHBD(..., imbh_mass=[m1, m2, ...])

If sink times are non-negligible, they need to be added to a column (3rd?)
of each seed's row in self.bhv ("tdf" -- the same field mergers.py already uses to
delay a merger product's infall into the center after a kick). Right now every
initial seed is hardcoded to tdf=0.0:


Usage:
    python clumpestimator.py
"""

import numpy as np
from cbhbd import cbhbd


def split_into_equal_clumps(M_total, n_clumps):
    """Simplest possible mass split: n_clumps of equal mass."""
    return np.full(n_clumps, M_total / n_clumps)


def sink_time_myr(M_clump, M_host, r_pc, ln_lambda=10.0):
    """
    Dynamical friction infall time for a clump orbiting within the central region of the dwarf galaxy.

    Galactic Dynamics pg 660: t_df ~ (1.17/lnLambda) * r * v_c^2 / (G*M_clump),
   
    G = 0.004499 pc^3/Msun/Myr^2 that cluster.py uses, so
    r_pc and M in Msun give t directly in Myr with no unit conversion.
    
    Get formula for ln_lambda on pg. 644
    """
    G = 0.004499
    vc = np.sqrt(G * M_host / r_pc)  # pc/Myr
    return (1.17 / ln_lambda) * r_pc * vc**2 / (G * M_clump)  # Myr


def estimate_clump_seeding(M_total, rhoh0, FeH, r_gal_pc,
                            n_clumps=10, t_window_myr=0.1):
    """
    M_total       : total progenitor cloud mass [Msun]
    rhoh0         : half-mass density assumed for all clump [Msun/pc^3]
    FeH           : metallicity [Fe/H]
    r_gal_pc      : characteristic radius of forming galaxy [pc] -- what radius is this??? radius of sphere
    n_clumps      : how many sub-clumps the cloud is splitting into
    t_window_myr  : the relevant sinking timescale (default 0.1 Myr)
    """
    clump_masses = split_into_equal_clumps(M_total, n_clumps)

    seeds, sinks = [], []
    for M_cl in clump_masses:
        seed_list = cbhbd.CBHBD.get_imbh_seed_rantala2026(M_cl, rhoh0, FeH)
        seeds.append(seed_list[0] if seed_list else 0.0)
        sinks.append(sink_time_myr(M_cl, M_total, r_gal_pc))

    seeds, sinks = np.array(seeds), np.array(sinks)
    forming = seeds > 0

    print(f"Split {M_total:.1e} Msun cloud into {n_clumps} clumps of "
          f"{M_total / n_clumps:.1e} Msun each")
    print(f"rho_h0 = {rhoh0:.2e} Msun/pc^3, [Fe/H] = {FeH}, "
          f"Dwarf galaxy size = {r_gal_pc:.0f} pc\n")

    print(f"{'clump':>6} {'M_seed [Msun]':>15} {'t_sink [Myr]':>15}")
    for i, (m, t, ok) in enumerate(zip(seeds, sinks, forming)):
        print(f"{i:6d} {m:15.1f} {t:15.3g}{'' if ok else '  (no IMBH forms)'}")

    n_imbh = int(forming.sum())
    print(f"\n{n_imbh} of {n_clumps} clumps form an IMBH seed.")
    print(f"Once all clumps have sunk in, the nuclear star cluster would start "
          f"with {n_imbh} separate IMBHs:")
    print(f"  masses [Msun]: {np.round(seeds[forming], 1).tolist()}")
    print(f"  summed mass if they all eventually merge: {seeds[forming].sum():.1f} Msun")

    if n_imbh == 0:
        print("\nNo clump reaches the density/metallicity threshold for IMBH.")
        return seeds, sinks

    fast = sinks[forming] < t_window_myr
    if fast.all():
        print(f"\nAll sink times are below {t_window_myr} Myr: the delay is "
              f"negligible.")
    else:
        slow = sinks[forming][~fast]
        print(f"\n{(~fast).sum()} of {n_imbh} seeded clumps sink slower than "
              f"{t_window_myr} Myr (up to {slow.max():.3g} Myr). This is not negligible.")

    return seeds, sinks


if __name__ == "__main__":
    # Example using the same order-of-magnitude ICs as your create_job_list.py grid
    estimate_clump_seeding(
        M_total=1e7,
        rhoh0=1e6,
        FeH=-1.7,
        r_gal_pc=50,
        n_clumps=10,
        t_window_myr=1,
    )