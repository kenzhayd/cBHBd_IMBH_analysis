"""
clump_estimator.py

Simple estimator for the "Omega Cen assembled from many sub-clumps" idea.
How many separate IMBHs would be sitting in the central star cluster once the
clumps have sunk in, and what are their masses?

Physical picture assumed here:
  - Clumps form at t=0 and undergo runaway stellar collisions early (fast, ~0.1 Myr,
    per Rantala et al. since clump mass is low enough for runaway collisions)
    Seed formation itself is assumed instantaneous at t=0, using CBHBD.get_imbh_seed_rantala2026.
  - Each clump/seed starts at its a uniformly sampled radius between 0 and R_max_pc.
  - Each BH seed then sinks toward the center via dynamical friction; sink time is
    computed per clump from its starting radius.

If sink times for the clumps, is faster than some relevant time scale (couple million years?), 
The initial seeds can just be hard-coded directly into the initial conditions. 
Ex. CBHBD(..., imbh_mass=[m1, m2, ...])

If sink times are non-negligible, BH seeds need to be added to a column (3rd?)
of each seed's row in self.bhv. Right now every initial seed is hardcoded to tdf=0.0 in mergers.py:
    imbh_row = np.array([[float(mass), 0.0, 0.0, 1 + i]])
                                            ^^^ a per-clump sink time would go here??
Questions:
What is the best way to define R_max_pc?
Clump sinks or BH sinks?
Clump mass sampling slope?

"""

import numpy as np
from cbhbd import cbhbd


def split_into_equal_clumps(M_total, n_clumps):
    """Simplest possible mass split: n_clumps of equal mass."""
    return np.full(n_clumps, M_total / n_clumps)


def sample_clump_masses_powerlaw(M_total, M_min, M_max, n_clumps, slope=-2.0, rng=None, seed = None):
    """
    Draw n_clumps masses from a power-law dN/dM ~ M^slope between
    M_min and M_max (slope=-2?), then make masses sum to M_total.

    Reading about CDFs: https://stats.libretexts.org/Courses/Saint_Mary's_College_Notre_Dame/MATH_345__-_Probability_(Kuter)/4%3A_Continuous_Random_Variables/4.1%3A_Probability_Density_Functions_(PDFs)_and_Cumulative_Distribution_Functions_(CDFs)_for_Continuous_Random_Variables,
                        https://www.youtube.com/watch?v=wQ6Q9W3Y1ZE
    """
    rng = np.random.default_rng(seed)
    
    # draw n_seeds amount of random numbers uniformly between 0 and 1
    u = rng.random(n_clumps)
    

    # Inverse transform sampling from a truncated power law (sample random probability U, and then find the mass corresponding to that probability). 
    # First, integrate the mass distribution (like pdf) to get a cumulative mass distribution (tells probability of getting a mass less than or equal to M): 
    # CDF = ∫M^α dM (from M_min to M) = (M^(α+1))/(α+1) - (M_min^(α+1))/(α+1) = P(M_min < M < M))
    # At M = M_max, CDF = 1, so we can normalize the CDF by dividing by (M_max^(α+1) - M_min^(α+1))/(α+1)
    # then the normalized CDF = (M^(α+1) - M_min^(α+1)) / (M_max^(α+1) - M_min^(α+1)) = P(M_min < M < M))
    # Random sample P and find the mass corresponding to that P.
    
    # M^(α+1) = (M_max^(α+1) - M_min^(α+1)) * U + M_min^(α+1) --> random point = minimum + P(maximum−minimum) 
    # M = ((M_max^(α+1) - M_min^(α+1)) * U + M_min^(α+1))^(1/(α+1))

    if slope != -1:
        raw = (u * (M_max ** (slope + 1) - M_min ** (slope + 1)) + M_min ** (slope + 1)) ** (1 / (slope + 1))
    else:
        raw = M_min * (M_max / M_min) ** u #zindtead integrate with M^-1 --> ln(M), normalize and simplify 
    return raw * (M_total / raw.sum()) # Scale all sampled masses so the sum is the total forming cluster mass


def sink_time_myr(M_clump, cluster_pot, r_pc, rh, rhoh0):
    """
    Chandrasekhar dynamical friction infall time for a clump/seed starting at
    radius r_pc, sinking through a background of mass M_host.

    t_df = (1.17 / ln_lambda) * r_pc^2 * v_c (R) / (G * M_clump)
    Eq. 19.45 in https://galaxiesbook.org/chapters/IV-03.-Hierarchical-Galaxy-Formation_4-Dynamical-processes.html
    with v_c(r) = sqrt(G * M_(r) / r) the circular velocity of the enclosed
    background mass at r.

    Uses G = 0.004499 pc^3/Msun/Myr^2 (same constant cluster.py uses), so r_pc
    and M in Msun give t directly in Myr.

    ln_lambda: log_Coulomb = log(b_max / numpy.maximum(rh, G * M / v_typ**2)) --> log(maximum impact parameter/ minimum impact parameter)?
    where v_typ ~ v_c(r) --> typical relative speed between the sinking clump and the background cluster stars
    and b_max ("it is common to set b_max to the radius at which the decelerated body is orbiting" and "rh is the object's half mass radius")
    Eq. 19.40 in https://galaxiesbook.org/chapters/IV-03.-Hierarchical-Galaxy-Formation_4-Dynamical-processes.html
    """
    
    G = 0.004499  # [pc^3/Msun/Myr^2]
    
    # Find the circular velocity (~v_typ) from the King potential at the given radius.
    #
    # Rforce evaluates the cylindrical radial force F_R per unit mass?
    # Is this the same as gravitational acceleration? That's all that's needed anyways. 
    #
    # Cylindrical coordinates (R, z) --> spherical King potential
    # Force at z = 0 and R = r_pc is the dinstance from the center to the clump. 
    #
    # KingPotential is initialized in cluster.py with ro = 1 pc and vo = 1 km/s so a_R is in vo^2 / ro = (km/s)^2 / pc
    #
    # |F_r| = |a_R| = |a_circ|= G M(<r) / r^2 = v_c^2 / r,
    #
    # so v_c = sqrt(r * |a_R|) in km/s
    F_R = np.abs(cluster_pot.Rforce(r_pc, 0.0)) # Evaluating radial force per unit mass at (R,z)=(r_pc,0 --> angle)
    vc_kms = np.sqrt(r_pc * F_R)
    # Convert km/s to pc/Myr (1 km/s ~ 1.023 pc/Myr, matching cluster.py)
    vc = vc_kms * 1.023  

    # b_min uses the clump's half-mass radius from density rhoh0 = (M/2)/((4/3) pi r_h^3):
    r_h_clump = (3.0 * M_clump / (8.0 * np.pi * rhoh0)) ** (1.0/3.0)

    b_min = np.maximum(r_h_clump, G * M_clump / vc**2)

    # if the clump starts within the center of the cluster, skip the dynamical friction timescale calculation ==> clump is effectively already in the core.
    if r_pc <= b_min:
        return 0.0
    

    log_Coulomb = np.log(r_pc / b_min)
    
    # Return dynamical friction timescale
    return (1.17 / log_Coulomb) * r_pc ** 2 * vc / (G * M_clump)  # Myr
 

def estimate_clump_seeding(M_total, rhoh0, FeH,
                            mass_mode="equal", n_clumps=10,
                            M_min=None, M_max=None, slope=-2.0,
                            target_sunk_time_Myr=0.1, seed=None, W0=7):
    """
    M_total       : total forming cluster mass [Msun]
    rhoh0         : half-mass density assumed for every clump [Msun/pc^3]
    FeH           : metallicity [Fe/H] assumed for every clump
    R_max_pc      : outer radius of the forming cluster complex [pc] 
    mass_mode     : "equal"     -> n_clumps of equal mass M_total/n_clumps
                    "powerlaw"  -> n_clumps masses drawn from dN/dM ~ M^slope
                                   between M_min and M_max, rescaled to sum to
                                   M_total (requires M_min, M_max)
    target_sunk_time_Myr  : the short timescale needed for IMBH seeds to sink to the center and explain runaway BH mergers
    seed                  : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    if mass_mode == "equal":
        clump_masses = split_into_equal_clumps(M_total, n_clumps)
    elif mass_mode == "powerlaw":
        if M_min is None or M_max is None:
            raise ValueError("mass_mode='powerlaw' requires M_min and M_max.")
        clump_masses = sample_clump_masses_powerlaw(
            M_total, M_min, M_max, n_clumps, slope, rng)
    else:
        raise ValueError(f"Unknown mass_mode {mass_mode!r}; use 'equal' or 'powerlaw'.")

    n = len(clump_masses)
    
    # Run a cbhbd test model to get rh0 
    test_model = cbhbd.CBHBD(
        M0=M_total,
        rhoh0=rhoh0,
        FeH=FeH,
        rg=26.25,
        W0=W0, # Central potential 
        compute_mergers=False, # No mergers
        galpy_potential=True,  # KingPotential is applied
    )
    
    rh0 = test_model.cluster.rh0
    R_max_pc = rh0
    
    # Each clump BH seed forms at t=0 (since runaway collisions are fast?) at its own
    # randomly sampled starting radius within the forming cluster.
    radii_pc = rng.uniform(0, R_max_pc, size=n_clumps)

    seeds, sinks = [], []
    for M_cl, r_i in zip(clump_masses, radii_pc):
        seed_list = cbhbd.CBHBD.get_imbh_seed_rantala2026(M_cl, rhoh0, FeH)
        seeds.append(seed_list[0] if seed_list else 0.0)
        sinks.append(sink_time_myr(M_cl, test_model.cluster.cluster_pot, r_i, rh0, rhoh0))

    seeds = np.array(seeds)
    sinks = np.array(sinks)
    forming = seeds > 0

    print(f"Split {M_total:.1e} Msun cloud into {n} clumps ({mass_mode} mass "
          f"function{f', slope={slope}' if mass_mode == 'powerlaw' else ''})")
    print(f"rho_h0 = {rhoh0:.2e} Msun/pc^3, [Fe/H] = {FeH}, "
          f"Assumed cluster outer radius = {R_max_pc:.2e} pc "
          f"(initial clump radii drawn uniformly in [0, {R_max_pc:.2e}] pc)\n")

    print(f"{'clump':>6} {'M_clump [Msun]':>15} {'r0 [pc]':>10} "
          f"{'M_seed [Msun]':>15} {'t_sink [Myr]':>15}")
    for i, (Mc, r, m, t, ok) in enumerate(zip(clump_masses, radii_pc, seeds, sinks, forming)):
        print(f"{i:6d} {Mc:15.2e} {r:10.3f} {m:15.1f} {t:15.3g}"
              f"{'' if ok else '  (no IMBH forms)'}")

    n_imbh = int(forming.sum())
    print(f"\n{n_imbh} of {n} clumps form an IMBH seed.")
    print(f"Once all clumps have sunk in, the nuclear star cluster would start "
          f"with {n_imbh} separate IMBHs:")
    print(f"masses [Msun]: {np.round(seeds[forming], 1).tolist()}")
    print(f"summed seed mass: {seeds[forming].sum():.1f} Msun")

    if n_imbh == 0:
        print("\nNo clump reaches the density/metallicity threshold for IMBH "
              "formation -- try a higher rhoh0 or lower |FeH|.")
        return seeds, sinks, radii_pc

    fast = sinks[forming] < target_sunk_time_Myr
    if fast.all():
        print(f"\nAll sink times are below {target_sunk_time_Myr} Myr: the delay is "
              f"negligible. Maybe simple addition to single_model will work?\n"
              f"  CBHBD(..., imbh_mass={np.round(seeds[forming], 1).tolist()})\n")
    else:
        slow = sinks[forming][~fast]
        print(f"\n{(~fast).sum()} of {n_imbh} seeded clumps sink slower than "
              f"{target_sunk_time_Myr} Myr (up to {slow.max():.3g} Myr). Not negligible "
              f"Maybe good idea to set each seed's initial 'tdf' in mergers.py to its "
              f"own sink time instead of assuming instant arrival.")

    return seeds, sinks, radii_pc


if __name__ == "__main__":
    estimate_clump_seeding(
        M_total=1e7,
        rhoh0=10**5.62,
        FeH=-1.7,
        mass_mode="powerlaw",
        n_clumps=15,
        M_min=1e4,
        M_max=1e6,
        slope=-2.0,
        target_sunk_time_Myr=10,
        seed=42,
        W0=7
    )