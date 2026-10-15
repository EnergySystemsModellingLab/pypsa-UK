# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning)
is provided in the
[documentation of PyPSA](https://docs.pypsa.org/latest/user-guide/network-optimization/).

The optimization is based on the `network.optimize` function.
Additionally, some extra constraints specified in [solve_network][] are added.
"""

import importlib
import logging
import os
import re
import sys
from functools import partial
from typing import Any

import linopy
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from linopy.constants import SolverStatus, TerminationCondition
from linopy.remote.oetc import OetcCredentials, OetcHandler, OetcSettings
from pypsa.descriptors import get_activity_mask

from scripts.prepare_sector_network import determine_emission_sectors
from scripts._benchmark import memory_logger
from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    get,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

# Allow for PyPSA versions <0.35
if PYPSA_V1:
    pypsa.network.power_flow.logger.setLevel(logging.WARNING)
else:
    pypsa.pf.logger.setLevel(logging.WARNING)


class ObjectiveValueError(Exception):
    pass

def calculate_co2_limit(
    investment_year: int,
    co2_budget: dict,
    regional_emissions: float,
    total_emissions: float,
) -> float | None:
    """Return the annual CO2 allowance in tonnes for a regional group."""
    limit = get(co2_budget["upper"], investment_year)
    if limit is None:
        return None
    if co2_budget["relative"]:
        return regional_emissions * limit
    if total_emissions == 0:
        return 0.0
    return limit * 1e9 * regional_emissions / total_emissions


def valid_local_co2_regions(
    local_co2: dict,
    co2_totals: pd.DataFrame,
    investment_year: int,
    countries: list[str],
) -> list[str]:
    """Return only configured local-CO2 regions with baseline data and a current limit."""
    valid = []
    for region, limits in local_co2.items():
        if (
            region in countries
            and region in co2_totals.index
            and investment_year in limits
        ):
            valid.append(region)
        else:
            logger.info(
                "Region %s has no applicable local CO2 constraint for %s; "
                "including it in the collective budget.",
                region,
                investment_year,
            )
    return valid


# TODO: move to compose_network
def add_land_use_constraint(n: pypsa.Network) -> None:
    """
    Add global constraints for tech capacity limit.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance

    Returns
    -------
    pypsa.Network
        Network with added land use constraints
    """
    logger.info("Add land-use constraint for perfect foresight")

    def check_p_min_p_max(p_nom_max):
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}"
            )

    grouper = [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ext_i = n.generators.p_nom_extendable
    # get technical limit per node and investment period
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max
    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0
    # check minimum capacities
    check_p_min_p_max(p_nom_max)
    # adjust name to fit syntax of nominal constraint per bus
    df = p_nom_max.reset_index()
    df["name"] = df.apply(
        lambda row: (
            f"nom_max_{row['carrier']}"
            + (f"_{row['build_year']}" if row["build_year"] is not None else "")
        ),
        axis=1,
    )

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values


def add_solar_potential_constraints(n: pypsa.Network, config: dict) -> None:
    """
    Add constraint to make sure the sum capacity of all solar technologies (fixed, tracking, ets. ) is below the region potential.

    Example:
    ES1 0: total solar potential is 10 GW, meaning:
           solar potential : 10 GW
           solar-hsat potential : 8 GW (solar with single axis tracking is assumed to have higher land use)
    The constraint ensures that:
           solar_p_nom + solar_hsat_p_nom * 1.13 <= 10 GW
    """
    land_use_factors = {
        "solar-hsat": config["renewable"]["solar"]["capacity_per_sqkm"]
        / config["renewable"]["solar-hsat"]["capacity_per_sqkm"],
    }
    rename = {} if PYPSA_V1 else {"Generator-ext": "Generator"}

    solar_carriers = ["solar", "solar-hsat"]
    solar = n.generators[
        n.generators.carrier.isin(solar_carriers) & n.generators.p_nom_extendable
    ].index

    solar_today = n.generators[
        (n.generators.carrier == "solar") & (n.generators.p_nom_extendable)
    ].index
    solar_hsat = n.generators[(n.generators.carrier == "solar-hsat")].index

    if solar.empty:
        return

    land_use = pd.DataFrame(1, index=solar, columns=["land_use_factor"])
    for carrier, factor in land_use_factors.items():
        land_use = land_use.apply(
            lambda x: (x * factor) if carrier in x.name else x, axis=1
        )

    location = pd.Series(n.buses.index, index=n.buses.index)
    ggrouper = n.generators.loc[solar].bus
    rhs = (
        n.generators.loc[solar_today, "p_nom_max"]
        .groupby(n.generators.loc[solar_today].bus.map(location))
        .sum()
        - n.generators.loc[solar_hsat, "p_nom"]
        .groupby(n.generators.loc[solar_hsat].bus.map(location))
        .sum()
        * land_use_factors["solar-hsat"]
    ).clip(lower=0)

    lhs = (
        (n.model["Generator-p_nom"].rename(rename).loc[solar] * land_use.squeeze())
        .groupby(ggrouper)
        .sum()
    )

    logger.info("Adding solar potential constraint.")
    n.model.add_constraints(lhs <= rhs, name="solar_potential")


def algebra_process_emissions(n, index):
    pe = n.model["Link-p"].loc[:, index]*n.links.loc[index].efficiency*n.snapshot_weightings["generators"]
    pe_sum = pe.sum()

    return pe_sum

def algebra_generation_emissions(n, index):
    ge = n.model["Link-p"].loc[:, index]*n.links.loc[index].efficiency2*n.snapshot_weightings["generators"]
    ge_sum = ge.sum()

    return ge_sum

def algebra_carboncapture(n, index, sign = "negative"):
    # bus0 is the electricity bus
    # bus1 is the urban heating bus 
    # bus2 is the CO2 atmosphere
    # bus3 is the CO2 storage bus
    # sign convention: when power is being discharged from bus0, then p0 (or "p" in the constraint) is positive
    
    cc = n.model["Link-p"].loc[:, index]*n.links.loc[index].efficiency3*n.snapshot_weightings["generators"]
    
    cc_sum = cc.sum() if sign == "positive" else -cc.sum()

    return cc_sum

def algebra_imports(n, index, sign = "positive"):
    # bus0 is the co2 atmosphere
    # bus1 is the imported commodity  
    # sign convention: when power is being discharged from bus0, then p0 (or "p" in the constraint) is positive
    # no conversion efficiency, since it already is in units of CO2 emissions
    
    im = n.model["Link-p"].loc[:, index]*n.snapshot_weightings["generators"]
    
    im_sum = im.sum() if sign == "positive" else -im.sum()

    return im_sum

def algebra_bio_gas(n, index):

    bg = n.model["Link-p"].loc[:, index]*n.links.loc[index].efficiency3*n.snapshot_weightings["generators"]
    bg_sum = bg.sum()

    return bg_sum

def update_UK_gas_price(n):
    """
    Use specific gas price  

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    Returns
    -------
    pypsa.Network
        Modified PyPSA network with updated marginal costs
    """

    # Set UK gas price
    uk_gas_price = 24.5 # in Eur/MWh

    # UK gas fuel subsidy for existing gas power plants (CCGT)
    uk_gas_fuel_subsidy = 1.0 # fraction of gas price subsidised

    # Set marginal cost of UK gas turbines
    marginal_cost_CCGT = 2.6 # Eur/MWh

    # Set marginal cost of UK gas CCGT for pre-built plants
    links = n.links.copy()
    gas_CCGT = links.query("carrier == 'CCGT'")
    uk_gas_CCGT = gas_CCGT.loc[gas_CCGT.index.str.contains("GB")]
    uk_gas_CCGT_prebuilt = uk_gas_CCGT.loc[uk_gas_CCGT.build_year < 2025]
    links.loc[uk_gas_CCGT_prebuilt.index, "marginal_cost"] = marginal_cost_CCGT - uk_gas_fuel_subsidy*uk_gas_price # assuming gas fuel expenditures are covered by subsidies for existing gas power plants
    n.links = links

    # Set marginal cost of UK gas production and extraction
    generators = n.generators.copy()
    uk_generators = generators.loc[generators.index.str.contains("GB")]
    uk_gas_generators = uk_generators.loc[uk_generators.carrier == "gas"]   
    generators.loc[uk_gas_generators.index, "marginal_cost"] = uk_gas_price
    n.generators = generators 

    return n

def add_UK_fixed_electricity_generation_mix(n, base_year, investment_year):
    investment_year = int(investment_year)

    if investment_year > base_year:
        logger.info("Planning year greater than base year, skipping UK minimum capacity factor constraint.")
        return

    # Exogenous electricity load
    elec_loads = n.loads.loc[n.loads.carrier.str.contains("electricity")]
    uk_elec_loads = elec_loads.loc[elec_loads.index.str.contains("GB")]

    load = (n.loads_t.p_set[uk_elec_loads.query("carrier == 'electricity'").index].sum().sum() + 
            (uk_elec_loads.loc[uk_elec_loads.carrier != 'electricity'].p_set*len(n.snapshots)).sum()
            )

    # Additional electricity demand from sector-coupling
    links_wo_transmission = n.links.drop(n.links.query("carrier == 'DC'").index)
    electricity_buses = list(n.buses.query('carrier == "AC"').index) + list(
        n.buses.query('carrier == "low voltage"').index
    )
    boolean_elec_demand_via_links = [
        links_wo_transmission.bus0[i] in electricity_buses
        for i in range(len(links_wo_transmission.bus0))
    ]
    boolean_elec_demand_via_links_series = pd.Series(boolean_elec_demand_via_links)
    elec_demand_via_links = links_wo_transmission.iloc[
        boolean_elec_demand_via_links_series[boolean_elec_demand_via_links_series].index
    ]

    # Drop storage dischargers as they are not a part of the generation mix
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[elec_demand_via_links.index.str.contains("discharge")]
    )

    # Drop distribution links
    elec_demand_via_links = elec_demand_via_links.drop(
        elec_demand_via_links.index[
            elec_demand_via_links.index.str.contains("distribution")
        ]
    )

    # Access data for UK
    elec_demand_via_links = elec_demand_via_links.loc[elec_demand_via_links.index.str.contains("GB")]

    uk_electricity_demand = n.model.variables["Link-p"].loc[:, elec_demand_via_links.index].sum() + load

    # https://grid.iamkate.com/, https://www.energyoasis.org.uk/blog/uk-renewable-energy-mix-2024  
    carriers = {
                "nuclear": 20, #[10, 20],
                "wind": 35, # [30, 35],
                "solar": 10,# [4, 8],
                # "gas": 30, #[25, 35]
                # "urban central solid biomass CHP": 8,
                }

    uk_buses = n.buses.loc[n.buses.index.str.contains("GB")].query("carrier == 'AC'")
    for carrier, c_range in carriers.items():

        print(carrier)

        if carrier in ["wind", "solar"]:

            uk_generators_t = n.generators.index[n.generators.index.str.contains(carrier) & n.generators.index.str.contains("GB")]

            # drop solar thermal 
            if carrier == "solar":
                uk_generators_t = uk_generators_t.drop(
                    uk_generators_t[uk_generators_t.str.contains("thermal")]
                )

            lhs = n.model.variables["Generator-p"].loc[:, uk_generators_t].sum()

        else:
            uk_power_generation_links = n.links.loc[n.links.bus1.isin(uk_buses)]
            carriers = f"{carrier} | OCGT | CCGT" if carrier == "gas" else carrier
            uk_power_generation_links = uk_power_generation_links.loc[uk_power_generation_links.index.str.contains(carriers)] 

            lhs = n.model.variables["Link-p"].loc[:, uk_power_generation_links.index].sum()

        # n.model.add_constraints(
        #     lhs >= (uk_electricity_demand * c_range / 100)
        #     ,
        #     name="lower_generation_limit_" + carrier,
        #     )

        n.model.add_constraints(
            lhs <= (uk_electricity_demand * c_range / 100)
            ,
            name="upper_generation_limit_" + carrier,
            )

        # undefine lhs 
        del lhs

        logger.info(f"Added generation limits for {carrier} in UK: {c_range}% of total electricity generation")

def add_UK_brownfield_minimum_capacity_factors(
    n, capacity_factors, base_year, investment_year
):

    # For example:
    # CCGT: 0.5 # https://wattdirection.substack.com/p/uk-combined-cycle-gas-power-stations-748
    # nuclear: 0.7 # https://assets.publishing.service.gov.uk/media/5a75a748e5274a545822d2b9/Nuclear_Capacity_in_the_UK.pdf 

    investment_year = int(investment_year)

    T = len(n.snapshots)
    for tech, minimum_capacity_factor in capacity_factors.items():

        if not investment_year in minimum_capacity_factor.keys():
            logger.info(f"No minimum capacity factor specified for {tech} in year {investment_year}, skipping constraint.")
            continue
        
        logger.info("Adding UK brownfield minimum capacity factors.")
        
        minimum_capacity_factor_t_y = minimum_capacity_factor[investment_year]

        # Select UK links of that tech
        tech_uk = n.links[
            n.links.index.str.contains("GB") & (n.links.carrier == tech)
        ]

        # Prebuilt (brownfield) capacities only
        tech_uk_prebuilt = tech_uk[tech_uk.build_year < investment_year] if not (investment_year == base_year) else tech_uk[tech_uk.build_year <= investment_year] 

        # If no prebuilt capacity, skip constraint
        if tech_uk_prebuilt.empty:
            continue

        # Total prebuilt capacity (check for zero or NaN)
        tech_uk_cap = tech_uk_prebuilt.p_nom.sum()

        if (tech_uk_cap is None) or (tech_uk_cap == 0) or (pd.isna(tech_uk_cap)):
            # log about zero capacity
            logger.info(f"No pre-built {tech} capacity in UK, skipping constraint.")
            continue

        # Total production of those links over all snapshots
        tech_uk_prod = n.model["Link-p"].loc[:, tech_uk_prebuilt.index].sum()

        # Minimum energy requirement: CF * capacity * time
        min_energy = minimum_capacity_factor_t_y * tech_uk_cap * T

        lhs = -tech_uk_prod + min_energy

        # enforce: tech_uk_prod >= min_energy
        n.model.add_constraints(lhs <= 0, name=f"capacity_factor_{tech}")

        # log
        logger.info(f"Added minimum capacity factor constraint for UK {tech}: {minimum_capacity_factor}")

def add_UK_deployment_rate_limits(n, deployment_rate_limits, base_year, investment_year):
    investment_year = int(investment_year)

    if not investment_year > base_year:
        logger.info("Planning year equals base year where capacities are fixed, and, thus, deployment rate limits are not needed.")
        return

    uk_generators = n.generators.query("bus.str.contains('GB')")
    uk_links = n.links.loc[n.links.index.str.contains("GB")]
    for tech in deployment_rate_limits.keys():

        if type(deployment_rate_limits[tech]) not in [int, float]:
            return 

        elif tech in ["heat pump"]:
            # by 2025, UK has 250,000 heat pumps installed (https://www.edie.net/uk-passes-250000-heat-pump-milestone/)            
            uk_links_hp = uk_links.loc[uk_links.index.str.contains(tech)]   
            uk_links_hp_extend = uk_links_hp.query("p_nom_extendable == True")
            COP_avg = 3 # assumed average COP of heat pumps
            lhs = n.model["Link-p_nom"].loc[uk_links_hp_extend.index].sum() * COP_avg
            rhs = deployment_rate_limits[tech] 

        elif tech in ["transmission"]:
            # Transmission lines are divided into AC and DC. The expansion limits should be applied to the total transmission volume, which is the sum of AC and DC lines. 
            uk_transmission_links = n.links.query("(bus0.str.contains('GB') and bus1.str.contains('GB')) and carrier == 'DC' and not index.str.contains('rev')") # including only internal lines
            uk_transmission_lines = n.lines.query("bus0.str.contains('GB') and bus1.str.contains('GB')") # including only internal lines

            # Today's total transmission volume in the UK (in MW-km)
            dc_links_vol_today = (n.links.loc[uk_transmission_links.index].p_nom_min * uk_transmission_links.length).sum()
            ac_lines_vol_today = (n.lines.loc[uk_transmission_lines.index].s_nom_min * uk_transmission_lines.length).sum()
            transmission_vol_today = dc_links_vol_today + ac_lines_vol_today

            # Extendable subsets
            ext_dc = uk_transmission_links[uk_transmission_links.p_nom_extendable]
            ext_ac = uk_transmission_lines[uk_transmission_lines.s_nom_extendable]

            # Variables
            dc_var = n.model["Link-p_nom"].loc[ext_dc.index]
            ac_var = n.model["Line-s_nom"].loc[ext_ac.index]

            # Existing lines/links capacity
            dc_existing = ext_dc.p_nom_min
            ac_existing = ext_ac.s_nom_min

            # Existing lines/links length
            dc_length = ext_dc.length
            ac_length = ext_ac.length

            # Build linear expressions
            dc_expansion = ((dc_var - dc_existing) * dc_length).sum()
            ac_expansion = ((ac_var - ac_existing) * ac_length).sum()

            # The expansion limit is defined as a fraction of today's total transmission volume in the UK. 
            expansion_fraction = deployment_rate_limits[tech] # fraction of today's transmission volume that can be added per 5-year period
            lhs = dc_expansion + ac_expansion
            rhs = expansion_fraction * transmission_vol_today

        else:
            if tech in ["offwind"]:
                uk_generators_vre = uk_generators.loc[uk_generators.index.str.contains(tech)]   
            else:
                uk_generators_vre = uk_generators.query("carrier == @tech")
    
            uk_generators_vre_extend = uk_generators_vre.query("p_nom_extendable == True")

            if uk_generators_vre_extend.empty:
                logger.info(f"No extendable {tech} capacity in UK, skipping deployment rate limit constraint.")
                continue

            lhs = n.model["Generator-p_nom"].loc[uk_generators_vre_extend.index].sum()
            rhs = deployment_rate_limits[tech] 
        
        n.model.add_constraints(lhs <= rhs, name=f"deployment_rate_limit_{tech}_{investment_year}")
        logger.info(f"Added deployment rate limit for UK {tech}: {rhs} MW / 5-year period")

def add_split_co2_constraints(n: pypsa.Network, local_co2: dict) -> None:
    """
    Add local CO2 caps and a collective cap for the remaining countries.

    Regions with no baseline data or no limit for the current horizon remain
    included in the collective budget.
    """
    co2_totals_file = snakemake.input.co2_totals
    co2_totals = 1e6 * pd.read_csv(co2_totals_file, index_col=0) # convert Mt to tCO2

    options = snakemake.params.sector
    co2_budget = snakemake.params.co2_budget
    sectors = determine_emission_sectors(options)
    nhours = n.snapshot_weightings.generators.sum()
    nyears = nhours / 8760

    investment_year = int(snakemake.wildcards.horizon)
    all_countries = snakemake.params.countries
    sectors = co2_totals.columns.intersection(sectors)
    local_co2_regions = valid_local_co2_regions(
        local_co2, co2_totals, investment_year, all_countries
    )

    expr = n.optimize.expressions.energy_balance(
                                                bus_carrier="co2",
                                                groupby=False
                                                )

    # add local CO2 emissions constraints
    mask_merged = None
    for region in local_co2_regions:
        # calculating the net CO2 emissions allowance for the country
        limit = local_co2[region][investment_year]

        co2_1990 = co2_totals.loc[region, sectors].sum() # tCO2 emissions per year
        net_co2_allowance = co2_1990 * limit * nyears
        logger.info(f"Net CO2 emissions cap for {region}: {net_co2_allowance}")

        # Add expression containing all CO2 sources and sinks attached to the country
        mask = expr.coords["Link"].str.contains(re.escape(region))
        expr_c_group = expr.sel(Link=mask, group = 0) # group 0 corresponds to links

        lhs = expr_c_group.sum()
        rhs = net_co2_allowance

        n.model.add_constraints(
            lhs <= rhs,
            name="local co2 emissions constraint " + region ,
        )

        mask_merged = mask if mask_merged is None else mask_merged | mask

    # Add global co2 emissions constraint
    collective = [x for x in all_countries if x not in local_co2_regions]
    collective = [x for x in collective if x in co2_totals.index]
    if not collective:
        return

    co2_1990 = co2_totals.loc[collective, sectors].sum().sum()
    total_co2_1990 = co2_totals.loc[
        co2_totals.index.intersection(all_countries), sectors
    ].sum().sum()
    annual_allowance = calculate_co2_limit(
        investment_year, co2_budget, co2_1990, total_co2_1990
    )
    if annual_allowance is None:
        logger.info("No collective CO2 budget configured for %s.", investment_year)
        return
    net_co2_allowance = annual_allowance * nyears
    logger.info(f"Collective CO2 emissions budget for {len(collective)} countries: {net_co2_allowance}")

    if mask_merged is not None:
        expr_collective_group = expr.sel(Link=~mask_merged, group = 0)
    else:
        expr_collective_group = expr.sel(group = 0)

    lhs = expr_collective_group.sum()
    rhs = net_co2_allowance

    n.model.add_constraints(
        lhs <= rhs,
        name="collective co2 emissions constraint",
    )

def add_co2_sequestration_limit(
    n: pypsa.Network,
    limit_dict: dict[str, float],
    planning_horizons: str | None,
) -> None:
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    limit_dict : dict[str, float]
        CO2 sequestration potential limit constraints by year.
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight
    """

    if not n.investment_periods.empty:
        nyears = n.snapshot_weightings.groupby(level="period").generators.sum() / 8760
        periods = n.investment_periods
        limit = pd.Series(
            {period: nyears[period] * get(limit_dict, period) for period in periods}
        )
        limit.index = limit.index.map(lambda s: f"co2_sequestration_limit-{s}")
        names = limit.index
    else:
        nyears = n.snapshot_weightings.generators.sum() / 8760
        limit = get(limit_dict, int(planning_horizons)) * nyears
        periods = np.nan
        names = "co2_sequestration_limit"

    n.add(
        "GlobalConstraint",
        names,
        sense=">=",
        constant=-limit * 1e6,
        type="operational_limit",
        carrier_attribute="co2 sequestered",
        investment_period=periods,
    )


VALID_GLC_SENSES = {"<=", ">=", "=="}


def get_glc_sense(glc: pd.Series) -> str:
    """Return the validated sense of a global constraint."""
    sense = glc.loc["sense"]
    if sense not in VALID_GLC_SENSES:
        raise ValueError(
            f"Unsupported sense {sense!r} for global constraint {glc.name!r}. "
            f"Expected one of {sorted(VALID_GLC_SENSES)}."
        )
    return sense


def emission_stores(n: pypsa.Network, emissions: pd.Index) -> pd.DataFrame:
    """Return the non-cyclic stores that accumulate emissions on their bus."""
    bus_carrier = n.stores.bus.map(n.buses.carrier)
    stores = n.stores[bus_carrier.isin(emissions) & ~n.stores.e_cyclic]
    seeded = stores.index[stores.e_initial.ne(0) | stores.e_initial_per_period]
    if not seeded.empty:
        raise ValueError(
            f"Emission stores {list(seeded)} start from a non-zero level "
            "(e_initial or e_initial_per_period). The CO2 constraints read the "
            "store level as cumulative emissions, which only holds from zero."
        )
    return stores


def add_carbon_constraint(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
    glcs = n.global_constraints.query('type == "co2_atmosphere"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        stores = emission_stores(n, emissions.index)
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            lhs = final_e.loc[time_i, :] - final_e.shift(snapshot=1).loc[time_i, :]

            rhs = glc.constant
            n.model.add_constraints(
                lhs, get_glc_sense(glc), rhs, name=f"GlobalConstraint-{name}"
            )


def add_carbon_budget_constraint(n: pypsa.Network, snapshots: pd.DatetimeIndex) -> None:
    glcs = n.global_constraints.query('type == "Co2Budget"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        stores = emission_stores(n, emissions.index)
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            weighting = n.investment_period_weightings.loc[time_valid, "years"]
            lhs = final_e.loc[time_i, :] * weighting

            rhs = glc.constant
            n.model.add_constraints(
                lhs, get_glc_sense(glc), rhs, name=f"GlobalConstraint-{name}"
            )


def add_max_growth(n: pypsa.Network, opts: dict) -> None:
    """
    Add maximum growth rates for different carriers.
    """

    # take maximum yearly difference between investment periods since historic growth is per year
    factor = n.investment_period_weightings.years.max() * opts["factor"]
    for carrier in opts["max_growth"].keys():
        max_per_period = opts["max_growth"][carrier] * factor
        logger.info(
            f"set maximum growth rate per investment period of {carrier} to {max_per_period} GW."
        )
        n.carriers.loc[carrier, "max_growth"] = max_per_period * 1e3

    for carrier in opts["max_relative_growth"].keys():
        max_r_per_period = opts["max_relative_growth"][carrier]
        logger.info(
            f"set maximum relative growth per investment period of {carrier} to {max_r_per_period}."
        )
        n.carriers.loc[carrier, "max_relative_growth"] = max_r_per_period


def add_retrofit_gas_boiler_constraint(
    n: pypsa.Network, snapshots: pd.DatetimeIndex
) -> None:
    """
    Allow retrofitting of existing gas boilers to H2 boilers and impose load-following must-run condition on existing gas boilers.
    Modifies the network in place, no return value.

    n : pypsa.Network
        The PyPSA network to be modified
    snapshots : pd.DatetimeIndex
        The snapshots of the network
    """
    c = "Link"
    # existing gas boilers
    mask = n.links.carrier.str.contains("gas boiler") & ~n.links.p_nom_extendable
    gas_i = n.links[mask].index
    mask = n.links.carrier.str.contains("retrofitted H2 boiler")
    h2_i = n.links[mask].index

    if gas_i.empty or h2_i.empty:
        return

    logger.info("Add constraint for retrofitting gas boilers to H2 boilers.")

    n.links.loc[gas_i, "p_nom_extendable"] = True
    p_nom = n.links.loc[gas_i, "p_nom"]
    n.links.loc[gas_i, "p_nom"] = 0

    # heat profile
    cols = n.loads_t.p_set.columns[
        n.loads_t.p_set.columns.str.contains("heat")
        & ~n.loads_t.p_set.columns.str.contains("industry")
        & ~n.loads_t.p_set.columns.str.contains("agriculture")
    ]
    profile = n.loads_t.p_set[cols].div(
        n.loads_t.p_set[cols].groupby(level=0).max(), level=0
    )
    # to deal if max value is zero
    profile.fillna(0, inplace=True)
    profile.rename(columns=n.loads.bus.to_dict(), inplace=True)
    profile = profile.reindex(columns=n.links.loc[gas_i, "bus1"])
    profile.columns = gas_i

    rhs = profile.mul(p_nom)

    dispatch = n.model["Link-p"]
    active = get_activity_mask(n, c, snapshots, gas_i)
    rhs = rhs[active]
    if PYPSA_V1:
        p_gas = dispatch.sel(name=gas_i)
        p_h2 = dispatch.sel(name=h2_i)
    else:
        p_gas = dispatch.sel(Link=gas_i)
        p_h2 = dispatch.sel(Link=h2_i)

    lhs = p_gas + p_h2

    n.model.add_constraints(lhs == rhs, name="gas_retrofit")


def enforce_autarky(n: pypsa.Network, only_crossborder: bool = False) -> None:
    """
    Remove transmission lines to enforce autarky.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    only_crossborder : bool
        If True, only remove cross-border connections. If False, remove all lines.
    """
    if only_crossborder:
        lines_rm = n.lines.loc[
            n.lines.bus0.map(n.buses.country) != n.lines.bus1.map(n.buses.country)
        ].index
        links_rm = n.links.loc[
            n.links.bus0.map(n.buses.country) != n.links.bus1.map(n.buses.country)
        ].index
    else:
        lines_rm = n.lines.index
        links_rm = n.links.loc[n.links.carrier == "DC"].index

    logger.info(
        f"Enforcing autarky: removing {len(lines_rm)} lines and {len(links_rm)} links"
    )
    n.remove("Line", lines_rm)
    n.remove("Link", links_rm)


def add_load_balance_components(n, config, sign=1):
    """
    Add load shedding or load sinks to the network with carrier 'load'.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to be modified.
    config : dict
        The load shedding or load sinks settings.
    sign : float
        Direction of the added generators. Positive for load shedding, negative for load sinks.

    Returns
    -------
    None
        Modifies PyPSA network in place.
    """
    if "load" not in n.carriers.index:
        n.add("Carrier", "load")

    carriers = config.get("carriers", {})
    default_cost = config.get("default_cost")
    balance_comp = "shedding" if sign > 0 else "sink"

    logger.info(
        f"Add load {balance_comp} for {'all carriers' if config.get('all_carriers') else ', '.join(carriers)}."
    )

    for bus_carrier, price in carriers.items():
        buses_i = n.buses[n.buses.carrier == bus_carrier].index
        n.add(
            "Generator",
            buses_i,
            f" load {balance_comp}",
            bus=buses_i,
            carrier="load",
            marginal_cost=price,
            p_nom=np.inf,
            sign=sign,
        )

    if config.get("all_carriers", False):
        buses_rest_i = n.buses[~n.buses.carrier.isin(carriers)].index
        n.add(
            "Generator",
            buses_rest_i,
            f" load {balance_comp}",
            bus=buses_rest_i,
            carrier="load",
            marginal_cost=default_cost,
            p_nom=np.inf,
            sign=sign,
        )


def prepare_network(
    n: pypsa.Network,
    config: dict,
    solve_opts: dict,
    foresight: str,
    planning_horizons: str | None,
    co2_sequestration_potential: dict[str, float],
    limit_max_growth: dict[str, Any] | None = None,
    rolling_horizon: bool = False,
) -> None:
    """
    Prepare network with various constraints and modifications.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    config : dict
        Configuration dictionary
    solve_opts : Dict
        Dictionary of solving options containing clip_p_max_pu, load_shedding etc.
    foresight : str
        Planning foresight type ('myopic' or 'perfect')
    planning_horizons : str or None
        The current planning horizon year or None for perfect foresight
    co2_sequestration_potential : Dict[str, float]
        CO2 sequestration potential constraints by year

    Returns
    -------
    pypsa.Network
        Modified PyPSA network with added constraints
    """
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.links_t.p_max_pu,
            n.links_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df.abs() > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if (load_shedding := solve_opts.get("load_shedding", {})).get("enable", False):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        add_load_balance_components(n, load_shedding)

    if (load_sinks := solve_opts.get("load_sinks", {})).get("enable", False):
        add_load_balance_components(n, load_sinks, sign=-1)

    if solve_opts.get("curtailment_mode"):
        n.add("Carrier", "curtailment", color="#fedfed", nice_name="Curtailment")
        n.generators_t.p_min_pu = n.generators_t.p_max_pu
        buses_i = n.buses.query("carrier == 'AC'").index
        n.add(
            "Generator",
            buses_i,
            suffix=" curtailment",
            bus=buses_i,
            p_min_pu=-1,
            p_max_pu=0,
            marginal_cost=-0.1,
            carrier="curtailment",
            p_nom=1e6,
        )

    if solve_opts.get("noisy_costs"):
        for t in n.components:
            # if 'capital_cost' in t.static:
            #    t.static['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.static)) - 0.5)
            if "marginal_cost" in t.static:
                t.static["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.static)) - 0.5
                )

        for t in n.components[["Line", "Link"]]:
            if t.static.empty:
                continue
            t.static["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.static)) - 0.5)
            ) * t.static["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if foresight == "perfect":
        add_land_use_constraint(n)
        if limit_max_growth is not None and limit_max_growth["enable"]:
            add_max_growth(n, limit_max_growth)

    if n.stores.carrier.eq("co2 sequestered").any():
        limit_dict = co2_sequestration_potential
        add_co2_sequestration_limit(
            n, limit_dict=limit_dict, planning_horizons=planning_horizons
        )

    # Apply autarky constraint if configured
    autarky_cfg = config["electricity"]["autarky"]
    if autarky_cfg["enable"]:
        only_crossborder = autarky_cfg["by_country"]
        enforce_autarky(n, only_crossborder=only_crossborder)

    # rolling horizon disables cyclic storage
    if rolling_horizon:
        n.storage_units.cyclic_state_of_charge = False
        n.storage_units.state_of_charge_initial = 0
        n.stores.e_cyclic = False
        n.stores.e_initial = 0


def add_CCL_constraints(
    n: pypsa.Network, config: dict, planning_horizons: str | None
) -> None:
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    config : dict
        Configuration dictionary
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24h]
    electricity:
        agg_p_nom_limits: data/agg_p_nom_minmax.csv
    """

    assert planning_horizons is not None, (
        "add_CCL_constraints are not implemented for perfect foresight, yet"
    )

    agg_p_nom_minmax = pd.read_csv(
        config["solving"]["agg_p_nom_limits"]["file"], index_col=[0, 1], header=[0, 1]
    )[planning_horizons]
    logger.info("Adding generation capacity constraints per carrier and country")
    p_nom = n.model["Generator-p_nom"]

    gens = n.generators.query("p_nom_extendable")

    if not PYPSA_V1:
        gens = gens.rename_axis(index="Generator-ext")

    if config["solving"]["agg_p_nom_limits"]["agg_offwind"]:
        rename_offwind = {
            "offwind-ac": "offwind-all",
            "offwind-dc": "offwind-all",
            "offwind-float": "offwind-all",
            "offwind": "offwind-all",
        }
        gens = gens.replace(rename_offwind)
    if config["solving"]["agg_p_nom_limits"]["agg_solar"]:
        rename_solar = {
            "solar": "solar-all",
            "solar-hsat": "solar-all",
            "solar rooftop": "solar-all",
        }
        gens = gens.replace(rename_solar)
    grouper = pd.concat([gens.bus.map(n.buses.country), gens.carrier], axis=1)
    lhs = p_nom.groupby(grouper).sum().rename(bus="country")

    if config["solving"]["agg_p_nom_limits"]["include_existing"]:
        gens_cst = n.generators.query("~p_nom_extendable").rename_axis(
            index="Generator-cst"
        )
        gens_cst = gens_cst[
            (gens_cst["build_year"] + gens_cst["lifetime"]) >= int(planning_horizons)
        ]
        if config["solving"]["agg_p_nom_limits"]["agg_offwind"]:
            gens_cst = gens_cst.replace(rename_offwind)
        if config["solving"]["agg_p_nom_limits"]["agg_solar"]:
            gens_cst = gens_cst.replace(rename_solar)
        rhs_cst = (
            pd.concat(
                [gens_cst.bus.map(n.buses.country), gens_cst[["carrier", "p_nom"]]],
                axis=1,
            )
            .groupby(["bus", "carrier"])
            .sum()
        )
        rhs_cst.index = rhs_cst.index.rename({"bus": "country"})
        rhs_min = agg_p_nom_minmax["min"].dropna()
        idx_min = rhs_min.index.join(rhs_cst.index, how="left")
        rhs_min = rhs_min.reindex(idx_min).fillna(0)
        rhs = (rhs_min - rhs_cst.reindex(idx_min).fillna(0).p_nom).dropna()
        rhs[rhs < 0] = 0
        minimum = xr.DataArray(rhs).rename(dim_0="group")
    else:
        minimum = xr.DataArray(agg_p_nom_minmax["min"].dropna()).rename(dim_0="group")

    index = minimum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) >= minimum.loc[index], name="agg_p_nom_min"
        )

    if config["solving"]["agg_p_nom_limits"]["include_existing"]:
        rhs_max = agg_p_nom_minmax["max"].dropna()
        idx_max = rhs_max.index.join(rhs_cst.index, how="left")
        rhs_max = rhs_max.reindex(idx_max).fillna(0)
        rhs = (rhs_max - rhs_cst.reindex(idx_max).fillna(0).p_nom).dropna()
        rhs[rhs < 0] = 0
        maximum = xr.DataArray(rhs).rename(dim_0="group")
    else:
        maximum = xr.DataArray(agg_p_nom_minmax["max"].dropna()).rename(dim_0="group")

    index = maximum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) <= maximum.loc[index], name="agg_p_nom_max"
        )


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24h]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    # TODO: Generalize to cover myopic and other sectors?
    float_regex = r"[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (
        (p * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling))
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n: pypsa.Network, config: dict) -> None:
    """
    Add business-as-usual (BAU) constraints for minimum capacities.

    Parameters
    ----------
    n : pypsa.Network
        PyPSA network instance
    config : dict
        Configuration dictionary containing BAU minimum capacities
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier)
    if not PYPSA_V1:
        ext_carrier_i = ext_carrier_i.rename_axis("Generator-ext")
    lhs = p_nom.groupby(ext_carrier_i).sum()
    rhs = mincaps[lhs.indexes["carrier"]].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance.
    config : dict
        Configuration dictionary.

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24h]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]  # noqa: F841
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX.jl/stable/Model_Reference/core/#Operational-Reserves.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance.
    sns : pd.DatetimeIndex
        Snapshots for the simulation.
    config : dict
        Configuration dictionary.

    Example
    -------
    config.yaml requires to specify operational_reserve:
    operational_reserve:
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 4000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    vres_carriers = [  # noqa: F841
        "ror" if c == "hydro" else c
        for c in config["electricity"]["renewable_carriers"]
    ]
    generator_dim = "Generator" if not PYPSA_V1 else "name"

    gen_i = n.generators.query("carrier != 'load'").index  # exclude load shedding
    fix_i = n.generators.loc[gen_i].query("not p_nom_extendable").index
    ext_i = n.generators.loc[gen_i].query("p_nom_extendable").index
    vres_i = n.generators.loc[gen_i].query("carrier in @vres_carriers").index

    # Reserve Variables
    n.model.add_variables(0, np.inf, coords=[sns, gen_i], name="Generator-r")
    reserve = n.model["Generator-r"]
    lhs = reserve.sum(generator_dim)

    # Share of extendable renewable capacities
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = n.model["Generator-p_nom"].loc[vres_i.intersection(ext_i)]
        if not PYPSA_V1:
            p_nom_vres = p_nom_vres.rename({"Generator-ext": "Generator"})
        lhs += (p_nom_vres * (-EPSILON_VRES * xr.DataArray(capacity_factor))).sum(
            generator_dim
        )

    # Total demand per t
    demand = n.get_switchable_as_dense("Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded

    dispatch = n.model["Generator-p"].sel(name=gen_i)
    reserve = n.model["Generator-r"]
    lhs = dispatch + reserve

    p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu")

    if not ext_i.empty:
        capacity_variable = n.model["Generator-p_nom"].sel(name=ext_i)
        if not PYPSA_V1:
            capacity_variable = capacity_variable.rename({"Generator-ext": "Generator"})
        lhs -= capacity_variable * xr.DataArray(p_max_pu[ext_i])

    capacity_fixed = n.generators.p_nom[fix_i]
    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def add_TES_energy_to_power_ratio_constraints(n: pypsa.Network) -> None:
    """
    Add TES constraints to the network.

    For each TES storage unit, enforce:
        Store-e_nom - etpr * Link-p_nom == 0

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA network with TES and heating sectors enabled.

    Raises
    ------
    ValueError
        If no valid TES storage or charger links are found.
    RuntimeError
        If the TES storage and charger indices do not align.
    """
    indices_charger_p_nom_extendable = n.links.index[
        n.links.index.str.contains("water tanks charger|water pits charger")
        & n.links.p_nom_extendable
    ]

    indices_stores_e_nom_extendable = n.stores.index[
            n.stores.index.str.contains("water tanks|water pits")
            & n.stores.e_nom_extendable
        ]

    if indices_charger_p_nom_extendable.empty or indices_stores_e_nom_extendable.empty:
        return 

    # Ensure indices of chargers and stores match
    indices_charger_p_nom_extendable_df = pd.DataFrame(indices_charger_p_nom_extendable)
    indices_charger_p_nom_extendable_df["index_reduced"] = indices_charger_p_nom_extendable.str.split(" charger", expand = True).get_level_values(0)
    indices_stores_e_nom_extendable_df = pd.DataFrame(indices_stores_e_nom_extendable)
    indices_stores_e_nom_extendable_df["index_reduced"] = indices_stores_e_nom_extendable.str.split("-", expand = True).get_level_values(0)

    indices_stores_e_nom_extendable_df = indices_stores_e_nom_extendable_df.reset_index().set_index("index_reduced")
    indices_charger_p_nom_extendable_df = indices_charger_p_nom_extendable_df.reset_index().set_index("index_reduced")

    index_in_common = indices_charger_p_nom_extendable_df.index.intersection(indices_stores_e_nom_extendable_df.index)

    indices_charger_p_nom_extendable = pd.Index(indices_charger_p_nom_extendable_df.loc[index_in_common]["Link"].values)
    indices_stores_e_nom_extendable = pd.Index(indices_stores_e_nom_extendable_df.loc[index_in_common]["Store"].values)

    if indices_charger_p_nom_extendable.empty or indices_stores_e_nom_extendable.empty:
        logger.debug(
            "No valid extendable charger links or stores found for TES energy-to-power constraints.Not enforcing TES energy-to-power ratio constraints!"
        )
        return

    energy_to_power_ratio_values = n.links.loc[
        indices_charger_p_nom_extendable, "energy to power ratio"
    ].values

    linear_expr_list = []
    for charger, tes, energy_to_power_value in zip(
        indices_charger_p_nom_extendable,
        indices_stores_e_nom_extendable,
        energy_to_power_ratio_values,
    ):
        charger_var = n.model["Link-p_nom"].loc[charger]
        if not tes == charger.replace(" charger", ""):
            # e.g. "DE0 0 urban central water tanks charger-2050" -> "DE0 0 urban central water tanks-2050"
            raise RuntimeError(
                f"Charger {charger} and TES {tes} do not match. "
                "Ensure that the charger and TES are in the same location and refer to the same technology."
            )
        store_var = n.model["Store-e_nom"].loc[tes]
        linear_expr = store_var - energy_to_power_value * charger_var
        linear_expr_list.append(linear_expr)

    # Merge the individual expressions
    dim = "Store-ext, Link-ext" if PYPSA_V1 else "name"
    merged_expr = linopy.expressions.merge(
        linear_expr_list, dim=dim, cls=type(linear_expr_list[0])
    )

    n.model.add_constraints(merged_expr == 0, name="TES_energy_to_power_ratio")


def add_TES_charger_ratio_constraints(n: pypsa.Network) -> None:
    """
    Add TES charger ratio constraints.

    For each TES unit, enforce:
        Link-p_nom(charger) - efficiency * Link-p_nom(discharger) == 0

    Parameters
    ----------
    n : pypsa.Network
        A PyPSA network with TES and heating sectors enabled.

    Raises
    ------
    ValueError
        If no valid TES discharger or charger links are found.
    RuntimeError
        If the charger and discharger indices do not align.
    """
    indices_charger_p_nom_extendable = n.links.index[
        n.links.index.str.contains(
            "water tanks charger|water pits charger|aquifer thermal energy storage charger"
        )
        & n.links.p_nom_extendable
    ]
    indices_discharger_p_nom_extendable = n.links.index[
        n.links.index.str.contains(
            "water tanks discharger|water pits discharger|aquifer thermal energy storage discharger"
        )
        & n.links.p_nom_extendable
    ]

    if (
        indices_charger_p_nom_extendable.empty
        or indices_discharger_p_nom_extendable.empty
    ):
        logger.debug(
            "No valid extendable TES discharger or charger links found for TES charger ratio constraints. Not enforcing TES charger_ratio constraints."
        )
        return

    for charger, discharger in zip(
        indices_charger_p_nom_extendable, indices_discharger_p_nom_extendable
    ):
        if not charger.replace(" charger", " ") == discharger.replace(
            " discharger", " "
        ):
            # e.g. "DE0 0 urban central water tanks charger-2050" -> "DE0 0 urban central water tanks-2050"
            raise RuntimeError(
                f"Charger {charger} and discharger {discharger} do not match. "
                "Ensure that the charger and discharger are in the same location and refer to the same technology."
            )

    eff_discharger = n.links.efficiency[indices_discharger_p_nom_extendable].values
    lhs = (
        n.model["Link-p_nom"].loc[indices_charger_p_nom_extendable]
        - n.model["Link-p_nom"].loc[indices_discharger_p_nom_extendable]
        * eff_discharger
    )

    n.model.add_constraints(lhs == 0, name="TES_charger_ratio")


def add_battery_constraints(n, battery_techs):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("|".join([f"{tech} discharger" for tech in battery_techs]))
    charger_bool = n.links.index.str.contains("|".join([f"{tech} charger" for tech in battery_techs]))

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_lossy_bidirectional_link_constraints(n):
    if not n.links.p_nom_extendable.any() or not any(n.links.get("reversed", [])):
        return

    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841
    backwards = n.links.query(
        "carrier in @carriers and p_nom_extendable and reversed and active"
    ).index
    forwards = backwards.str.replace("-reversed", "")
    lhs = n.model["Link-p_nom"].loc[backwards]
    rhs = n.model["Link-p_nom"].loc[forwards]
    n.model.add_constraints(lhs == rhs, name="Link-bidirectional_sync")


def add_chp_constraints(n):
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {} if PYPSA_V1 else {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    if "reversed" not in n.links.columns:
        n.links["reversed"] = False
    gas_pipes_i = n.links.query(
        "carrier == 'gas pipeline' and p_nom_extendable and ~reversed and active"
    ).index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable and ~reversed and active"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    config = n.config
    CH4_per_H2 = 1 / config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i]
    if not PYPSA_V1:
        rhs = rhs.rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")


def add_flexible_egs_constraint(n):
    """
    Upper bounds the charging capacity of the geothermal reservoir according to
    the well capacity.
    """
    well_index = n.links.loc[n.links.carrier == "geothermal heat"].index
    storage_index = n.storage_units.loc[
        n.storage_units.carrier == "geothermal heat"
    ].index

    p_nom_rhs = n.model["Link-p_nom"].loc[well_index]
    p_nom_lhs = n.model["StorageUnit-p_nom"].loc[storage_index]

    n.model.add_constraints(
        p_nom_lhs <= p_nom_rhs,
        name="upper_bound_charging_capacity_of_geothermal_reservoir",
    )


def add_import_limit_constraint(
    n: pypsa.Network,
    sns: pd.DatetimeIndex,
    type: str,
    limit_sense: str = "<=",
    limit: float | dict | bool | None = None,
) -> None:
    """
    Add constraint for limiting green energy imports (synthetic and biomass).
    Does not include fossil fuel imports.
    """

    nyears = n.snapshot_weightings.generators.sum() / 8760
    weightings = n.snapshot_weightings.loc[sns, "generators"]

    import_links = n.links.loc[n.links.carrier.str.contains("import")].index
    import_gens = n.generators.loc[n.generators.carrier.str.contains("import")].index

    if import_links.empty and import_gens.empty:
        return

    senses = {"<=": "<=", ">=": ">=", "==": "=", "=": "="}
    if limit_sense not in senses:
        raise ValueError(f"Unsupported import limit sense: {limit_sense}")

    if isinstance(limit, dict):
        limits = limit.items()
    elif isinstance(limit, (int, float, np.number)) and not isinstance(
        limit, (bool, np.bool_)
    ):
        limits = [(None, limit)]
    else:
        return

    for country, value in limits:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.number))
            or not np.isfinite(value)
        ):
            continue

        if country is None:
            import_gens_c = import_gens
            import_links_c = import_links
        else:
            country_pattern = re.escape(str(country))
            import_gens_c = (
                import_gens[
                    import_gens.str.contains(country_pattern, regex=True)
                ]
                if not import_gens.empty
                else import_gens
            )
            import_links_c = (
                import_links[
                    import_links.str.contains(country_pattern, regex=True)
                ]
                if not import_links.empty
                else import_links
            )
        if import_links_c.empty and import_gens_c.empty:
            continue

        # everything needs to be in MWh_fuel
        eff = n.links.loc[import_links_c, "efficiency"]

        if import_gens_c.empty:
            lhs = 0
        else:
            p_gens = n.model["Generator-p"].loc[sns, import_gens_c]
            lhs = (p_gens * weightings).sum()
        if not import_links_c.empty:
            p_links = n.model["Link-p"].loc[sns, import_links_c]
            lhs += (p_links * eff * weightings).sum()
        rhs = value * 1e6 * nyears
        suffix = f"_{country}" if country is not None else ""
        name = f"{type}_import_limit{suffix}"
        n.model.add_constraints(lhs, senses[limit_sense], rhs, name=name)
        logger.info(
            "Added %s import limit constraint%s with %s %s TWh_fuel/year",
            type,
            f" for {country}" if country is not None else "",
            limit_sense,
            value,
        )


def add_co2_atmosphere_constraint(n, snapshots):
    logger.info("Adding global CO2 constraint.")
    glcs = n.global_constraints[n.global_constraints.type == "co2_atmosphere"]

    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        stores = emission_stores(n, emissions.index)
        if not stores.empty:
            last_i = snapshots[-1]
            lhs = n.model["Store-e"].loc[last_i, stores.index]
            rhs = glc.constant

            n.model.add_constraints(
                lhs, get_glc_sense(glc), rhs, name=f"GlobalConstraint-{name}"
            )


def extra_functionality(
    n: pypsa.Network,
    snapshots: pd.DatetimeIndex,
    planning_horizons: str | None = None,
    snakemake=None,
) -> None:
    """
    Add custom constraints and functionality.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance with config and params attributes
    snapshots : pd.DatetimeIndex
        Simulation timesteps
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight.

    Notes
    -----
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    config = n.config
    constraints = config["solving"].get("constraints", {})
    if constraints["BAU"] and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if constraints["SAFE"] and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if constraints["CCL"] and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config, planning_horizons)

    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)

    if EQ_o := constraints["EQ"]:
        add_EQ_constraints(n, EQ_o.replace("EQ", ""))

    if {"solar-hsat", "solar"}.issubset(
        config["electricity"]["renewable_carriers"]
    ) and {"solar-hsat", "solar"}.issubset(
        config["electricity"]["extendable_carriers"]["Generator"]
    ):
        add_solar_potential_constraints(n, config)

    add_TES_energy_to_power_ratio_constraints(n)
    add_TES_charger_ratio_constraints(n)
    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)
    add_pipe_retrofit_constraint(n)
    if n._multi_invest:
        # add_carbon_constraint(n, snapshots)
        # add_carbon_budget_constraint(n, snapshots)
        add_retrofit_gas_boiler_constraint(n, snapshots)
    elif not isinstance(config["local_co2"], dict):
        add_co2_atmosphere_constraint(n, snapshots)

    if config["sector"]["enhanced_geothermal"]["enable"]:
        add_flexible_egs_constraint(n)

    sector_options = config.get("sector", {})
    import_options = {
        "green": sector_options.get(
            "green_imports", sector_options.get("imports", {})
        ),
        "fossil": sector_options.get("fossil_imports", {}),
        "other": sector_options.get("other_imports", {}),
    }
    for import_type, options in import_options.items():
        if options.get("enable", False):
            add_import_limit_constraint(
                n,
                snapshots,
                import_type,
                options.get("limit_sense", "<="),
                options.get("limit"),
            )

    base_year = snakemake.params.planning_horizons[0]
    investment_year = int(snakemake.wildcards.horizon)
    uk_settings = snakemake.params.uk_settings

    if uk_settings["uk_max_electric_heating_share"]:
        logger.info("Adding UK maximum electricity share of heating.")
        add_heating_constraint_uk(n, base_year, uk_settings["uk_max_electric_heating_share"])

    if uk_settings["uk_fixed_electricity_generation_mix"]:
        logger.info("Adding UK fixed electricity generation mix.")
        add_UK_fixed_electricity_generation_mix(n, base_year, investment_year)
    
    if isinstance(uk_settings["uk_brownfield_minimum_capacity_factors"], dict):
        capacity_factors = uk_settings["uk_brownfield_minimum_capacity_factors"]
        add_UK_brownfield_minimum_capacity_factors(
            n, capacity_factors, base_year, investment_year
        )

    if isinstance(uk_settings["uk_deployment_rate_limits"], dict):
        logger.info("Adding UK deployment rates.")
        add_UK_deployment_rate_limits(
            n, uk_settings["uk_deployment_rate_limits"], base_year, investment_year
        )

    uk_settings_prepare = snakemake.params.uk_settings_prepare
    df_keadby = snakemake.input.keadby_h2_demand
    df_billingham = snakemake.input.billingham_h2_demand
    if isinstance(uk_settings_prepare["planned_hydrogen_projects"], dict) and (int(investment_year) > int(base_year)):
        add_utilization_constraint_keadby(n, df_keadby, uk_settings_prepare, investment_year)
        add_ammonia_demand(n, df_billingham, uk_settings_prepare, investment_year)

    if isinstance(config.get("local_co2", False), dict):
        logger.info("Adding local CO2 constraint.")
        add_split_co2_constraints(n, config["local_co2"])

    if n.params.custom_extra_functionality:
        source_path = n.params.custom_extra_functionality
        assert os.path.exists(source_path), f"{source_path} does not exist"
        sys.path.append(os.path.dirname(source_path))
        module_name = os.path.splitext(os.path.basename(source_path))[0]
        module = importlib.import_module(module_name)
        custom_extra_functionality = getattr(module, module_name)
        custom_extra_functionality(n, snapshots, snakemake)


def check_objective_value(n: pypsa.Network, solving: dict, horizon: str) -> None:
    """
    Check if objective value matches expected value within tolerance.

    Parameters
    ----------
    n : pypsa.Network
        Network with solved objective
    solving : Dict
        Dictionary containing objective checking parameters
    horizon : str
        Planning horizon of the current solve, used to select the expected
        value for myopic foresight (per-horizon mapping)

    Raises
    ------
    ObjectiveValueError
        If objective value differs from expected value beyond tolerance
    """
    check_objective = solving["check_objective"]
    if not check_objective["enable"]:
        return
    atol = check_objective["atol"]
    rtol = check_objective["rtol"]
    expected_value = check_objective["expected_value"]
    if isinstance(expected_value, dict):
        expected_value = expected_value[int(horizon)]
    if not np.isclose(n.objective, expected_value, atol=atol, rtol=rtol):
        raise ObjectiveValueError(
            f"Objective value {n.objective} differs from expected value "
            f"{expected_value} by more than {atol}."
        )

def save_duals(n: pypsa.Network) -> None:
    """
    Save dual values of CO2 constraints to CSV files.
    """
    # model specs
    investment_year = snakemake.wildcards.horizon

    constraints = pd.Series(n.model.constraints)
    co2_constraints = constraints.loc[constraints.str.contains("co2", case=False)]
    heating_constraints = constraints.loc[
        constraints.str.contains("maximum_electric_heating_UK", case=False)
    ]
    keadby_constraints = constraints.loc[
        constraints.str.contains("keadby", case=False)
    ]

    for constraint_name in pd.concat(
        [co2_constraints, heating_constraints, keadby_constraints]
    ).unique():
        dual = n.model.dual[constraint_name]
        df = pd.DataFrame(dual.values)
        coordinates = list(dual.coords)
        if len(coordinates) == 1:
            df.index = dual.coords[coordinates[0]].values
            df.index.name = coordinates[0]
        elif len(coordinates) == 2:
            df.index = dual.coords[coordinates[0]].values
            df.columns = dual.coords[coordinates[1]].values
            df.index.name = coordinates[0]
            df.columns.name = coordinates[1]
        df.to_csv(
            f"results/{snakemake.params.RDIR}/networks/"
            f"{constraint_name}_{investment_year}.csv"
        )

def collect_kwargs(
    config: dict,
    solving: dict,
    planning_horizons: str | None = None,
    log_fn: str | None = None,
    mode: str = "single",
    horizon: int | None = None,
    overlap: int | None = None,
) -> tuple[dict, dict]:
    """
    Prepare keyword arguments separated for model creation and model solving.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing solver settings
    solving : dict
        Dictionary of solving options and configuration
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight
    log_fn : str, optional
        Path to solver log file
    mode : str, optional
        Optimization mode: 'single', 'rolling_horizon', or 'iterative'
        Default is 'single'

    Returns
    -------
    tuple[dict, dict]
        Two dictionaries: (model_kwargs, solve_kwargs)
        - model_kwargs: Arguments for n.optimize.create_model()
        - solve_kwargs: Arguments for n.optimize.solve_model()
        For 'rolling_horizon' and 'iterative' modes, returns merged kwargs
        with additional mode-specific parameters
    """
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    # Model creation kwargs
    model_kwargs = {}
    model_kwargs["multi_investment_periods"] = config["foresight"] == "perfect"
    model_kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    model_kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment", False
    )

    # Solve kwargs
    solver_name = solving["solver"]["name"]
    solver_options = solving["solver_options"][set_of_options] if set_of_options else {}

    solve_kwargs = {}
    solve_kwargs["solver_name"] = solver_name
    solve_kwargs["solver_options"] = solver_options
    solve_kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    solve_kwargs["io_api"] = cf_solving.get("io_api", None)
    solve_kwargs["keep_files"] = cf_solving.get("keep_files", False)

    if log_fn:
        solve_kwargs["log_fn"] = log_fn

    oetc = solving.get("oetc", None)
    if oetc:
        oetc["credentials"] = OetcCredentials(
            email=os.environ["OETC_EMAIL"], password=os.environ["OETC_PASSWORD"]
        )
        oetc["solver"] = solver_name
        oetc["solver_options"] = solver_options
        oetc_settings = OetcSettings(**oetc)
        oetc_handler = OetcHandler(oetc_settings)
        solve_kwargs["remote"] = oetc_handler

    if solver_name == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    # Handle special modes
    if mode == "rolling_horizon":
        all_kwargs = {**model_kwargs, **solve_kwargs}
        all_kwargs["horizon"] = (
            horizon if horizon is not None else cf_solving["horizon"]
        )
        all_kwargs["overlap"] = (
            overlap if overlap is not None else cf_solving["overlap"]
        )
        return all_kwargs, {}

    elif mode == "iterative":
        all_kwargs = {**model_kwargs, **solve_kwargs}
        all_kwargs["track_iterations"] = cf_solving["track_iterations"]
        all_kwargs["min_iterations"] = cf_solving["min_iterations"]
        all_kwargs["max_iterations"] = cf_solving["max_iterations"]

        if cf_solving["post_discretization"].get("enable", False):
            logger.info("Add post-discretization parameters.")
            cf_solving["post_discretization"].pop("enable", None)
            all_kwargs.update(cf_solving["post_discretization"])

        return all_kwargs, {}

    return model_kwargs, solve_kwargs


def create_optimization_model(
    n: pypsa.Network,
    config: dict,
    params: dict,
    model_kwargs: dict,
    solve_kwargs: dict,
    planning_horizons: str | None = None,
    snakemake=None,
) -> None:
    """
    Prepare optimization problem by creating model and adding extra functionality.

    This function:
    1. Attaches config and params to network for extra_functionality
    2. Creates the optimization model
    3. Adds extra functionality (custom constraints)

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network instance
    config : dict
        Configuration dictionary containing solver settings
    params : dict
        Dictionary of solving parameters
    model_kwargs : dict
        Arguments for n.optimize.create_model()
    solve_kwargs : dict
        Arguments for n.optimize.solve_model()
    planning_horizons : str, optional
        The current planning horizon year or None in perfect foresight
    """
    # Add config and params to network for extra_functionality
    n.config = config
    n.params = params

    # Create optimization model
    logger.info("Creating optimization model...")
    n.optimize.create_model(**model_kwargs)

    # Add extra functionality (custom constraints)
    logger.info("Adding extra functionality (custom constraints)...")
    extra_functionality(n, n.snapshots, planning_horizons, snakemake=snakemake)

    save_duals(n)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_network",
            configfiles="config/test/config.overnight.yaml",
            horizon=2030,
        )
    configure_logging(snakemake)  # pylint: disable=E0606
    set_scenario_config(snakemake)

    solve_opts = snakemake.params.solving["options"]
    cf_solving = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    # Load network
    n = pypsa.Network(snakemake.input.network)
    planning_horizons = snakemake.wildcards.horizon

    # Prepare network (settings before solving)
    prepare_network(
        n,
        config=snakemake.config,
        solve_opts=snakemake.params.solving["options"],
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params["sector"]["limit_max_growth"],
        rolling_horizon=cf_solving["rolling_horizon"],
    )

    # Determine solve mode
    rolling_horizon = cf_solving["rolling_horizon"]
    skip_iterations = cf_solving.get("skip_iterations", False)

    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )

    # Solve network based on mode
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        if rolling_horizon:
            logger.info("Using rolling horizon optimization...")
            all_kwargs, _ = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="rolling_horizon",
            )

            n.config = snakemake.config
            n.params = snakemake.params
            all_kwargs["extra_functionality"] = partial(
                extra_functionality,
                planning_horizons=planning_horizons,
                snakemake=snakemake,
            )
            n.optimize.optimize_with_rolling_horizon(**all_kwargs)
            status, condition = "", ""

        elif skip_iterations:
            logger.info("Using single-pass optimization...")
            model_kwargs, solve_kwargs = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="single",
            )
            create_optimization_model(
                n,
                config=snakemake.config,
                params=snakemake.params,
                model_kwargs=model_kwargs,
                solve_kwargs=solve_kwargs,
                planning_horizons=planning_horizons,
                snakemake=snakemake,
            )

            logger.info("Solving model...")
            status, condition = n.optimize.solve_model(**solve_kwargs)

        else:
            logger.info("Using iterative transmission expansion optimization...")

            all_kwargs, _ = collect_kwargs(
                snakemake.config,
                snakemake.params.solving,
                planning_horizons,
                log_fn=snakemake.log.solver,
                mode="iterative",
            )

            n.config = snakemake.config
            n.params = snakemake.params
            all_kwargs["extra_functionality"] = partial(
                extra_functionality,
                planning_horizons=planning_horizons,
                snakemake=snakemake,
            )
            status, condition = n.optimize.optimize_transmission_expansion_iteratively(
                **all_kwargs
            )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    # Check results
    if not rolling_horizon:
        if status != SolverStatus.ok:
            logger.warning(
                f"Solving status '{status}' with termination condition '{condition}'"
            )
        check_objective_value(n, snakemake.params.solving, snakemake.wildcards.horizon)

    if condition in [
        TerminationCondition.infeasible,
        TerminationCondition.infeasible_or_unbounded,
    ]:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'. Infeasibilities computed.")

    if status == SolverStatus.warning:
        raise RuntimeError("Solving status 'warning'. Discarding solution.")

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)

    if snakemake.output.get("model"):
        n.model.to_netcdf(snakemake.output.model)

    config_output = getattr(snakemake.output, "config", None)
    if config_output:
        with open(config_output, "w") as file:
            yaml.dump(
                n.meta,
                file,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
