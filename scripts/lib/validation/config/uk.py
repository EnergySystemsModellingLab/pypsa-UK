# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""UK-specific configuration options."""

from typing import Literal

from pydantic import Field

from scripts.lib.validation.config._base import ConfigModel


class UKDataConfig(ConfigModel):
    uk_energy_balance_year: int = Field(2024, description="UK energy-balance data year.")
    uk_electricity_demand_year: int = Field(2025, description="UK electricity-demand data year.")
    uk_new_gas_storage_data: bool = Field(True)
    uk_new_powerplants_data: bool = Field(True)
    uk_new_industry_data: bool = Field(False)
    uk_new_distribution_key: bool = Field(False)


class PlannedHydrogenProjectsConfig(ConfigModel):
    scenario: Literal["pipeline-led", "network-upgrades-led"] = "pipeline-led"
    demand_level: str = "high"
    implementation_year: int = 2030


class UKPrepareConfig(ConfigModel):
    uk_scale_up_offwind_potential: float = 2
    uk_include_offshore_buses: bool = False
    uk_transmission_limits: dict[int, float] = Field(
        default_factory=lambda: {
            2025: 0,
            2030: 0,
            2035: 0.15,
            2040: 0.15,
            2045: 0.15,
            2050: 0.15,
        }
    )
    reduce_model_in_the_east: bool | dict[str, list[str]] = False
    use_dynamic_gas_prices: bool = False
    gas_prices: dict[str, float] = Field(
        default_factory=lambda: {"UK": 36, "EU": 34, "imports": 50}
    )
    planned_hydrogen_projects: PlannedHydrogenProjectsConfig | bool = Field(
        default=False
    )


class UKSolveConfig(ConfigModel):
    uk_only: bool = True
    uk_freeze_base_year_capacities: bool = True
    uk_freeze_wind_projects: dict[str, list[int]] = Field(
        default_factory=lambda: {"onwind": [2030], "offwind": [2030]}
    )
    uk_brownfield_minimum_capacity_factors: dict[str, dict[int, float]] = Field(
        default_factory=lambda: {"CCGT": {2025: 0.35}}
    )
    uk_max_electric_heating_share: bool | float = False
    uk_fixed_electricity_generation_mix: bool = False
    uk_remove_storage: bool = False
    uk_deployment_rate_limits: dict[str, float] = Field(
        default_factory=lambda: {"onwind": 15000}
    )


class UKSettingsConfig(ConfigModel):
    data: UKDataConfig = Field(default_factory=UKDataConfig)
    prepare: UKPrepareConfig = Field(default_factory=UKPrepareConfig)
    solve: UKSolveConfig = Field(default_factory=UKSolveConfig)
