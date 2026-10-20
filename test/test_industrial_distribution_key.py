# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for industrial distribution-key data preparation."""

import pandas as pd

from scripts.build_industrial_distribution_key import (
    UK_FERTILISER_SOURCE_DATE,
    add_uk_fertiliser_ammonia,
    select_uk_ammonia_demand,
)


def test_add_uk_fertiliser_ammonia_without_file_preserves_plants():
    ammonia_plants = pd.DataFrame(
        [{"Ammonia [kt/a]": 2.0, "Country": "France"}]
    )

    result = add_uk_fertiliser_ammonia(ammonia_plants, None)

    pd.testing.assert_frame_equal(result, ammonia_plants)


def test_add_uk_fertiliser_ammonia_appends_uk_sites(tmp_path):
    ammonia_plants = pd.DataFrame(
        columns=[
            "Ammonia [kt/a]",
            "Latitude",
            "Longitude",
            "Country",
            "Source",
            "Date",
        ]
    )
    fertiliser_file = tmp_path / "uk_fertiliser.csv"
    pd.DataFrame(
        [
            {
                "Production (tonnes/year)": 1000,
                "lat": 54.0,
                "lon": -1.0,
                "Sources": "UK source",
                "Unmatched": "not retained",
            }
        ],
        index=["Plant"],
    ).to_csv(fertiliser_file)

    result = add_uk_fertiliser_ammonia(ammonia_plants, fertiliser_file)

    plant = result.loc["Plant"]
    assert plant["Ammonia [kt/a]"] == 0.5
    assert plant["Latitude"] == 54.0
    assert plant["Longitude"] == -1.0
    assert plant["Country"] == "United Kingdom"
    assert plant["Source"] == "UK source"
    assert plant["Date"] == UK_FERTILISER_SOURCE_DATE
    assert "Unmatched" not in result.columns


def test_select_uk_ammonia_demand_excludes_non_uk_sites():
    ammonia_plants = pd.DataFrame(
        [
            {"Country": "United Kingdom", "Ammonia [kt/a]": 0.5},
            {"Country": "France", "Ammonia [kt/a]": 2.0},
        ]
    )

    result = select_uk_ammonia_demand(ammonia_plants)

    assert result["Country"].tolist() == ["United Kingdom"]


def test_select_uk_ammonia_demand_is_empty_without_uk_sites():
    ammonia_plants = pd.DataFrame(
        [{"Country": "France", "Ammonia [kt/a]": 2.0}]
    )

    result = select_uk_ammonia_demand(ammonia_plants)

    assert result.empty
