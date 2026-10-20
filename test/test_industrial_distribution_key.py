# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""Tests for industrial distribution-key data preparation."""

import pandas as pd

from scripts.build_industrial_distribution_key import add_uk_fertiliser_ammonia


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
            }
        ],
        index=["Plant"],
    ).to_csv(fertiliser_file)

    result = add_uk_fertiliser_ammonia(ammonia_plants, fertiliser_file)

    assert result.loc["Plant", "Ammonia [kt/a]"] == 0.5
    assert result.loc["Plant", "Country"] == "United Kingdom"
