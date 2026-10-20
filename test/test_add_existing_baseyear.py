# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

import pandas as pd
import pypsa

from scripts.add_existing_baseyear import add_gas_storage_data


def test_gas_storage_data_can_be_disabled(monkeypatch) -> None:
    n = pypsa.Network()
    n.add("Bus", "GB0", carrier="AC")
    n.add(
        "Store",
        "GB0 gas Store-2030",
        bus="GB0",
        carrier="gas",
        e_nom=10,
        e_nom_min=10,
    )
    n.add("Bus", "DE0", carrier="AC")
    n.add(
        "Store",
        "DE0 gas Store-2030",
        bus="DE0",
        carrier="gas",
        e_nom=1e6,
        e_nom_min=1e6,
    )
    agsi_data = pd.DataFrame(
        [
            {
                "Type": "country",
                "Status": "C",
                "Name": "Germany",
                "Technical Capacity (TWh)": 1.0,
            }
        ]
    )
    monkeypatch.setattr(
        "scripts.add_existing_baseyear.pd.read_csv", lambda *args, **kwargs: agsi_data
    )

    add_gas_storage_data(
        n,
        onshore=None,
        offshore=None,
        baseyear=2030,
        uk_settings_data={"uk_new_gas_storage_data": False},
        uk_settings_prepare={"reduce_model_in_the_east": False},
    )

    assert n.stores.at["GB0 gas Store-2030", "e_nom"] == 10
