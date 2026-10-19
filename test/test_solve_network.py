# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pandas as pd
import pypsa
import pytest

from scripts.prepare_sector_network import add_import_options, get_import_configs
from scripts.solve_network import (
    add_co2_atmosphere_constraint,
    add_import_limit_constraint,
    calculate_co2_limit,
    emission_stores,
    import_assets_by_type,
    valid_local_co2_regions,
)


@pytest.fixture(scope="function")
def co2_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=3, freq="h"))
    n.add("Carrier", "co2", co2_emissions=1.0)
    n.add("Bus", "co2 atmosphere", carrier="co2")
    n.add("Store", "co2 atmosphere", bus="co2 atmosphere", e_nom=1e6, e_min_pu=-1)
    n.add("Bus", "electricity")
    n.add("Load", "load", bus="electricity", p_set=1.0)
    n.add("Generator", "gas", bus="electricity", p_nom=10.0, marginal_cost=1.0)
    return n


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_add_co2_atmosphere_constraint_honours_sense(
    co2_network: pypsa.Network, sense: str
) -> None:
    co2_network.add(
        "GlobalConstraint",
        "CO2Bound",
        type="co2_atmosphere",
        carrier_attribute="co2_emissions",
        sense=sense,
        constant=100.0,
    )
    co2_network.optimize.create_model()
    add_co2_atmosphere_constraint(co2_network, co2_network.snapshots)

    constraint = co2_network.model.constraints["GlobalConstraint-CO2Bound"]
    expected = {"<=": "<=", ">=": ">=", "==": "="}[sense]
    assert (constraint.sign == expected).all()
    assert (constraint.rhs == 100.0).all()


def test_add_co2_atmosphere_constraint_rejects_unknown_sense(
    co2_network: pypsa.Network,
) -> None:
    co2_network.add(
        "GlobalConstraint",
        "CO2Bound",
        type="co2_atmosphere",
        carrier_attribute="co2_emissions",
        sense="<",
        constant=100.0,
    )
    co2_network.optimize.create_model()
    with pytest.raises(ValueError, match="Unsupported sense"):
        add_co2_atmosphere_constraint(co2_network, co2_network.snapshots)


@pytest.mark.parametrize(
    "attr,value", [("e_initial", 5.0), ("e_initial_per_period", True)]
)
def test_emission_stores_rejects_seeded_level(
    co2_network: pypsa.Network, attr: str, value: float | bool
) -> None:
    co2_network.stores.loc["co2 atmosphere", attr] = value
    with pytest.raises(ValueError, match="non-zero level"):
        emission_stores(co2_network, pd.Index(["co2"]))


def test_emission_stores_selects_accumulating_stores(
    co2_network: pypsa.Network,
) -> None:
    assert list(emission_stores(co2_network, pd.Index(["co2"])).index) == [
        "co2 atmosphere"
    ]
    co2_network.stores.loc["co2 atmosphere", "e_cyclic"] = True
    assert emission_stores(co2_network, pd.Index(["co2"])).empty


@pytest.mark.parametrize(
    ("limit", "constraint_name"),
    [
        (1.0, "green_import_limit"),
        ({"GB": 1.0}, "green_import_limit_GB"),
    ],
)
def test_add_import_limit_constraint_honours_limit(
    limit: float | dict, constraint_name: str
) -> None:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=2, freq="h"))
    n.add("Carrier", "import gas")
    n.add("Bus", "gas source")
    n.add("Bus", "GB fuel")
    n.add("Bus", "DE fuel")
    n.add(
        "Link",
        "GB gas import",
        bus0="gas source",
        bus1="GB fuel",
        carrier="import gas",
        p_nom=10,
        efficiency=1,
        marginal_cost=1,
    )
    n.add(
        "Generator",
        "DE gas import",
        bus="DE fuel",
        carrier="import gas",
        p_nom=10,
        marginal_cost=1,
    )
    n.add("Load", "GB load", bus="GB fuel", p_set=1)
    n.add("Load", "DE load", bus="DE fuel", p_set=1)
    n.optimize.create_model()

    add_import_limit_constraint(n, n.snapshots, import_type="green", limit=limit)

    constraint = n.model.constraints[constraint_name]
    assert constraint.rhs.item() == pytest.approx(1e6 * 2 / 8760)


def test_import_assets_are_scoped_to_category() -> None:
    n = pypsa.Network()
    n.add("Carrier", "import gas")
    n.add("Carrier", "import methanol")
    n.add("Carrier", "import oil")
    n.add("Carrier", "import H2")
    n.add("Carrier", "import NH3")
    n.add("Bus", "source")
    n.add("Bus", "fuel")
    n.add(
        "Link",
        "green gas import",
        bus0="source",
        bus1="fuel",
        carrier="import gas",
        p_nom=1,
        efficiency=1,
    )
    n.add(
        "Link",
        "green oil import",
        bus0="source",
        bus1="fuel",
        carrier="import oil",
        p_nom=1,
        efficiency=1,
    )
    n.add(
        "Link",
        "green methanol import",
        bus0="source",
        bus1="fuel",
        carrier="import methanol",
        p_nom=1,
        efficiency=1,
    )
    n.add(
        "Generator",
        "fossil gas import",
        bus="fuel",
        carrier="import gas",
        p_nom=1,
    )
    n.add(
        "Generator",
        "green hydrogen import",
        bus="fuel",
        carrier="import H2",
        p_nom=1,
    )
    n.add(
        "Generator",
        "other ammonia import",
        bus="fuel",
        carrier="import NH3",
        p_nom=1,
    )

    green_links, green_gens = import_assets_by_type(n, "green")
    fossil_links, fossil_gens = import_assets_by_type(n, "fossil")
    other_links, other_gens = import_assets_by_type(n, "other")

    assert set(green_links) == {
        "green gas import",
        "green methanol import",
        "green oil import",
    }
    assert set(green_gens) == {"green hydrogen import"}
    assert fossil_links.empty
    assert set(fossil_gens) == {"fossil gas import"}
    assert other_links.empty
    assert set(other_gens) == {"other ammonia import"}


def test_import_assets_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="Unsupported import category"):
        import_assets_by_type(pypsa.Network(), "unknown")


def test_import_assets_rejects_unclassified_carriers() -> None:
    n = pypsa.Network()
    n.add("Carrier", "import ammonia")
    n.add("Bus", "fuel")
    n.add(
        "Generator",
        "ammonia import",
        bus="fuel",
        carrier="import ammonia",
        p_nom=1,
    )

    with pytest.raises(ValueError, match="Unclassified import generators"):
        import_assets_by_type(n, "other")


def test_add_import_limit_constraint_requires_one_category() -> None:
    n = pypsa.Network()
    snapshots = pd.DatetimeIndex([])

    with pytest.raises(TypeError, match="Missing required import category"):
        add_import_limit_constraint(n, snapshots)
    with pytest.raises(ValueError, match="Specify only one"):
        add_import_limit_constraint(n, snapshots, "green", type="green")


def test_add_import_limit_constraint_supports_legacy_type_keyword() -> None:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=1, freq="h"))

    with pytest.warns(DeprecationWarning, match="use `import_type`"):
        add_import_limit_constraint(n, n.snapshots, type="green", limit=False)


def test_add_import_limit_constraint_ignores_disabled_limit() -> None:
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=1, freq="h"))
    n.add("Carrier", "import gas")
    n.add("Bus", "GB fuel")
    n.add(
        "Generator",
        "GB gas import",
        bus="GB fuel",
        carrier="import gas",
        p_nom=10,
        marginal_cost=1,
    )
    n.optimize.create_model()

    add_import_limit_constraint(n, n.snapshots, import_type="green", limit=False)

    assert "green_import_limit" not in n.model.constraints


def test_get_import_configs_supports_categories_and_legacy_green_fallback() -> None:
    green = {"enable": True, "price": {"gas": 122}}
    fossil = {"enable": True, "price": {"gas": 50}}
    other = {"enable": True, "price": {"NH3": 125}}

    assert get_import_configs(
        {
            "green_imports": green,
            "fossil_imports": fossil,
            "other_imports": other,
            "imports": {"enable": False},
        }
    ) == {"green": green, "fossil": fossil, "other": other}
    assert get_import_configs({"imports": green}) == {
        "green": green,
        "fossil": {},
        "other": {},
    }
    assert get_import_configs(
        {"green_imports": {}, "imports": green}
    )["green"] == {}


@pytest.mark.parametrize(
    ("import_type", "carrier", "settings", "bus", "p_nom"),
    [
        ("fossil", "import gas", {"gas": 50}, "GB gas", 1.0),
        ("other", "import NH3", {"NH3": 125}, "GB ammonia", 1e7),
    ],
)
def test_add_import_options_builds_category_assets(
    import_type, carrier, settings, bus, p_nom
):
    n = pypsa.Network()
    n.add("Carrier", carrier)
    n.add("Bus", bus)
    if import_type == "fossil":
        gas_input_nodes = pd.DataFrame({"lng": [1.0]}, index=["GB"])
        spatial = SimpleNamespace(gas=SimpleNamespace(nodes=[bus]))
    else:
        gas_input_nodes = pd.DataFrame({"lng": [], "pipeline": []})
        spatial = SimpleNamespace(ammonia=SimpleNamespace(nodes=[bus]))

    add_import_options(
        n,
        pd.DataFrame(),
        {"ammonia": True},
        spatial,
        gas_input_nodes,
        import_type,
        {"price": settings},
    )

    generator = n.generators.loc[n.generators.carrier == carrier]
    assert generator.index.tolist() == [f"{bus} import"]
    assert generator.bus.item() == bus
    assert generator.p_nom.item() == pytest.approx(p_nom)


def test_calculate_co2_limit_supports_relative_and_absolute_budgets() -> None:
    relative = {"relative": True, "upper": {2030: 0.5}}
    absolute = {"relative": False, "upper": {2030: 2.0}}

    assert calculate_co2_limit(2030, relative, 10.0, 50.0) == 5.0
    assert calculate_co2_limit(2030, absolute, 10.0, 50.0) == 400e6


def test_invalid_local_co2_entries_remain_out_of_local_constraints() -> None:
    co2_totals = pd.DataFrame({"electricity": [1.0, 2.0]}, index=["GB", "DE"])
    local_co2 = {
        "GB": {2030: 0.5},
        "XX": {2030: 0.5},
        "DE": {2040: 0.5},
    }

    assert valid_local_co2_regions(local_co2, co2_totals, 2030, ["GB", "DE"]) == [
        "GB"
    ]
