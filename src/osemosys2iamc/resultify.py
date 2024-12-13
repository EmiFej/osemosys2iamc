"""Create an IAMC dataset from a package of OSeMOSYS results

Run the command::

    python resultify.py <input_path> <results_path> <config_path> <output_path>"

where:

    ``input_path`` is the path to the folder of CSV files containing input files
    ``results_path`` is the path to the folder of CSV files holding OSeMOSYS results
    ``config_path`` is the path to the ``config.yaml`` file containing the results mapping
    ``output_path`` is the path to the CSV file written out in IAMC format

"""
import functools
from multiprocessing.sharedctypes import Value
from sqlite3 import DatabaseError
import pandas as pd
import pyam
from iso3166 import countries_by_alpha2, countries_by_alpha3
import sys
import os
from typing import List, Dict, Optional
from yaml import load, SafeLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

countries_by_alpha2["UK"] = countries_by_alpha2["GB"]
countries_by_alpha2["EL"] = countries_by_alpha2["GR"]


def iso_to_country(
    iso_format: str, index: List[str], osemosys_param: str
) -> pd.DataFrame:

    countries_list = []
    no_country_extracted = []
    format_regex = r"^iso[23]_([1-9]\d*|start|end)$"

    
    if re.search(format_regex, iso_format) is not None:

        iso_type, abbr_loc = iso_format[3:].split("_")

        
        if iso_type == "2":
            country_dict = countries_by_alpha2
        elif iso_type == "3":
            country_dict = countries_by_alpha3

        
        if abbr_loc == "start":
            region_regex = r"^(.{" + iso_type + r"}).*$"
        elif abbr_loc == "end":
            region_regex = r"^.*(.{" + iso_type + r"})$"
        elif abbr_loc.isnumeric:
            region_regex = (
                r"^.{" + str(int(abbr_loc) - 1) + r"}(.{" + iso_type + r"}).*$"
            )

        for i in index:
            if re.search(region_regex, i.upper()) != None:
                code = re.search(region_regex, i.upper()).groups()[0]
                if code in country_dict:
                    countries_list.append(country_dict[code].name)
                else:
                    countries_list.append("")
                    no_country_extracted.append(i)
            else:
                countries_list.append("")
                no_country_extracted.append(i)

    else:
        raise ValueError(
            "Invalid ISO type or abbreviation location. Valid locations are 'start', 'end', and a positive number denoting the start of the abbreviation in the string."
        )

    if len(no_country_extracted) > 0:
        print(
            f"Using the ISO option, Countries were not found from the following technologies/fuels: {set(no_country_extracted)}"
        )
        print(
            f"Kindly check your region naming option or the technology/fuel names in file: {osemosys_param}.\n"
        )

    return countries_list


def read_file(path: str, osemosys_param: str, region_name_option: str) -> pd.DataFrame:


    filename = os.path.join(path, osemosys_param + ".csv")
    df = pd.read_csv(filename)


    if "iso" in region_name_option:
        if "FUEL" in df.columns:
            df["REGION"] = iso_to_country(
                region_name_option, df["FUEL"], osemosys_param
            )
        elif "TECHNOLOGY" in df.columns:
            df["REGION"] = iso_to_country(
                region_name_option, df["TECHNOLOGY"], osemosys_param
            )
        elif "EMISSION" in df.columns:
            df["REGION"] = iso_to_country(
                region_name_option, df["EMISSION"], osemosys_param
            )
    elif region_name_option == "from_csv":
        df["REGION"] = df["REGION"]
    else:
        df["REGION"] = region_name_option

    return df


def filter_regex(df: pd.DataFrame, patterns: List[str], column: str) -> pd.DataFrame:

    masks = [df[column].str.match(p) for p in patterns]
    return pd.concat([df[mask] for mask in masks])


def filter_fuels(df: pd.DataFrame, fuels: List[str]) -> pd.DataFrame:

    return filter_regex(df, fuels, "FUEL")


def filter_technologies(df: pd.DataFrame, technologies: List[str]) -> pd.DataFrame:

    return filter_regex(df, technologies, "TECHNOLOGY")


def filter_technology_fuel(
    df: pd.DataFrame, technologies: List, fuels: List
) -> pd.DataFrame:
    
    df = filter_technologies(df, technologies)
    df = filter_fuels(df, fuels)

    df = df.groupby(by=["REGION", "YEAR"], as_index=False)["VALUE"].sum()
    return df[df.VALUE != 0]


def filter_emission_tech(
    df: pd.DataFrame, emission: List[str], technologies: Optional[List[str]] = None
) -> pd.DataFrame:


    df = filter_regex(df, emission, "EMISSION")

    if technologies:
        
        df = filter_technologies(df, technologies)

    df = df.groupby(by=["REGION", "YEAR"], as_index=False)["VALUE"].sum()
    return df[df.VALUE != 0]


def filter_capacity(df: pd.DataFrame, technologies: List[str]) -> pd.DataFrame:

    df = filter_technologies(df, technologies)

    df = df.groupby(by=["REGION", "YEAR"], as_index=False)["VALUE"].sum()
    return df[df.VALUE != 0]


def filter_final_energy(df: pd.DataFrame, fuels: List) -> pd.DataFrame:

    df_f = filter_fuels(df, fuels)

    df = df_f.groupby(by=["REGION", "YEAR"], as_index=False)["VALUE"].sum()
    return df[df.VALUE != 0]


def calculate_trade(results: dict, techs: List) -> pd.DataFrame:
    """Return dataframe with the net exports of a commodity"""

    exports = filter_capacity(results["UseByTechnology"], techs).set_index(
        ["REGION", "YEAR"]
    )
    imports = filter_capacity(results["ProductionByTechnologyAnnual"], techs).set_index(
        ["REGION", "YEAR"]
    )
    df = exports.subtract(imports, fill_value=0)

    return df.reset_index()


def extract_results(df: pd.DataFrame, technologies: Optional[List] = None, fuels: Optional[List] = None) -> pd.DataFrame:
    if technologies and "TECHNOLOGY" in df.columns:
        mask = df.TECHNOLOGY.isin(technologies)
        return df[mask]
    elif fuels and "FUEL" in df.columns:
        mask = df.FUEL.isin(fuels)
        return df[mask]
    else:
        return df


def load_config(filepath: str) -> Dict:

    with open(filepath, "r") as configfile:
        config = load(configfile, Loader=SafeLoader)
    return config


def make_plots(df: pyam.IamDataFrame, model: str, scenario: str, regions: List[str]):


    args = dict(model=model, scenario=scenario)
    print(args, regions)

    fig, ax = plt.subplots()

    assert isinstance(regions, list)

    for region in regions:
        assert isinstance(region, str)
        print(f"Plotting {region}")
        # Plot primary energy
        data = df.filter(
            **args, variable="Primary Energy|*", region=region
        )  # type: pyam.IamDataFrame
        if data:
            print(data)
            locator = mdates.AutoDateLocator(minticks=10)
            # locator.intervald['YEARLY'] = [10]
            data.plot.bar(ax=ax, stacked=True, title="Primary energy mix %s" % region)
            plt.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left")
            plt.tight_layout()
            ax.xaxis.set_major_locator(locator)
            fig.savefig(
                "primary_energy_%s.pdf" % region,
                bbox_inches="tight",
                transparent=True,
                pad_inches=0,
            )
            plt.clf()
        # Plot secondary energy (electricity generation)
        se = df.filter(**args, variable="Secondary Energy|Electricity|*", region=region)
        if se:
            locator = mdates.AutoDateLocator(minticks=10)
            # locator.intervald['YEARLY'] = [10]
            se.plot.bar(ax=ax, stacked=True, title="Power generation mix %s" % region)
            plt.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left")
            plt.tight_layout()
            ax.xaxis.set_major_locator(locator)
            fig.savefig(
                "electricity_generation_%s.pdf" % region,
                bbox_inches="tight",
                transparent=True,
                pad_inches=0,
            )
            plt.clf()
        # Create generation capacity plot
        cap = df.filter(**args, variable="Capacity|Electricity|*", region=region)
        if cap:
            cap.plot.bar(ax=ax, stacked=True, title="Generation Capacity %s" % region)
            plt.legend(bbox_to_anchor=(0.0, -0.25), loc="upper left")
            plt.tight_layout()
            fig.savefig(
                "capacity_%s.pdf" % region,
                bbox_inches="tight",
                transparent=True,
                pad_inches=0,
            )
            plt.clf()

    # Create emissions plot
    emi = df.filter(**args, variable="Emissions|CO2*").filter(
        region="World", keep=False
    )
    # print(emi)
    if emi:
        emi.plot.bar(
            ax=ax,
            bars="region",
            stacked=True,
            title="CO2 emissions by region",
            cmap="tab20",
        )
        plt.legend(bbox_to_anchor=(1.0, 1.05), loc="upper left", ncol=2)
        fig.savefig("emission.pdf", bbox_inches="tight", transparent=True, pad_inches=0)
        plt.clf()


def main(config: Dict, inputs_path: str, results_path: str) -> pyam.IamDataFrame:
    blob = []
    filename = os.path.join(inputs_path, "YEAR.csv")
    years = pd.read_csv(filename)

    try:
        for input in config["inputs"]:
            inputs = read_file(inputs_path, input["osemosys_param"], config["region"])
            unit = input["unit"]

            if "variable_cost" in input.keys():
                technologies = input["variable_cost"]
                data = filter_capacity(inputs, technologies)
            elif "reg_tech_param" in input.keys():
                technologies = input["reg_tech_param"]
                data = filter_technologies(inputs, technologies)
                list_years = years["VALUE"]
                data["YEAR"] = [list_years] * len(data)
                data = data.explode("YEAR").reset_index(drop=True)
                data = data.drop(["TECHNOLOGY"], axis=1)
            elif "new_variable" in input.keys():
                data = process_new_variable(inputs, config)
            if not data.empty:
                data = data.rename(
                    columns={"REGION": "region", "YEAR": "year", "VALUE": "value"}
                )
                iamc = pyam.IamDataFrame(
                    data,
                    model=config["model"],
                    scenario=config["scenario"],
                    variable=input["iamc_variable"],
                    unit=unit,
                )
                blob.append(iamc)
    except KeyError:
        pass

    try:
        for result in config["results"]:
            if isinstance(result["osemosys_param"], str):
                results = read_file(
                    results_path, result["osemosys_param"], config["region"]
                )

                try:
                    technologies = result.get("technology", None)
                except KeyError:
                    technologies = None

                unit = result["unit"]
                if "fuel" in result.keys():
                    fuels = result["fuel"]
                    data = filter_technology_fuel(results, technologies, fuels)
                elif "emissions" in result.keys():
                    if "tech_emi" in result.keys():
                        emission = result["emissions"]
                        technologies = result["tech_emi"]
                        data = filter_emission_tech(results, emission, technologies)
                    else:
                        emission = result["emissions"]
                        data = filter_emission_tech(results, emission)
                elif "capacity" in result.keys():
                    technologies = result["capacity"]
                    data = filter_capacity(results, technologies)
                elif "primary_technology" in result.keys():
                    technologies = result["primary_technology"]
                    data = filter_capacity(results, technologies)
                elif "excluded_prod_tech" in result.keys():
                    technologies = result["excluded_prod_tech"]
                    data = filter_capacity(results, technologies)
                elif "el_prod_technology" in result.keys():
                    technologies = result["el_prod_technology"]
                    data = filter_capacity(results, technologies)
                elif "demand" in result.keys():
                    demands = result["demand"]
                    data = filter_final_energy(results, demands)
                else:
                    fuels = result.get("fuel", None)
                    data = extract_results(results, technologies, fuels)

            elif isinstance(result["osemosys_param"], list):
                results = {}
                unit = result["unit"]
                for p in result["osemosys_param"]:
                    results[p] = read_file(results_path, p, config["region"])

                if "trade_tech" in result.keys():
                    technologies = result["trade_tech"]
                    data = calculate_trade(results, technologies)

                else:
                    name = result["iamc_variable"]
                    raise ValueError(f"No data found for {name}")

            else:
                name = result["iamc_variable"]
                msg = f"Error in configuration file for entry {name}. The `osemosys_param` key must be a string or a list"
                raise ValueError(msg)

            if "transform" in result.keys():
                if result["transform"] == "abs":
                    data["VALUE"] = data["VALUE"].abs()
                else:
                    pass

            if not data.empty:
                data = data.rename(
                    columns={"REGION": "region", "YEAR": "year", "VALUE": "value"}
                )

                iamc = pyam.IamDataFrame(
                    data,
                    model=config["model"],
                    scenario=config["scenario"],
                    variable=result["iamc_variable"],
                    unit=unit,
                )
                blob.append(iamc)
    except KeyError:
        pass

    if len(blob) > 0:
        all_data = pyam.concat(blob)

        all_data.convert_unit("PJ/yr", to="EJ/yr", inplace=True)
        all_data.convert_unit("ktCO2/yr", to="Mt CO2/yr", factor=0.001, inplace=True)
        all_data.convert_unit(
            "MEUR_2015/PJ", to="EUR_2020/GJ", factor=1.05, inplace=True
        )
        all_data.convert_unit(
            "MEUR_2015/GW", to="EUR_2020/kW", factor=1.05, inplace=True
        )
        all_data.convert_unit("kt CO2/yr", to="Mt CO2/yr", inplace=True)

        dic_country_name_variants = {
            "Netherlands": "The Netherlands",
            "Czechia": "Czech Republic",
            "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        }
        all_data.rename(region=dic_country_name_variants, inplace=True)

        all_data = pyam.IamDataFrame(all_data)
        return all_data
    else:
        raise ValueError("No data found")

def aggregate(func):
    """Decorator for filters which returns the aggregated data"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the dataframe from the filter
        data = func(*args, **kwargs)
        # Apply the aggregation
        data = data.groupby(by=["REGION", "YEAR"]).sum()
        # Make the IAMDataFrame
        return pyam.IamDataFrame(
            data,
            model=iam_model,
            scenario=iam_scenario,
            variable=iam_variable,
            unit=iam_unit,
        )

    return wrapper


def entry_point():

    args = sys.argv[1:]

    if len(args) != 4:
        print(
            "Usage: osemosys2iamc <inputs_path> <results_path> <config_path> <output_path>"
        )
        exit(1)

    inputs_path = args[0]
    results_path = args[1]
    configpath = args[2]
    outpath = args[3]

    config = load_config(configpath)

    all_data = main(config, inputs_path, results_path)

    model = config["model"]
    scenario = config["scenario"]
    regions = config["region"]


    all_data.to_excel(outpath, sheet_name="data")


if __name__ == "__main__":

    entry_point()
