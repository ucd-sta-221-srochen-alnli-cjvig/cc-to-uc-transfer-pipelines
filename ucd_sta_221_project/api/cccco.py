"""
This module contains functions to interact with the California Community
Colleges Chancellor's Office (CCCCO) API. The API provides information about
colleges, districts, and programs in the California Community Colleges system.

The API documentation can be found at https://api.cccco.edu.
"""

import pandas as pd
import requests
import urllib3

# Disable insecure request warnings for Chancellor's Office API
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_ccc_colleges(search_param: str | None = None) -> pd.DataFrame:
    """
    Sends a `GET` request to the CCCCO API to retrieve a list of colleges.

    :param search_param: The parameters for the search query. Either the college
        MIS ID or the college name. If `None`, all colleges are returned.
    :return: A pandas DataFrame containing the list of colleges.
    """

    cccco_api_url = "https://api.cccco.edu/"

    if not search_param:
        cccco_api_url += "colleges"
    elif search_param.isnumeric():
        cccco_api_url += f"colleges/{search_param}"
    else:
        cccco_api_url += f"colleges/search/{search_param}"

    try:
        response = requests.get(cccco_api_url, verify=False)
        response.raise_for_status()
        return pd.DataFrame(response.json()).drop(columns=["CollegeContacts"])
    except Exception as e:
        print(f"Error occurred for `{search_param = }`: {e}")
        return pd.DataFrame()


def get_ccc_districts(search_param: str | None = None) -> pd.DataFrame:
    """
    Sends a `GET` request to the CCCCO API to retrieve a list of districts.

    :param search_param: The parameters for the search query. Either the
        district MIS ID or the college name. If `None`, all districts are
        returned.
    :return: A pandas DataFrame containing the list of districts.
    """

    cccco_api_url = "https://api.cccco.edu/"

    if not search_param:
        cccco_api_url += "districts"
    elif search_param.isnumeric():
        cccco_api_url += f"districts/{search_param}"
    else:
        cccco_api_url += f"districts/search/{search_param}"

    try:
        response = requests.get(cccco_api_url, verify=False)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        print(f"Error occurred for `{search_param = }`: {e}")
        return pd.DataFrame()


def get_ccc_programs(search_param: str | None = None) -> pd.DataFrame:
    """
    Sends a `GET` request to the CCCCO API to retrieve a list of programs.

    :param search_param: The parameters for the search query. Either the TOP
        code or a keyword the program title contains. If `None`, all TOP codes
        and program titles are returned.
    :return: A pandas DataFrame containing the list of programs.
    """

    cccco_api_url = "https://api.cccco.edu/"

    if not search_param:
        cccco_api_url += "programs"
    elif search_param.isnumeric():
        cccco_api_url += f"programs/search/{search_param}"
    else:
        cccco_api_url += f"programs/search/{search_param}"

    try:
        response = requests.get(cccco_api_url, verify=False)
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        print(f"Error occurred for `{search_param = }`: {e}")
        return pd.DataFrame()
