""" 
The various data sources used in this project name the colleges slightly
differently. For intance, College Scorecard uses "Solano Community College"
while CCCCO uses "Solano". The easiest way to match colleges across data sources
is to use the unique college IDs provided by CCCCO.
"""

import pandas as pd
import time

from ucd_sta_221_project.api.cccco import get_ccc_colleges


def get_unique_cc_ids(college_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe containing college names, retrieve the unique CCCCO
    college IDs via a call to the CCCCO API.

    :param college_name: The name of the college to search for.
    :param df: The dataframe containing the college names.
    :return: A dataframe containing the unique CCCCO college IDs, together with
        the college name from the CCCCO API and the passed in college name.
    """
    colleges = pd.DataFrame(
        columns=["CollegeID", "CollegeName", "InputCollegeName"]
    )

    for name in df[college_name].unique():
        time.sleep(0.25)
        cc_college = get_ccc_colleges(search_param=name)

        # If no results were found, ignore this college.
        if not cc_college.empty:
            cc_college = cc_college.loc[
                :, ["CollegeID", "CollegeName"]
            ]
            cc_college["InputCollegeName"] = name
            colleges = pd.concat(
                [colleges, cc_college],
                ignore_index=True
            )

    return colleges

if __name__ == "__main__":
    base_path = "ucd_sta_221_project/ml/q2_persistence_after_transfer/processed_data"

    # NOTE: Some manual imputation was required for colleges that do not show up
    # in the CCCCO API results.

    # For math success and retention data
    cc_math = pd.read_csv(
        f"{base_path}/datamart_math_success_retention_170100_normalized.csv"
    )

    unique_cc_ids = get_unique_cc_ids(
        college_name="College",
        df=cc_math
    )

    unique_cc_ids.to_csv(
        f"{base_path}/cc_unique_ids_math.csv",
        index=False
    )

    # For english success and retention data
    cc_english = pd.read_csv(
        f"{base_path}/datamart_engl_success_retention_150100_normalized.csv"
    )

    unique_cc_ids = get_unique_cc_ids(
        college_name="College",
        df=cc_english
    )

    unique_cc_ids.to_csv(
        f"{base_path}/cc_unique_ids_engl.csv",
        index=False
    )

    # For EOPS data
    eops = pd.read_csv(
        f"{base_path}/cc_eops.csv"
    )

    unique_cc_ids = get_unique_cc_ids(
        college_name="College",
        df=eops
    )

    unique_cc_ids.to_csv(
        f"{base_path}/cc_unique_ids_eops.csv",
        index=False
    )

    # For college scorecard data
    cc_scorecard = pd.read_csv(
        "ucd_sta_221_project/data_files/cc_scorecard.csv",
        usecols=["school.name"]
    )

    unique_cc_ids = get_unique_cc_ids(
        college_name="school.name",
        df=cc_scorecard
    )

    unique_cc_ids.to_csv(
        f"{base_path}/cc_unique_ids_scorecard.csv",
        index=False
    )
