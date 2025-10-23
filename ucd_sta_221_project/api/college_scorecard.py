import os
import pandas as pd
import requests
import time

from ucd_sta_221_project.api.utils import flatten_dict


def _get_api_key() -> str:
    """
    Get the College Scorecard API key from the SCORECARDAPI environment
    variable. If the environment variable is not set, throws a runtime error.
    """

    key = os.getenv("SCORECARDAPI")
    if not key:
        raise RuntimeError(
            "Environment variable SCORECARDAPI is not set. "
            "Set SCORECARDAPI to your College Scorecard API key."
        )
    return key.strip()


def get_scorecards_by_state(state: str = "CA") -> list[dict]:
    """
    Sends a `GET` request to the College Scorecard API to retrieve a list of
    colleges in a given state.

    :param state: The state to retrieve colleges from. Defaults to "CA".
    :return list: A list of dictionaries containing the college data.
    """

    url = "https://api.data.gov/ed/collegescorecard/v1/schools"
    params = {
        "api_key": _get_api_key(),
        "school.state": state,
        "page": 1,
        "per_page": 100,
    }

    time.sleep(1)
    response = requests.get(url, params=params)
    print(response.json()["metadata"])
    data = response.json()

    number_of_pages = (
        data["metadata"]["total"] // data["metadata"]["per_page"] + 1
    )

    for page in range(2, number_of_pages + 1):
        time.sleep(1)
        params["page"] = page
        response = requests.get(url, params=params)
        print(response.json()["metadata"])
        data["results"] += response.json()["results"]

    return data["results"]


def get_latest_student_scorecard_data_by_state(
        state: str = "CA"
    ) -> pd.DataFrame:
    """
    Get the latest information for colleges in a given state from the College
    Scorecard API

    :param state: The state to retrieve colleges from. Defaults to "CA".
    :return: A DataFrame containing the latest information for colleges in the
        specified state
    """
    data = get_scorecards_by_state(state)

    student = pd.DataFrame()
    for college in data:
        df_temp = pd.DataFrame.from_dict(
            flatten_dict(college.get("latest").get("student")), orient="index"
        ).T
        df_temp["college"] = college.get("school").get("name")
        df_temp = df_temp[
            ["college"] + [col for col in df_temp.columns if col != "college"]
        ]
        student = pd.concat([student, df_temp])

    return student


def get_scorecard_by_college(
        college_name: str,
        college_city: str | None = None,
        college_state: str | None = None
    ) -> pd.DataFrame:
    """
    Get the latest information for a specified college from the College Scorecard API

    :param college_name: The name of the college
    :param college_city: The city where the college is located
    :param college_state: The state where the college is located
    :return: A DataFrame containing the latest information for the college
    """

    url = "https://api.data.gov/ed/collegescorecard/v1/schools"

    # There are hundreds of fields available in the College Scorecard API.
    # These are the ones that are the most readily useful for our purposes.
    fields = [
        "school.name",
        "latest.student.size",
        "latest.student.enrollment.undergrad_12_month",
        "latest.student.demographics.over_23_at_entry",
        "latest.student.demographics.first_generation",
        "latest.student.demographics.median_hh_income",
        "latest.student.demographics.student_faculty_ratio",
        "latest.student.FAFSA_applications",
    ]

    params = {
        "api_key": _get_api_key(),
        "school.name": college_name,
        "fields": ",".join(fields),
    }

    if college_city:
        params["school.city"] = college_city

    if college_state:
        params["school.state"] = college_state

    response = requests.get(url, params=params)
    data = response.json()["results"]

    # Adding missing fields to the data.
    for college in data:
        for field in fields:
            if field not in college:
                college[field] = 0

    # Replacing periods with underscores in the keys.
    data = [
        {
            k.replace(".", "_"): v for k, v in college.items()
        } for college in data
    ]

    data = pd.DataFrame(data)
    # Reorder the columns so that `school_name` is the first column.
    columns = ["school_name"] + [col for col in data.columns if col != "school_name"]
    data = data[columns]
    data[
        [
            "latest_student_size",
            "latest_student_enrollment_undergrad_12_month",
            "latest_student_demographics_over_23_at_entry",
            "latest_student_demographics_first_generation",
            "latest_student_demographics_median_hh_income",
            "latest_student_demographics_student_faculty_ratio",
            "latest_student_FAFSA_applications",
        ]
    ] = data[
        [
            "latest_student_size",
            "latest_student_enrollment_undergrad_12_month",
            "latest_student_demographics_over_23_at_entry",
            "latest_student_demographics_first_generation",
            "latest_student_demographics_median_hh_income",
            "latest_student_demographics_student_faculty_ratio",
            "latest_student_FAFSA_applications",
        ]
    ].apply(pd.to_numeric)

    return data
