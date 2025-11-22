import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------
DATA_RAW = Path("data_files")
DATA_PROC = Path("ml/q3_enrollment_melt/processed_data")
DATA_PROC.mkdir(exist_ok=True, parents=True)

# -------------------------
def academic_to_calendar_year(val):
    if pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if "-" not in s:
        try:
            return int(float(s))
        except ValueError:
            return pd.NA

    first, second = s.split("-", 1)
    try:
        first_int = int(first)
    except ValueError:
        return pd.NA

    sec_digits = "".join(ch for ch in second if ch.isdigit())
    if not sec_digits:
        return pd.NA
    if len(sec_digits) == 2:
        last2 = int(sec_digits)
        century = (first_int // 100) * 100
        return century + last2
    else:
        return int(sec_digits)

def normalize_cc(name: str):
    if pd.isna(name):
        return name

    s = str(name).upper().strip()

    s = s.replace(".", "")

    for token in [
        " COMMUNITY COLLEGE",
        " COMM COLLEGE",
        " CMTY",          
        " JR COLLEGE",
        " JUNIOR COLLEGE",
        " COLLEGE",
        "COLLEGE ",
        " CCD",
        " C.C.",
        " DISTRICT",
        " TOTAL",
    ]:
        s = s.replace(token, "")

    s = " ".join(s.split())

    cc_aliases = {
        "CHABOT HAYWARD": "CHABOT",

        "PASADENA": "PASADENA CITY",
        "RIVERSIDE": "RIVERSIDE CITY",
        "SAN BERNARDINO": "SAN BERNARDINO VALLEY",
        "SANTA BARBARA": "SANTA BARBARA CITY",
        "SANTA ROSA": "SANTA ROSA JUNIOR",

        "COALINGA": "WEST HILLS COALINGA",
        "LEMOORE": "WEST HILLS LEMOORE",
        "WEST LA": "WEST LOS ANGELES",

        "NAPA": "NAPA VALLEY",
        "LAS POSITAS": "LAS POSITAS", 
        "DEANZA": "DE ANZA",

        "OF THE SISKIYOUS": "SISKIYOUS",
        "OF THE SEQUOIAS": "SEQUOIAS",
        "OF THE REDWOODS": "REDWOODS",
        "OF THE DESERT": "DESERT",
        "OF THE CANYONS": "CANYONS",
        "OF SAN MATEO": "SAN MATEO",
        "OF MARIN": "MARIN",
        "OF ALAMEDA": "ALAMEDA",

        "SAN MATEO": "SAN MATEO",
        "MARIN": "MARIN",
        "ALAMEDA": "ALAMEDA",
        "SAN FRANCISCO": "CITY OF SAN FRANCISCO",
        "SAN FRANCISCO CTRS": "CITY OF SAN FRANCISCO",
        "CITY OF SAN FRANCISCO": "CITY OF SAN FRANCISCO",

        "MT SAN JACINTO": "MOUNT SAN JACINTO",
        "MT SAN ANTONIO": "MOUNT SAN ANTONIO",
       
        "EAST LA": "EAST LOS ANGELES",
        "IMPERIAL": "IMPERIAL VALLEY",
        "IRVINE": "IRVINE VALLEY",
        "LONG BEACH": "LONG BEACH CITY",

        "LA CITY": "LOS ANGELES CITY",
        "LA HARBOR": "LOS ANGELES HARBOR",
        "LA MISSION": "LOS ANGELES MISSION",
        "LA PIERCE": "LOS ANGELES PIERCE",
        "LA SWEST": "LOS ANGELES SOUTHWEST",
        "LA TRADE": "LOS ANGELES TRADE TECHNICAL",
        "LA VALLEY": "LOS ANGELES VALLEY",

        "MODESTO": "MODESTO JUNIOR",
        "MONTEREY": "MONTEREY PENINSULA",
    }

    s = cc_aliases.get(s, s)
    return s

def normalize_uc(name: str):
    if pd.isna(name):
        return name
    s = str(name).strip()

    mapping = {
        # Berkeley
        "University of California-Berkeley": "Berkeley",
        "UC Berkeley": "Berkeley",
        "UCB": "Berkeley",
        "Berkeley": "Berkeley",

        # Davis
        "University of California-Davis": "Davis",
        "UC Davis": "Davis",
        "UCD": "Davis",
        "Davis": "Davis",

        # Irvine
        "University of California-Irvine": "Irvine",
        "UC Irvine": "Irvine",
        "UCI": "Irvine",
        "Irvine": "Irvine",

        # Los Angeles
        "University of California-Los Angeles": "Los Angeles",
        "University of California, Los Angeles": "Los Angeles",
        "UC Los Angeles": "Los Angeles",
        "UCLA": "Los Angeles",
        "Los Angeles": "Los Angeles",

        # Merced
        "University of California-Merced": "Merced",
        "UC Merced": "Merced",
        "UCM": "Merced",
        "Merced": "Merced",

        # Riverside
        "University of California-Riverside": "Riverside",
        "UC Riverside": "Riverside",
        "UCR": "Riverside",
        "Riverside": "Riverside",

        # San Diego
        "University of California-San Diego": "San Diego",
        "UC San Diego": "San Diego",
        "UCSD": "San Diego",
        "San Diego": "San Diego",

        # Santa Barbara
        "University of California-Santa Barbara": "Santa Barbara",
        "UC Santa Barbara": "Santa Barbara",
        "UCSB": "Santa Barbara",
        "Santa Barbara": "Santa Barbara",

        # Santa Cruz
        "University of California-Santa Cruz": "Santa Cruz",
        "UC Santa Cruz": "Santa Cruz",
        "UCSC": "Santa Cruz",
        "Santa Cruz": "Santa Cruz",
    }

    return mapping.get(s, s)

# =====================================================
# 1. Scorecard features: CC + UC
# =====================================================

cc_score_cols = [
    "year",
    "school.name",
    "aid.ftft_pell_grant_rate",
    "aid.ftft_federal_loan_rate",
    "aid.pell_grant_rate",
    "aid.federal_loan_rate",
    "student.enrollment.undergrad_12_month",
    "cost.attendance.academic_year",
]

uc_score_cols = cc_score_cols + ["admissions.admission_rate.overall"]

# ----- CC scorecard -----
cc_raw = pd.read_csv(DATA_RAW / "cc_scorecard.csv", usecols=cc_score_cols)

cc_feat = (
    cc_raw.rename(
        columns={
            "school.name": "cc_name_raw",
            "aid.ftft_pell_grant_rate": "cc_ftft_pell_rate",
            "aid.ftft_federal_loan_rate": "cc_ftft_fedloan_rate",
            "aid.pell_grant_rate": "cc_pell_rate",
            "aid.federal_loan_rate": "cc_fedloan_rate",
            "student.enrollment.undergrad_12_month": "cc_ug_enroll_12m",
            "cost.attendance.academic_year": "cc_coa_ay",
        }
    )
    .assign(
        cc_name=lambda d: d["cc_name_raw"].apply(normalize_cc),
        year=lambda d: d["year"].apply(academic_to_calendar_year).astype("Int64"),
    )
    .drop(columns=["cc_name_raw"])
    .sort_values(["cc_name", "year"])
)

cc_feat.to_csv(DATA_PROC / "cc_scorecard_features.csv", index=False)

# ----- UC scorecard -----
uc_raw = pd.read_csv(DATA_RAW / "uc_scorecard.csv", usecols=uc_score_cols)

uc_feat = (
    uc_raw.rename(
        columns={
            "school.name": "uc_name_full",
            "aid.ftft_pell_grant_rate": "uc_ftft_pell_rate",
            "aid.ftft_federal_loan_rate": "uc_ftft_fedloan_rate",
            "aid.pell_grant_rate": "uc_pell_rate",
            "aid.federal_loan_rate": "uc_fedloan_rate",
            "student.enrollment.undergrad_12_month": "uc_ug_enroll_12m",
            "cost.attendance.academic_year": "uc_coa_ay",
            "admissions.admission_rate.overall": "uc_admit_rate_overall",
        }
    )
    .assign(
        uc_campus=lambda d: d["uc_name_full"].apply(normalize_uc),
        year=lambda d: d["year"].apply(academic_to_calendar_year).astype("Int64"),
    )
    .drop(columns=["uc_name_full"])
    .sort_values(["uc_campus", "year"])
)

uc_feat.to_csv(DATA_PROC / "uc_scorecard_features.csv", index=False)

# =====================================================
# 2. cc2uc_major
# =====================================================

maj_cols = ["Year", "UC", "Field", "Major", "Enrolls", "CC"]
maj_raw = pd.read_csv(DATA_RAW / "cc2uc_major.csv", usecols=maj_cols)

maj_feat = (
    maj_raw.rename(
        columns={
            "Year": "year_acad",
            "UC": "uc_raw",
            "CC": "cc_name_raw",
            "Field": "field",
            "Major": "major",
            "Enrolls": "enrolls",
        }
    )
    .assign(
        year=lambda d: d["year_acad"].apply(academic_to_calendar_year).astype("Int64"),
        cc_name=lambda d: d["cc_name_raw"].apply(normalize_cc),
        uc_campus=lambda d: d["uc_raw"].apply(normalize_uc),
    )
    .drop(columns=["year_acad", "cc_name_raw", "uc_raw"])
)

maj_feat.to_csv(DATA_PROC / "cc2uc_major_features.csv", index=False)

maj_summary = (
    maj_feat.groupby(["cc_name", "uc_campus", "year"], as_index=False)
    .agg(total_major_enrolls=("enrolls", "sum"))
)

maj_summary.to_csv(DATA_PROC / "cc2uc_major_summary.csv", index=False)

# =====================================================
# 3. cc2uc_3status：gender + ethnicity
# =====================================================

status_frames = []
status_specs = [
    ("cc2uc_3status_gnd.csv", "Gender", "gender"),
    ("cc2uc_3status_eth.csv", "Ethnicity", "ethnicity"),
]

for fname, group_col, group_type in status_specs:
    df_raw = pd.read_csv(DATA_RAW / fname)

    df_clean = (
        df_raw.rename(
            columns={
                "City": "cc_city",
                "County": "cc_county",
                "School": "cc_name_raw",
                "UC": "uc_raw",
                "Year": "year_acad",
                "Count": "scope",     # App / Adm / Enr
                "Num": "n_students",
                group_col: "group_value",
            }
        )
        .assign(
            group_type=group_type,
            cc_name=lambda d: d["cc_name_raw"].apply(normalize_cc),
            uc_campus=lambda d: d["uc_raw"].apply(normalize_uc),
            year=lambda d: d["year_acad"].apply(academic_to_calendar_year).astype("Int64"),
        )
        .loc[
            :,
            [
                "cc_city",
                "cc_county",
                "cc_name",
                "uc_campus",
                "year",
                "scope",
                "n_students",
                "group_type",
                "group_value",
            ],
        ]
    )

    status_frames.append(df_clean)

status_long = pd.concat(status_frames, ignore_index=True)
status_long.to_csv(DATA_PROC / "cc2uc_3status_long.csv", index=False)

# =====================================================
# 4. overall melt（App / Adm / Enr）
# =====================================================

eth_raw = pd.read_csv(DATA_RAW / "cc2uc_3status_eth.csv")

eth_clean = (
    eth_raw.rename(
        columns={
            "City": "cc_city",
            "County": "cc_county",
            "School": "cc_name_raw",   
            "UC": "uc_raw",           
            "Year": "year_acad",       
            "Count": "scope",          # App / Adm / Enr
            "Num": "n_students",      
            "Ethnicity": "ethnicity",
        }
    )
    .assign(
        cc_name=lambda d: d["cc_name_raw"].apply(normalize_cc),
        uc_campus=lambda d: d["uc_raw"].apply(normalize_uc),
        year=lambda d: d["year_acad"].apply(academic_to_calendar_year).astype("Int64"),
    )
)

eth_all = eth_clean.query("ethnicity == 'All'").copy()

overall = (
    eth_all
    .groupby(["cc_name", "uc_campus", "year", "scope"], as_index=False)
    .agg(n_students=("n_students", "sum"))
)

melt = (
    overall.pivot(
        index=["cc_name", "uc_campus", "year"],
        columns="scope",
        values="n_students",
    )
    .reset_index()
    .rename(
        columns={
            "App": "n_app",
            "Adm": "n_admit",
            "Enr": "n_enroll",
        }
    )
)

for col in ["n_app", "n_admit", "n_enroll"]:
    melt[col] = pd.to_numeric(melt[col], errors="coerce")

melt = melt[melt["n_admit"].notna() & (melt["n_admit"] > 0)]

melt["melt_count"] = melt["n_admit"] - melt["n_enroll"]
melt["melt_rate"] = melt["melt_count"] / melt["n_admit"]

melt.to_csv(DATA_PROC / "cc2uc_melt_overall.csv", index=False)

# =====================================================
# 5. gender / ethnicity
# =====================================================
# ----- gender composition -----
gender_enr = (
    status_long
    .query("group_type == 'gender' and scope == 'Enr' and group_value != 'All'")
    .groupby(["cc_name", "uc_campus", "year", "group_value"], as_index=False)
    .agg(n_enroll=("n_students", "sum"))
)

gender_wide = gender_enr.pivot(
    index=["cc_name", "uc_campus", "year"],
    columns="group_value",
    values="n_enroll",
).reset_index()

gender_cols = [c for c in gender_wide.columns if c not in ["cc_name", "uc_campus", "year"]]

for c in gender_cols:
    gender_wide[c] = pd.to_numeric(gender_wide[c], errors="coerce")

gender_wide["gender_total_enr"] = gender_wide[gender_cols].sum(axis=1)

for c in gender_cols:
    col_safe = str(c).lower().replace(" ", "_")
    gender_wide[f"share_gender_{col_safe}"] = gender_wide[c] / gender_wide["gender_total_enr"]

gender_feat = gender_wide[
    ["cc_name", "uc_campus", "year"]
    + [c for c in gender_wide.columns if c.startswith("share_gender_")]
]

share_gender_cols = [c for c in gender_feat.columns if c.startswith("share_gender_")]
gender_feat[share_gender_cols] = gender_feat[share_gender_cols].fillna(0)

gender_feat.to_csv(DATA_PROC / "cc2uc_gender_features.csv", index=False)

# ----- ethnicity composition -----
eth_enr = (
    status_long
    .query("group_type == 'ethnicity' and scope == 'Enr' and group_value != 'All'")
    .groupby(["cc_name", "uc_campus", "year", "group_value"], as_index=False)
    .agg(n_enroll=("n_students", "sum"))
)

eth_wide = eth_enr.pivot(
    index=["cc_name", "uc_campus", "year"],
    columns="group_value",
    values="n_enroll",
).reset_index()

eth_cols = [c for c in eth_wide.columns if c not in ["cc_name", "uc_campus", "year"]]

for c in eth_cols:
    eth_wide[c] = pd.to_numeric(eth_wide[c], errors="coerce")

eth_wide["eth_total_enr"] = eth_wide[eth_cols].sum(axis=1)

for c in eth_cols:
    col_safe = (
        str(c)
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )
    eth_wide[f"share_eth_{col_safe}"] = eth_wide[c] / eth_wide["eth_total_enr"]

eth_feat = eth_wide[
    ["cc_name", "uc_campus", "year"]
    + [c for c in eth_wide.columns if c.startswith("share_eth_")]
]

share_eth_cols = [c for c in eth_feat.columns if c.startswith("share_eth_")]
eth_feat[share_eth_cols] = eth_feat[share_eth_cols].fillna(0)

eth_feat.to_csv(DATA_PROC / "cc2uc_ethnicity_features.csv", index=False)
# =====================================================
# cc_uc_drive_distance
# =====================================================

dist_raw = pd.read_csv(DATA_RAW / "cc_uc_drive_distances.csv")

dist_feat = (
    dist_raw
    .rename(columns={
        "cc_name": "cc_name_raw",
        "uc_name": "uc_name_full",
    })
    .assign(
        cc_name=lambda d: d["cc_name_raw"].apply(normalize_cc),
        uc_campus=lambda d: d["uc_name_full"].apply(normalize_uc),
    )
    .loc[
        :,
        ["cc_name", "uc_campus", "distance_miles", "duration_hours"],
    ]
    .rename(
        columns={
            "distance_miles": "cc_uc_distance_miles",
            "duration_hours": "cc_uc_drive_hours",
        }
    )
)

dist_feat.to_csv(DATA_PROC / "cc_uc_distance_features.csv", index=False)

# =====================================================
# StudentCitizenshipStatus
# =====================================================
cit_raw = pd.read_csv(DATA_RAW / "StudentCitizenshipStatus.csv", encoding="latin1")

cit_raw = cit_raw.rename(columns={"Unnamed: 0": "area", "Unnamed: 1": "status"})
cit_raw = cit_raw.iloc[1:].reset_index(drop=True)

cit_raw["area_filled"] = cit_raw["area"].ffill()

cit_raw["cc_name"] = cit_raw["area_filled"].apply(normalize_cc)

cit_sel = cit_raw[cit_raw["status"].astype(str).str.contains("Permanent Resident", na=False)].copy()

count_cols = [
    c
    for c in cit_sel.columns
    if isinstance(c, str)
    and c.startswith("Fall ")
    and ".1" not in c       
]

cit_long = cit_sel.melt(
    id_vars=["cc_name"],
    value_vars=count_cols,
    var_name="term",
    value_name="count_raw",
)

cit_long["year"] = cit_long["term"].str.extract(r"(\d{4})")[0].astype("Int64")

cit_long["count"] = (
    cit_long["count_raw"]
    .astype(str)
    .str.replace(",", "", regex=False)
)
cit_long["count"] = pd.to_numeric(cit_long["count"], errors="coerce")

cit_long = cit_long.dropna(subset=["year", "count"])

perm_feat = (
    cit_long
    .groupby(["cc_name", "year"], as_index=False)
    .agg(cc_perm_resident_count=("count", "sum"))
)

perm_feat.to_csv(DATA_PROC / "cc_perm_resident_by_year.csv", index=False)

# =====================================================
# 6. StudentFinAidSumm
# =====================================================

aid_raw = pd.read_csv(DATA_RAW / "StudentFinAidSumm.csv", encoding="latin1")

aid_raw = aid_raw.rename(columns={"Unnamed: 0": "area", "Unnamed: 1": "label"})

aid_raw["area_filled"] = aid_raw["area"].ffill()

aid_raw["cc_name"] = (
    aid_raw["area_filled"]
    .str.replace(" Total", "", regex=False)
    .apply(normalize_cc)     
)

wanted_labels = [
    "California College Promise Grant Total",
    "Grants Total",
    "Loans Total",
    "Scholarship Total",
]
aid_sel = aid_raw[aid_raw["label"].isin(wanted_labels)].copy()

amount_cols = [
    c for c in aid_sel.columns
    if isinstance(c, str) and c.startswith("Annual ") and c.endswith(".2")
]

aid_long = aid_sel.melt(
    id_vars=["cc_name", "label"],
    value_vars=amount_cols,
    var_name="year_col",
    value_name="amount_raw",
)

aid_long["year_acad"] = aid_long["year_col"].str.extract(r"Annual (\d{4}-\d{4})")[0]
aid_long["year"] = aid_long["year_acad"].apply(academic_to_calendar_year).astype("Int64")

aid_long["amount"] = (
    aid_long["amount_raw"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
)
aid_long["amount"] = pd.to_numeric(aid_long["amount"], errors="coerce")

aid_long = aid_long.dropna(subset=["year", "amount"])

aid_feat = aid_long.pivot_table(
    index=["cc_name", "year"],
    columns="label",
    values="amount",
    aggfunc="sum",
).reset_index()

aid_feat = aid_feat.rename(
    columns={
        "California College Promise Grant Total": "cc_aid_promise_amt",
        "Grants Total": "cc_aid_grants_amt",
        "Loans Total": "cc_aid_loans_amt",
        "Scholarship Total": "cc_aid_scholarship_amt",
    }
)

aid_feat.to_csv(DATA_PROC / "cc_aid_features_by_year.csv", index=False)

# =====================================================
# 7. merge master panel
# =====================================================

panel = melt.copy()

# CC / UC / major / gender / ethnicity
panel = panel.merge(cc_feat, on=["cc_name", "year"], how="left")
panel = panel.merge(uc_feat, on=["uc_campus", "year"], how="left")
panel = panel.merge(maj_summary, on=["cc_name", "uc_campus", "year"], how="left")
panel = panel.merge(gender_feat, on=["cc_name", "uc_campus", "year"], how="left")
panel = panel.merge(eth_feat, on=["cc_name", "uc_campus", "year"], how="left")

# ---- CC-UC ----
panel = panel.merge(
    dist_feat,
    on=["cc_name", "uc_campus"],
    how="left",
)


panel = panel.merge(
    aid_feat[
        [
            "cc_name",
            "year",
            "cc_aid_promise_amt",
            "cc_aid_grants_amt",
            "cc_aid_loans_amt",
            "cc_aid_scholarship_amt",
        ]
    ],
    on=["cc_name", "year"],
    how="left",
)

panel = panel.merge(
    perm_feat[
        [
            "cc_name",
            "year",
            "cc_perm_resident_count",
        ]
    ],
    on=["cc_name", "year"],
    how="left",
)

cols_to_drop = [
    "cc_ftft_pell_rate",
    "cc_ftft_fedloan_rate",
    "cc_pell_rate",
    "cc_fedloan_rate",
    "uc_admit_rate_overall",
    "uc_ftft_pell_rate",
    "uc_ftft_fedloan_rate",
    "uc_pell_rate",
    "uc_fedloan_rate",
    "share_gender_other",
    "share_gender_unknown",
    "share_eth_domestic_unknown",
    "total_major_enrolls"
]

panel = panel.drop(columns=cols_to_drop, errors="ignore")

# =====================================================
#  cc_*/uc_*  enrollment & cost imputation
# =====================================================

#Median
cc_cols = ["cc_ug_enroll_12m", "cc_coa_ay"] 

for col in cc_cols:
    if col in panel.columns:
        panel[col] = (
            panel
            .groupby("cc_name")[col]
            .transform(lambda s: s.fillna(s.median()))
        )
        panel[col] = panel[col].fillna(panel[col].median())

# Linear

def backcast_uc_with_trend(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    df = panel.copy()

    for campus, grp in df.groupby("uc_campus"):
        grp = grp.sort_values("year")
        years = grp["year"].to_numpy()
        values = grp[col].to_numpy(dtype="float64")

        series = pd.Series(values, index=years)
        series_interp = series.interpolate()  

        mask_obs = series_interp.notna().to_numpy()

        x = years[mask_obs].astype(float)
        y = series_interp.to_numpy()[mask_obs].astype(float)

        a, b = np.polyfit(x, y, 1)
        filled = series_interp.to_numpy().copy()

        year_min_obs = x.min()
        early_mask = years < year_min_obs
        if early_mask.any():
            filled[early_mask] = a * years[early_mask] + b

        df.loc[grp.index, col] = filled

    return df

for col in ["uc_ug_enroll_12m", "uc_coa_ay"]:
    panel = backcast_uc_with_trend(panel, col)

for col in ["cc_aid_loans_amt", "cc_aid_scholarship_amt"]:
    if col in panel.columns:
        panel[col] = panel[col].fillna(0)

panel.to_csv(DATA_PROC / "melt_panel_master.csv", index=False)