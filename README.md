# Transfer Pipelines between the California Community College System and the University of California system.

This repository contains code used to generate the final report for STA 221 at UC Davis (Fall 2025 quarter).

## Code Structure

* `./docs/` contains documentation from the data sources used in the project.
* `./ucd_sta_221_project/` is the path to the code.
    * `./ucd_sta_221_project/api/` contains code written to interface with the College Scorecard, CCCCO, and Google Maps APIs.
    * `./ucd_sta_221_project/data_files/` contains raw API responses and data downloads from external sources, e.g., the CCCCO Data Mart.
    * `./ucd_sta_221_project/ml/` contains the code that directly addresses the three questions posed in the paper.
* `./samples.ipynb` is a Jupyter Notebook containing various API calls and code samples from `./ucd_sta_221_project/api/`
