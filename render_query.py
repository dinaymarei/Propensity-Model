from google.cloud import bigquery
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from pathlib import Path
from configs.lookup import VERTICAL_LOOKUP

def render(anchor_date, use_case):
    if (not Path("input/cached_input_data_.csv").exists()):
        project_id = "followbreadfast"
        client = bigquery.Client(project=project_id)

        env = Environment(loader=FileSystemLoader("templates"))

        template = env.get_template("query")
        query = template.render(anchor_date = anchor_date, use_case_filter =VERTICAL_LOOKUP[use_case]) 

        customers = client.query(query).to_dataframe()

        customers.to_csv("input/cached_input_data.csv")
    else: 
        customers = pd.read_csv("input/cached_input_data.csv")

    return customers
