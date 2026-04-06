from google.cloud import bigquery
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from pathlib import Path
from configs.lookup import BQ_PROJECT, VERTICAL_LOOKUP


def render(anchor_date: str, use_case: str, fps: list = None) -> pd.DataFrame:
    cache_path = Path(f"input/{use_case}/{anchor_date}.csv")

    if not cache_path.exists():
        client = bigquery.Client(project=BQ_PROJECT)
        env = Environment(loader=FileSystemLoader("templates"))

        template_name = "food_query" if use_case == "food" else "query"
        template = env.get_template(template_name)
        query = template.render(
            anchor_date=anchor_date,
            use_case_filter=VERTICAL_LOOKUP[use_case],
            fps=fps,
        )

        customers = client.query(query).to_dataframe()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        customers.to_csv(cache_path, index=False)
    else:
        customers = pd.read_csv(cache_path)
        print(f"Using cached data from: {cache_path}")

    if fps is not None:
        customers = customers[customers["primary_fp"].isin(fps)]

    print(f"{len(customers)} customers loaded for use case: {use_case}")
    print(f"{customers['customer_id'].nunique()} unique customers")
    print(f"{customers['converted'].sum()} conversions")

    return customers
