import logging
from datetime import date
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import pandas as pd
from configs.lookup import BQ_PROJECT, BQ_DATASET, get_template_context

logger = logging.getLogger(__name__)


def render(anchor_date: str = date.today().isoformat(), use_case: str = None) -> pd.DataFrame:
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{use_case}_{anchor_date.replace('-', '_')}"
    client = bigquery.Client(project=BQ_PROJECT)

    try:
        # Reuse cached BQ table if this use_case + anchor_date was already run
        table = client.get_table(table_id)
        customers = client.list_rows(table).to_dataframe(create_bqstorage_client=False)
        logger.info("Using cached BQ table: %s", table_id)
    except NotFound:
        env = Environment(loader=FileSystemLoader("templates"))
        try:
            template = env.get_template(f"{use_case}_query")
        except TemplateNotFound:
            template = env.get_template("query")
        query = template.render(**get_template_context(use_case, anchor_date))
        job_config = bigquery.QueryJobConfig(
            destination=table_id,
            write_disposition="WRITE_TRUNCATE",
        )
        client.query(query, job_config=job_config).result()
        logger.info("Results written to BQ table: %s", table_id)
        customers = client.list_rows(table_id).to_dataframe(create_bqstorage_client=False)

    logger.info(
        "%d customers loaded | %d unique | %d conversions",
        len(customers),
        customers["customer_id"].nunique(),
        customers["converted"].sum(),
    )
    return customers
