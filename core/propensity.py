import argparse
import logging
from datetime import date

from .render import render
from core.accuracy import score as accuracy_score
from models.logistic_regression import train, score, analyze
from configs.lookup import FOOD_FPS


def main(anchor_date: str, use_case: str, mode: str):
    logger = logging.getLogger(__name__)
    logger.info("Starting | use_case=%s | anchor_date=%s | mode=%s", use_case, anchor_date, mode)

    df = render(anchor_date=anchor_date, use_case=use_case)

    if mode == "train":
        y_test, y_pred, y_proba = train(df, anchor_date, use_case)
        fps = FOOD_FPS if use_case == "food" else None
        accuracy_score("model", y_test, y_pred, y_proba, anchor_date, use_case, fps=fps)

    elif mode == "score":
        score(df, anchor_date, use_case)

    elif mode == "analyze":
        analyze(df, anchor_date, use_case)

    logger.info("Done | use_case=%s | anchor_date=%s | mode=%s", use_case, anchor_date, mode)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-date", default=date.today().isoformat(), help="Anchor date for data extraction (format: YYYY-MM-DD)")
    parser.add_argument("--use-case", default="beauty", choices=["beauty", "food", "shops"])
    parser.add_argument("--mode", default="train", choices=["train", "score", "analyze"])

    args = parser.parse_args()
    main(args.anchor_date, args.use_case, args.mode)
