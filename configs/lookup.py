BQ_PROJECT = "bf-data-dev-qz06"
BQ_DATASET = "test_dinayasser"

FOOD_FPS = ["New Cairo FP #1", "New Cairo FP #2", "New Cairo FP #3", "New Cairo FP #4", "New Cairo FP #5", "New Cairo FP #6", "New Cairo FP #7"]

DROP_COLS = ["customer_id", "converted", "unique_views_in_cat", "primary_fp"]

VERTICAL_LOOKUP = {
    "food": "order_vertical = 'restaurant'",
    "beauty": "main_category_name = 'Fragrances & Beauty'",
    "shops": "main_category_name = 'Shops'",
}

def get_template_context(use_case: str, anchor_date: str, fps: list = None) -> dict:
    return {
        "anchor_date": anchor_date,
        "use_case_filter": VERTICAL_LOOKUP[use_case],
        "fps": fps if fps is not None else (FOOD_FPS if use_case == "food" else None),
    }

# Features to include per use case. Set to None to use all available features.
FEATURES = {
    "food": [
        "days_since_last_order",
        "bought_coffee",
        "office_ratio",
        "min_basket_size",
        "basket_size_stddev",
        "perc_orders_with_discount",
        "order_value_stddev",
        "bought_rte",
        "aov",
        "max_order_value",
        "max_basket_size",
        "last_order_contains_coffee",
        "rte_ratio",
        "last_order_basket_size",
        "last_order_value",
        "is_last_order_weekday",
        "is_last_order_peak_hours",
        "orders_per_week",
    ],
    "beauty": None,
    "shops": None,
}
