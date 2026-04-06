import pandas as pd
def feature_engineer(df):

    features = pd.DataFrame({
        "customer_id": df["customer_id"].unique()
    })
    features = features.set_index("customer_id")

    is_female = df["is_female"] = (df["gender"] == "Female").groupby(df["customer_id"]).max().astype(int).fillna(0)

    # Basket Affinity
    bought_coffee = (
        df["main_category_name"].str.lower().eq("breadfast coffee")
        .groupby(df["customer_id"])
        .any()
        .fillna(0)
        .astype(int)
    )

    maximum_item_price = (
    df["order_subtotal"].groupby(df["customer_id"]) \
        .max()
    )

    average_item_price = (
    df["order_subtotal"].groupby(df["customer_id"]) \
        .mean()
    )

    item_price_volatility = (
    df["order_subtotal"].groupby(df["customer_id"]) \
        .std()
    )

    item_spend_flexibility = maximum_item_price.astype(float)/average_item_price.astype(float)


    bought_rte = (
        df["main_category_name"].str.lower().eq("ready to eat")
        .groupby(df["customer_id"])
        .any()
        .fillna(0)
        .astype(int)
    )

    last_order_df = df[df["rank_desc"] == 1]

    last_order_contains_coffee = (
        last_order_df["main_category_name"].str.lower().eq("breadfast coffee")
        .groupby(last_order_df["customer_id"])
        .any()
        .fillna(0)
        .astype(int)
    )

    last_order_contains_rte = (
        last_order_df["main_category_name"].str.lower().eq("ready to eat")
        .groupby(last_order_df["customer_id"])
        .any()
        .fillna(0)
        .astype(int)
    )

    cats_per_order = (
        df.groupby(["customer_id", "order_id"])["main_category_name"]
        .nunique()
        .reset_index(name="cats_in_order")
    )
    avg_cats_customer = (
        cats_per_order
        .groupby("customer_id")["cats_in_order"]
        .mean()
    )

    basket_per_order = (
        df.groupby(["customer_id", "order_id", "rank_desc"])["order_product_quantity"]
        .sum()
        .reset_index(name="basket_size")
    )
    basket_stats = (
        basket_per_order
        .groupby("customer_id")["basket_size"]
        .agg(
            min_basket_size="min",
            max_basket_size="max",
            avg_basket_size="mean"
        )
    )



    last_order_basket_size = (df[df["rank_desc"]==1].groupby(["customer_id"])["order_product_quantity"]
    .sum())

    last_orders = (
        df[df["rank_desc"] == 1]
        .sort_values("order_date")
        .drop_duplicates("customer_id")
        [["customer_id", "address_label_tag"]]
    )

    is_last_order_from_office = (
        last_orders
        .set_index("customer_id")["address_label_tag"]
        .str.lower()
        .ne("household")
        .astype(int)
    )

    orders = (
        df[["customer_id", "order_id", "order_date", "address_label_tag"]]
        .drop_duplicates(["customer_id", "order_id"])
    )

    orders["is_office_order"] = (
        orders["address_label_tag"]
        .str.lower()
        .ne("household")
        .astype(int)
    )

    office_ratio = (
        orders
        .groupby("customer_id")["is_office_order"]
        .mean()
    )

    weekend_spend_ratio = (
        df[df["dow"].isin([4,5])].groupby("customer_id")["order_subtotal"].sum().astype(float)
        / df.groupby("customer_id")["order_subtotal"].sum().astype(float)
    )


    # avg_gap = (
    #     orders.sort_values(["customer_id", "order_date"]).groupby("customer_id")["order_date"]
    #     .diff()
    #     .dt.days
    # )


    favorite_dow = df.groupby("customer_id")["dow"].agg(lambda x: x.mode()[0])

    favorite_hour = df.groupby("customer_id")["hour"].agg(lambda x: x.mode()[0])

    features["customer_id"] =  df.groupby("customer_id")["customer_id"].max().fillna(0).astype(int)
    features["converted"] =  df.groupby("customer_id")["converted"].max().fillna(0).astype(int)
    features["bought_coffee"] = bought_coffee
    features["bought_rte"] = bought_rte
    features["last_order_contains_coffee"] = last_order_contains_coffee
    features["last_order_contains_rte"] = last_order_contains_rte
    features["min_basket_size"] = basket_stats["min_basket_size"]
    features["max_basket_size"] = basket_stats["max_basket_size"]
    features["abs"] = basket_stats["avg_basket_size"]
    features["average_cats_per_order"] = avg_cats_customer
    features["last_order_basket_size"] = last_order_basket_size
    features["is_last_order_from_office"] = is_last_order_from_office
    features["office_ratio"] = office_ratio
    features["maximum_item_price"] = maximum_item_price
    features["item_spend_flexibility"] = item_spend_flexibility
    features["weekend_spend_ratio"] = weekend_spend_ratio
    features["is_female"] = is_female
    features["favorite_dow"] = favorite_dow
    features["favorite_hour"] = favorite_hour
    # features["average_days_between_orders"] = avg_gap.mean()


    return features