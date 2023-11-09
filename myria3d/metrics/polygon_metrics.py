import pandas as pd
import geopandas as gpd
import pandas as pd
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

PATCHES = (
    "/mnt/store-lidarhd/projet-LHD/IA/BDForet/Data/PureForestID/metadata/PureForestID-patches.gpkg"
)
CLASS_CODE2CLASS_NAME = {
    0: "FF1-09-09_Hêtre",
    1: "FF1-10-10_Châtaignier",
    2: "FF1-14-14_Robinier",
    3: "FF1G01-01_Chêne_décidus",
    4: "FF1G06-06_Chêne_sempervirent",
    5: "FF2-51-51_Pin_maritime",
    6: "FF2-52-52_Pin_sylvestre",
    7: "FF2-57-57_Pin_alep",
    8: "FF2-63-63-Mélèze",
    9: "FF2-64-64_Douglas",
    10: "FF2G53-53_Pin_laricio_Pin_noir",
    11: "FF2G61-61_Sapin-Epicéa",
}


def load_patches(patches_file):
    gdf = gpd.read_file(patches_file)
    gdf = gdf[gdf.split.isin(["test", "val"])]
    gdf = gdf[
        [
            "bdforet_identifier",
            "patch_num",
            "bdforet_cat_code",
            "bdforet_cat_name",
            "geometry",
        ]
    ]
    return gdf


def load_predictions(prediction_file):
    df = pd.read_csv(prediction_file)
    df["targets"] = df["targets"].apply(lambda t: CLASS_CODE2CLASS_NAME[t])
    df["preds"] = df["preds"].apply(lambda t: CLASS_CODE2CLASS_NAME[t])
    return df


def make_pivot_table(df):
    groups = (
        df.groupby(["bdforet_identifier", "bdforet_cat_code", "targets", "preds"])
        .size()
        .reset_index()
    )
    pivot = (
        groups.pivot(index=["bdforet_identifier", "targets"], columns="preds", values=0)
        .fillna(0)
        .astype(int)
    )
    pivot["consensus"] = pivot.idxmax(axis=1)  # consensus of predictions.
    pivot = pivot.reset_index()
    pivot["accurate"] = pivot["consensus"] == pivot["targets"]
    return pivot


def make_polygon_cm(pivot):
    # Confusion matrix at polygon level
    counts = (
        pivot.groupby(["targets", "consensus"]).size().reset_index().rename(columns={0: "count"})
    )
    for class_name_targets in CLASS_CODE2CLASS_NAME.values():
        for class_name_preds in CLASS_CODE2CLASS_NAME.values():
            if not len(
                counts.query(
                    f"targets == '{class_name_targets}' & consensus == '{class_name_preds}'"
                )
            ):
                # print(class_name_targets, class_name_preds)
                counts = pd.concat(
                    [
                        counts,
                        pd.DataFrame(
                            data=[
                                {
                                    "targets": class_name_targets,
                                    "consensus": class_name_preds,
                                    "count": 0,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
    cm = counts.pivot(index="targets", columns="consensus", values="count").fillna(0).astype(int)

    return cm


def make_accuracy_table(pivot):
    # Class accuracy
    accuracy_table = (
        pivot.groupby("targets").agg({"accurate": "mean", "bdforet_identifier": "size"}).round(2)
    )
    accuracy_table = accuracy_table.rename(
        columns={"accurate": "accuracy", "bdforet_identifier": "num_bdforet_polygons"}
    )
    # Global accuracy
    OK = len(pivot[pivot.accurate])
    NOK = len(pivot[~pivot.accurate])
    accuracy = OK / (OK + NOK)
    global_accuracy_row = {
        "targets": "All classes",
        "accuracy": accuracy,
        "num_bdforet_polygons": accuracy_table["num_bdforet_polygons"].sum(),
    }
    accuracy_table = pd.concat(
        [pd.DataFrame(data=[global_accuracy_row]), accuracy_table.reset_index()]
    ).reset_index(drop=True)
    return accuracy_table


def save_polygon_cm(cm, cm_polygon_path):
    fig, ax = plt.subplots(figsize=(31, 31))
    ConfusionMatrixDisplay(cm.values, display_labels=cm.index.values).plot(ax=ax)
    plt.xticks(
        rotation=40, rotation_mode="anchor", ha="right"
    )  # Rotates X-Axis Ticks by 45-degrees
    plt.tight_layout()
    plt.savefig(cm_polygon_path)
    print(f"Saved cm_polygon_path to {cm_polygon_path}")


def make_polygon_metrics(prediction_file: str = "predictions.csv", log_dir=None):
    """prediction_file [patch_id, targets, preds]"""
    if log_dir is None:
        log_dir = Path(prediction_file).parent.resolve()

    df = load_predictions(prediction_file)
    print("Num predicted patches: ", len(df))
    gdf = load_patches(PATCHES)
    merge = gdf.merge(df, on="patch_num", how="inner")
    print("Num predicted patches after merging with geometry information: ", len(df))
    merge["accurate"] = merge["targets"] == merge["preds"]
    print("Sanity check - Accuracy at patch level is ", merge["accurate"].mean())
    pivot = make_pivot_table(merge)
    pivot.to_csv(log_dir / "pivot-table-predictions-by-polygon.csv", index=False)

    cm = make_polygon_cm(pivot)
    cm_polygon_path = log_dir / "CM-by-polygon.png"
    save_polygon_cm(cm, cm_polygon_path)
    print(f"Saved cm_polygon to {cm_polygon_path}")

    accuracy_table = make_accuracy_table(pivot)
    accuracy_table_path = log_dir / "accuracy-table.csv"
    accuracy_table.to_csv(accuracy_table_path)
    print(f"Saved accuracy_table_path to {accuracy_table_path}")
    return cm_polygon_path
