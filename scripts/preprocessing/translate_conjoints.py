import pandas as pd
import re

# %% 

ch_df = pd.read_csv("data/data_clean_untranslated_ch.csv")
cn_df = pd.read_csv("data/data_clean_untranslated_cn.csv")

# %% 
def apply_mapping(df, mapping_dict, column_pattern=None):
    """
    Apply a mapping to columns in the DataFrame based on a dictionary, with
    an optional string or list of strings column_pattern to filter column names.
    If None, all columns are considered.
    """
    
     # turn column_pattern into a list it already isn't
    if isinstance(column_pattern, str):
        column_patterns = [column_pattern]
    elif isinstance(column_pattern, list) and all(isinstance(pat, str) for pat in column_pattern):
        column_patterns = column_pattern
    elif column_pattern is None:
        column_patterns = []
    else:
        raise ValueError("column_pattern should be a string, list of strings, or None.")
    
    # identify columns to apply the mapping
    if column_patterns:
        columns_to_map = [col for col in df.columns if any(pat in col for pat in column_patterns)]
    else:
        columns_to_map = df.columns
    
    # apply mapping to the identified columns
    for column in columns_to_map:
        df[column] = df[column].replace(mapping_dict)
    
    return df

def reshape_conjoint_to_long(df, respondent_id_col = None):
    """
    Reshape randomized conjoint data into long format.
    One row per respondent, per task, per package (p1/p2).
    Columns are: respondent ID (optional), task, package, attr_[name]

    Parameters:
    df : pd.DataFrame, Input wide-format dataframe with columns like 
    c1_atr1_name, c1_atr1_p1, etc.
    respondent_id_col : str or None, Optional column name for
    respondent identifiers

    Returns:
    pd.DataFrame
        Long-format dataframe with one row per task/package per respondent.
    """
    n_rows = df.shape[0]
    name_cols = [col for col in df.columns if re.match(r"c\d+_atr\d+_name", col)]
    long_data = []

    for name_col in name_cols:
        match = re.match(r"c(\d+)_atr(\d+)_name", name_col)
        if not match:
            continue
        task, atr = match.groups()
        p1_col = f"c{task}_atr{atr}_p1"
        p2_col = f"c{task}_atr{atr}_p2"

        if p1_col not in df.columns or p2_col not in df.columns:
            continue

        for i in range(n_rows):
            attr = df[name_col].iloc[i]
            if pd.isna(attr):
                continue
            row_base = {}
            if respondent_id_col:
                row_base["id"] = df[respondent_id_col].iloc[i]
            row_base["task"] = int(task)

            # One row per package
            long_data.append({
                **row_base,
                "package": "1",
                f"attr_{attr}": df[p1_col].iloc[i]
            })
            long_data.append({
                **row_base,
                "package": "2",
                f"attr_{attr}": df[p2_col].iloc[i]
            })

    # Combine and pivot to wide-format attributes per row
    long_df = pd.DataFrame(long_data)

    # Group and pivot attributes into columns
    index_cols = ["task", "package"]
    if respondent_id_col:
        index_cols = ["id"] + index_cols

    final_df = long_df.pivot_table(
        index=index_cols,
        aggfunc="first"
    ).reset_index()

    # Flatten multiindex columns if any
    final_df.columns.name = None
    final_df = final_df.rename_axis(columns=None)

    return final_df

# %% translate attribute names

attr_names_dict = {
    "Das Kohlendioxid wird gespeichert in": "vicinity",
    "The carbon dioxide is stored in": "vicinity",
    "Le dioxyde de carbone est stocké dans": "vicinity",
    "二氧化碳被储存在": "vicinity",

    "Dieser Standort wurde gewählt, weil er": "reason",
    "Cet endroit a été choisi parce qu’il": "reason",
    "This location was chosen because it": "reason",
    "选择储存在这里是因为": "reason",

    "CCS wird angewendet bei": "industry",
    "CCS is applied to": "industry",
    "Le CSC est appliqué aux": "industry",
    "碳捕集与封存被应用在": "industry",

    "Das gespeicherte Kohlendioxid stammt aus": "source",
    "The carbon dioxide stored will come from": "source",
    "Le dioxyde de carbone stocké proviendra": "source",
    "储存的二氧化碳来自": "source",

    "Der Zweck des Kohlendioxid-Speicherprojekts ist": "purpose",
    "The purpose of carbon dioxide storage project is": "purpose",
    "L’objectif du projet de stockage du dioxyde de carbone est": "purpose",
    "储存二氧化碳的目的是": "purpose",

    "Die Kosten für die Lagerung trägt": "costs",
    "The cost of storage will be borne by": "costs",
    "Les frais de stockage seront à la charge de": "costs",
    "储存费用由谁来承担": "costs",

    "Bei Entscheidungen über Speicherprojekte werden Sie": "engagement",
    "In decisions about storage projects, you will": "engagement",
    "Dans les décisions concernant les projets de stockage, vous allez": "engagement",
    "在储存项目的决策中，您将": "engagement"
}

ch_df = apply_mapping(ch_df, attr_names_dict, column_pattern='name')
cn_df = apply_mapping(cn_df, attr_names_dict, column_pattern='name')

# %% restructure data

ch_df_long = reshape_conjoint_to_long(ch_df, respondent_id_col = "id")
cn_df_long = reshape_conjoint_to_long(cn_df, respondent_id_col = "id")

# %% 

dfs = [ch_df, cn_df]
long_dfs = []

for df in dfs:
    df_long = reshape_conjoint_to_long(df, respondent_id_col = "id")

    used_cols = [col for col in df.columns if re.match(r"c\d+_atr\d+_(name|p1|p2)", col)]
    meta_cols = [col for col in df.columns if col not in used_cols]
    
    df_meta = df[meta_cols].drop_duplicates()
    df_long = df_long.merge(df_meta, on = "id", how = "left")
    long_dfs.append(df_long)

# %% translate attribute levels


