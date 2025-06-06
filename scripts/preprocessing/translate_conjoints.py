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

def reshape_conjoint_to_long(df, respondent_id_col = None, country = "CH"):
    """
    Reshape randomized conjoint data into long format with
    one row per respondent, per task, per package (p1/p2)
    Columns are: respondent ID (optional), task, package, attr_[name]

    Parameters:
    df : pd.DataFrame wide-format dataframe
    respondent_id_col : str or None for respondent identifiers
    """
    n_rows = df.shape[0]
    name_cols = [col for col in df.columns if re.match(r"c\d+_atr\d+_name", col)]
    long_data = []

    for name_col in name_cols:
        match = re.match(r"c(\d+)_atr(\d+)_name", name_col)
        if not match:
            continue
        task, atr = match.groups()
        task = int(task)

        p1_col = f"c{task}_atr{atr}_p1"
        p2_col = f"c{task}_atr{atr}_p2"
        if p1_col not in df.columns or p2_col not in df.columns:
            continue

        choose_col = f"{task}_conjoint_choose12"
        plan1_col = f"{task}_conjoint_plan1"
        plan2_col = f"{task}_conjoint_plan2"

        for i in range(n_rows):
            attr = df[name_col].iloc[i]
            if pd.isna(attr):
                continue

            chosen_plan = df[choose_col].iloc[i] if choose_col in df.columns else None
            plan1_support = df[plan1_col].iloc[i] if plan1_col in df.columns else None
            plan2_support = df[plan2_col].iloc[i] if plan2_col in df.columns else None

            if pd.isna(chosen_plan):
                chosen_plan = None
            if isinstance(plan1_support, str):
                plan1_support = 1 if plan1_support == "In favor" else 0
            if isinstance(plan2_support, str):
                plan2_support = 1 if plan2_support == "In favor" else 0

            row_base = {
                "task": task,
                "chosen_plan": chosen_plan
            }
            if respondent_id_col:
                row_base["id"] = df[respondent_id_col].iloc[i]

            # Package 1
            long_data.append({
                **row_base,
                "package": "1",
                f"attr_{attr}": df[p1_col].iloc[i],
                "chosen": 1 if chosen_plan == "Plan 1" else 0,
                "supported": plan1_support
            })

            # Package 2
            long_data.append({
                **row_base,
                "package": "2",
                f"attr_{attr}": df[p2_col].iloc[i],
                "chosen": 1 if chosen_plan == "Plan 2" else 0,
                "supported": plan2_support
            })

    long_df = pd.DataFrame(long_data)

    if long_df.empty:
        print("Warning: No rows created. Check for missing attribute columns.")
        return long_df

    index_cols = ["task", "package"]
    if respondent_id_col:
        index_cols.insert(0, "id")

    long_df = long_df.groupby(["id", "task", "package"], as_index=False).first()

    attr_cols = [col for col in long_df.columns if col.startswith("attr_")]
    other_cols = [col for col in long_df.columns if not col.startswith("attr_")]
    long_df = long_df[other_cols + attr_cols]

    return long_df

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


