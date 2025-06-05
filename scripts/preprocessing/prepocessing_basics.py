import pandas as pd


# %% import data

pd.set_option('display.max_columns', None)

files = {
    "CH": "raw_data/ccs_conjoint_CH_240225_1004.csv",
    "CN": "raw_data/ccs_conjoint_CN_240225_1752.csv"
}

dataframes = {}

for country, file_name in files.items():
    # read the file, skipping the first two rows
    df = pd.read_csv(file_name, skiprows=[1, 2])
    df["StartDate"] = pd.to_datetime(df["StartDate"])
    
    # store the processed dataframe in the dictionary
    dataframes[country] = df

# %% filter data

# survey launch time
cutoff = pd.Timestamp("2025-02-13 10:00:00")

for country, df in dataframes.items():
    # filter out rows where DistributionChannel is 'preview' or screened_out is True
    df = df[(df['DistributionChannel'] != 'preview') & (df['Finished'] != False)]
    df = df[(df['Q_TerminateFlag'] != "QuotaMet")]
    df = df[(df['Q_TerminateFlag'] != "Screened")]

    # filter out testing IDs using the cutoff time of official launch
    df = df[df["StartDate"] >= cutoff]
    
    # update the dataframe in the dictionary
    dataframes[country] = df

# %% fix typos and delete empty columns

for country, df in dataframes.items():
    cols_to_drop = [col for col in df.columns if col.startswith("Recipient")]
    dataframes[country] = df.drop(columns=cols_to_drop)

# dataframes["US"]["personal_income"] = dataframes["US"]["personal_income"].replace("15k_35k", "15k_25k")
# dataframes["CH"] = dataframes["CH"].rename(columns={"recent_flights": "flying_recent_number"})
# dataframes["CN"] = dataframes["CN"].rename(columns={"recent_flights": "flying_recent_number"})

# %% add response IDs and other coolumns

# get the total number of rows across all dataframes
total_rows = sum(len(df) for df in dataframes.values())

# initialize an ID counter
id_counter = 1

for country, df in dataframes.items():
    df['id'] = range(id_counter, id_counter + len(df))
    id_counter += len(df)
    dataframes[country] = df

ch_df = dataframes["CH"]
cn_df = dataframes["CN"]

# %% save clean data

ch_df.to_csv("data/data_clean_untranslated_ch.csv")
cn_df.to_csv("data/data_clean_untranslated_cn.csv")


# %%
