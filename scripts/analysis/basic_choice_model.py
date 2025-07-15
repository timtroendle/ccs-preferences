import pymc as pm 
import pandas as pd
import arviz as az
import numpy as np
import xarray as xr

# %% pymc bug workaround

import pytensor
pytensor.config.cxx = '/usr/bin/clang++'

# %% 

df = pd.read_csv("data/hcm_input.csv")

# %% define lists and translations

attributes = [ 
    "attr_engagement",
    "attr_vicinity",
    "attr_industry",
    "attr_costs",
    "attr_reason",
    "attr_source_purpose"
]

baseline_dict = {
    "attr_engagement": "inform",
    "attr_vicinity": "abroad",
    "attr_industry": "waste incineration",
    "attr_costs": "taxpayer",
    "attr_reason": "sparsely-populated",
    "attr_source_purpose": "domestic"
}

# %% define dummies

# reorder each attribute column by making it categorical with the baseline first
for attr in attributes:
    baseline = baseline_dict[attr]
    df[attr] = pd.Categorical(df[attr], categories=[baseline] + 
                              [level for level in df[attr].unique() if level != baseline], 
                              ordered=True)

# generate dummies with columns in the correct order
dummies = pd.get_dummies(df[attributes], drop_first=False)

# reorder columns to place baseline first for each attribute
ordered_columns = []
for attr in attributes:
    # Collect the columns related to the attribute and put baseline first
    attr_columns = [col for col in dummies.columns if col.startswith(attr)]
    baseline_column = f"{attr}_{baseline_dict[attr]}"
    ordered_columns.append(baseline_column)
    ordered_columns.extend([col for col in attr_columns if col != baseline_column])

# reorder dummies according to ordered columns list
dummies = dummies[ordered_columns]
dummies = dummies.loc[:, ~dummies.columns.duplicated()]

df["framing"] = df["framing"].astype("category")
df["country"] = df["country"].astype("category")

coords = {
    "level": dummies.columns.values,
    "task": np.arange(len(df) // 2),
    "framing": df["framing"].cat.categories,
    "country": df["country"].cat.categories
}


# %% define model

with pm.Model(coords=coords) as bayes_model:
    
    # main effect of attribute levels
    beta = pm.Normal("beta", mu=0, sigma=2, dims="level")

    # framing-specific shift for each attribute level
    delta = pm.Normal("delta", mu=0, sigma=1, dims="level")
    
    # country effect
    gamma = pm.Normal("gamma", mu=0, sigma=1, dims=["country", "level"])
    
    # framing (0 or 1), same for all tasks per participant
    f = pm.Data("f", df.loc[df.package == 1, "framing"].cat.codes.values, dims="task")
    country_idx = df.loc[df.package == 1, "country"].cat.codes.values
    c = pm.Data("c", country_idx, dims = "task")

    # observed choices
    observed_choice_left = pm.Data(
        "observed_choice_left", 
        df.loc[df.package == 1, "chosen"].values, 
        dims="task"
    )

    # attribute dummies
    attribute_levels_left = pm.Data(
        "attribute_levels_left", 
        dummies[df.package == 1].values, 
        dims=["task", "level"]
    )

    attribute_levels_right = pm.Data(
        "attribute_levels_right", 
        dummies[df.package == 2].values, 
        dims=["task", "level"]
    )

    # compute modified coefficients depending on framing
    # this gives beta + delta * framing per task and level
    # adding country effect
    beta_framed = beta + delta * f[:, None] + gamma[c, :]

    # compute utility
    utility_left = pm.Deterministic(
        "utility_left",
        pm.math.sum(attribute_levels_left * beta_framed, axis=1),
        dims="task"
    )

    utility_right = pm.Deterministic(
        "utility_right",
        pm.math.sum(attribute_levels_right * beta_framed, axis=1),
        dims="task"
    )

    # choice probability via logit
    probability_choice_left = pm.Deterministic(
        "probability_choice_left",
        pm.math.exp(utility_left) / (pm.math.exp(utility_left) + pm.math.exp(utility_right)),
        dims="task"
    )

    # likelihood
    choice_distribution = pm.Bernoulli(
        "choice_distribution", 
        p=probability_choice_left, 
        observed=observed_choice_left
    )

# %% get priors

priors = pm.sample_prior_predictive(
    samples = 1000, 
    model = bayes_model, 
    random_seed = 42, 
)

# %% check priors

priors.prior
az.summary(priors, var_names = ["delta"])
# az.plot_forest(priors, var_names=["delta"], combined=True)
# delta_draws = priors.prior["delta"]

# %% test model 

with bayes_model:
    approx = pm.fit(n=10000, method="advi")
    trace = approx.sample(1000)

# %%

# run model with MCMC with 1000 draws, 500 tune samples, and 4 chains on 6 cores
inference_data = pm.sample(
    model = bayes_model, 
    draws = 1000, 
    tune = 500, 
    chains = 4,
    cores = 6, 
    random_seed = 42, 
    return_inferencedata = True, 
    target_accept = 0.9
)

# %% diagnostics

az.summary(inference_data, var_names=["beta", "delta", "gamma"])
az.plot_trace(inference_data, var_names=["beta", "delta", "gamma"])
az.plot_forest(inference_data, var_names=["beta"], combined=True)
az.plot_forest(inference_data, var_names=["delta"], combined=True)
az.plot_forest(inference_data, var_names=["gamma"], combined=True)

# %% save to file 

inference_data.to_netcdf("output/inference_basic_choice.nc")

# %%
