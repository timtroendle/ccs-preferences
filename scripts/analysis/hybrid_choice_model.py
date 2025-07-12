import pymc as pm 
import pandas as pd
import arviz as az
import xarray as xr
import numpy as np

# %% pymc bug workaround

import pytensor
pytensor.config.cxx = '/usr/bin/clang++'

# %% 

df = pd.read_csv("data/hcm_input.csv")

# %% define attributes and baselines

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

# create design matric to include dummies for latent variables
X_attr = dummies.values
latent_vars = df[["lreco", "galtan", "socio_ecol"]].values
X = np.hstack([X_attr, latent_vars])

# add dimensions
coords = {
    "level": dummies.columns.values.tolist() + ["lreco", "galtan", "socio_ecol"],
    "framing": df["framing"].cat.categories
}

# %% define HCM

with pm.Model(coords=coords) as hcm_model:
    
    #TODO (the individual dim) 1. Measurement model: Latent traits per individual
    theta_lreco = pm.Normal("theta_lreco", mu=0, sigma=1, dims="individual")
    theta_galtan = pm.Normal("theta_galtan", mu=0, sigma=1, dims="individual")
    theta_socio = pm.Normal("theta_socio", mu=0, sigma=1, dims="individual")

    #TODO (likert scale normalised to 0 to 1) Thresholds (assuming ordered categorical Likert scale: 6-point for most, 5-point for some)
    # You'd adjust this per item if necessary
    cutpoints = pm.Normal("cutpoints", mu=np.arange(5), sigma=1.0, shape=(3, 5))

    #TODO (df_individuals) Measurement model (IRT-style)
    for i, item in enumerate(["lreco_1", "lreco_2", "lreco_3"]):
        pm.OrderedLogistic(f"{item}_obs", 
                           eta=theta_lreco[df_individuals], 
                           cutpoints=cutpoints[0],
                           observed=df[item])

    for i, item in enumerate(["galtan_1", "galtan_2", "galtan_3"]):
        pm.OrderedLogistic(f"{item}_obs", 
                           eta=theta_galtan[df_individuals], 
                           cutpoints=cutpoints[1],
                           observed=df[item])

    for i, item in enumerate(["socio_ecol_1", "socio_ecol_2", "socio_ecol_3"]):
        pm.OrderedLogistic(f"{item}_obs", 
                           eta=theta_socio[df_individuals], 
                           cutpoints=cutpoints[2],
                           observed=df[item])

    # 2. Choice model: Main effects
    beta = pm.Normal("beta", mu=0, sigma=2, dims="level")

    # Framing-specific shift + moderation by latent traits
    delta = pm.Normal("delta", mu=0, sigma=1, dims="level")
    
    # New: moderation coefficients (one per latent trait, per level)
    gamma_lreco = pm.Normal("gamma_lreco", mu=0, sigma=1, dims="level")
    gamma_galtan = pm.Normal("gamma_galtan", mu=0, sigma=1, dims="level")
    gamma_socio = pm.Normal("gamma_socio", mu=0, sigma=1, dims="level")

    #TODO (framing codes, individual id per task) Data input
    f = pm.Data("f", framing_codes, dims="task")  # 0/1
    individual_idx = pm.Data("individual_idx", individual_id_per_task, dims="task")

    # Compute framing effects modulated by latent traits
    delta_full = (
        delta
        + gamma_lreco * theta_lreco[individual_idx]
        + gamma_galtan * theta_galtan[individual_idx]
        + gamma_socio * theta_socio[individual_idx]
    )

    beta_framed = beta + delta_full * f[:, None]

    #TODO (dot dot dots) Utility for each package
    attribute_levels_left = pm.Data("attribute_levels_left", ..., dims = ["task", "level"])
    attribute_levels_right = pm.Data("attribute_levels_right", ..., dims = ["task", "level"])

    utility_left = pm.Deterministic("utility_left", 
        pm.math.sum(attribute_levels_left * beta_framed, axis=1), dims = "task")

    utility_right = pm.Deterministic("utility_right", 
        pm.math.sum(attribute_levels_right * beta_framed, axis=1), dims = "task")

    # Logit probability
    prob_choice_left = pm.Deterministic("probability_choice_left", 
        pm.math.exp(utility_left) / (pm.math.exp(utility_left) + pm.math.exp(utility_right)),
        dims="task"
    )

    #TODO (observed_choice_left) Likelihood
    pm.Bernoulli("choice_distribution", p=prob_choice_left, observed = observed_choice_left)
