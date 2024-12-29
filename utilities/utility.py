import csv
import json
import os
from itertools import combinations

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.lines import Line2D
from ogb.utils.features import atom_feature_vector_to_dict, bond_feature_vector_to_dict
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from tqdm import tqdm


def normalize_dataset_name(name: str) -> str:
    """
    Normalize the dataset name to match the canonical form.
    :param name: The name of the dataset, which may vary in case and formatting.
    :return: The canonical form of the dataset name if it matches known datasets, otherwise None.
    """
    # Remove common delimiters and convert to lowercase
    return name.lower().replace("-", "").replace("_", "")


def read_from_json(file_name: str) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def get_names(dataset: str, dir: str = "") -> list:
    """
    Get the names of the molecules in the dataset
    :param dataset: Name of the dataset
    :return: List of molecule names
    """

    # Normalize the dataset name
    normalized_name = normalize_dataset_name(dataset)

    # Construct the filename based on the normalized dataset name
    if normalized_name in ["bbbp", "esol", "freesolv", "bace"]:
        filename = f"{dir}/dataset/ogbg_mol{normalized_name}/mapping/mol.csv.gz"
        if normalized_name == "bace":
            filepath = os.path.join(dir, "dataset", "BACE.csv")
            df = pd.read_csv(filepath)
            return df["mol"].tolist()
        df = pd.read_csv(filename, compression="gzip")
        return df["mol_id"].tolist()
    elif normalized_name == "mutag":
        filename = "{dir}dataset/MUTAGname.csv"
        names = []
        with open(filename, "r") as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip the header
            names = [row[1] for row in reader]
        return names
    else:
        err_msg = f"Dataset name {dataset} is not recognized.\nOnly the following datasets are supported:\nBBBP, ESOL, FreeSolv, MUTAG"
        raise ValueError(err_msg)


def save_dictionary_to_json(dictionary: dict, file_name: str):
    with open(file_name, "w") as f:
        json.dump(dictionary, f)
    print(f"Saved dictionary to {file_name}.")


def get_keys_from_raw_data(raw_data: list, check: bool = False) -> list:
    """
    Get all keys from the raw data.
    :param raw_data: List of dictionaries
    :return: List of keys
    """
    if not raw_data:
        return []

    # Get the keys from the first dictionary and ensure the order is preserved
    first_keys = []
    for x in raw_data[0]:
        if ":" in x:
            first_keys.append(x.split(":")[0].strip("-").strip())
    # first_keys = [x.split(":")[0].strip("-").strip() for x in raw_data[0]]

    # Make sure all characters are lowercase
    first_keys = [x.lower() for x in first_keys]

    # Check if all dictionaries have the same set of keys
    for entry in raw_data[1:]:
        keys = []
        for x in entry:
            if ":" in x:
                keys.append(x.split(":")[0].strip("-").strip())
        # keys = [x.split(":")[0].strip("-").strip() for x in entry]
        keys = [x.lower() for x in keys]
        # Update the first_keys if the current entry has more keys
        if len(keys) > len(first_keys):
            first_keys = keys
        if check and keys != first_keys:
            raise ValueError("All dictionaries in the list should have the same keys.")

    return first_keys


def parse_entry(entry, all_keys: list) -> dict:
    """
    Parse an entry (either a list or dictionary) into a dictionary with numerical values.
    :param entry: List or Dictionary of strings
    :param all_keys: List of keys to be parsed
    :return: Dictionary of parsed keys and values
    """
    all_keys = [key.lower() for key in all_keys]
    parsed = {key: None for key in all_keys}  # Initialize with None
    if isinstance(entry, dict):  # If entry is a dictionary
        # Convert all keys to lowercase
        entry = {key.lower(): value for key, value in entry.items()}
        for key in all_keys:
            if key in entry:
                value = entry[key]
                if key != "name":  # Keep 'Name' as is
                    # Remove any non-numeric characters and try to convert to float
                    value = "".join(filter(lambda x: x.isdigit() or x == ".", value))
                    value = try_float(value)
                parsed[key] = value
    elif isinstance(entry, list):  # If entry is a list
        for x in entry:
            if ":" not in x:
                continue
            key, value_str = x.split(":", 1)
            key = key.strip("-").strip().lower()
            # print(f'key: {key}, value_str: {value_str}')
            if key in all_keys:
                value = (
                    try_float(value_str.split()[0])
                    if key != "Name"
                    else value_str.strip()
                )
                parsed[key] = value

    return parsed


def try_float(s: str) -> float or None:
    try:
        return float(s)
    except:
        return None


def fill_nones_with_column_average(X: np.ndarray) -> (np.ndarray, list):
    """
    Fill None values in a 2D array with the column average.
    :param X: 2D array
    :return: 2D array with None values filled with column average, list of rows filled
    """
    rows_filled = []  # To keep track of rows filled with the column average

    # Iterate through each column
    for col_idx in range(X.shape[1]):

        # Extract the column
        col = X[:, col_idx]

        non_none_values = [x for x in col if x is not None and not np.isnan(x)]

        # Calculate the column average for non-None values
        if non_none_values:
            col_average = np.mean(non_none_values)

        # Replace None with the column average
        for row_idx in range(X.shape[0]):
            if X[row_idx, col_idx] is None or np.isnan(X[row_idx, col_idx]):
                X[row_idx, col_idx] = col_average
                if row_idx not in rows_filled:
                    rows_filled.append(row_idx)

    print("Rows filled with column average:", rows_filled)

    return X, rows_filled


def calculate_missing_value_rates(X: np.ndarray, keys: list) -> pd.DataFrame:
    """
    Calculate the missing value rates for each column in a 2D array.

    :param X: 2D array
    :param keys: List of column names
    :return: DataFrame with column names and their corresponding missing value rates
    """
    missing_rates = {}

    # Iterate through each column
    for col_idx, key in enumerate(keys):
        col = X[:, col_idx]

        # Count the number of missing values (None or NaN)
        missing_count = sum(x is None or np.isnan(x) for x in col)

        # Calculate the missing rate
        missing_rate = missing_count / X.shape[0]
        missing_rates[key] = missing_rate * 100

    # Create a DataFrame for better visualization
    missing_rates_df = pd.DataFrame(
        list(missing_rates.items()), columns=["Property", "Missing Rate (%)"]
    )
    return missing_rates_df


def num_missing_value_by_row(X: np.ndarray, keys: list, plot=True) -> pd.DataFrame:
    """
    Calculate the missing value rates for each row in a 2D array.

    :param X: 2D array
    :param keys: List of column names
    :return: DataFrame with row indices and their corresponding missing value rates
    """
    missing_counts = {}

    # Iterate through each row
    for row_idx in range(X.shape[0]):
        row = X[row_idx]

        # Count the number of missing values (None or NaN)
        missing_count = sum(x is None or np.isnan(x) for x in row)
        missing_counts[row_idx] = missing_count

    # Create a DataFrame for better visualization
    missing_rates_df = pd.DataFrame(
        list(missing_counts.items()), columns=["Row Index", "Number of Missing Values"]
    )

    if plot:
        plt.figure()
        plt.scatter(
            missing_rates_df["Row Index"], missing_rates_df["Number of Missing Values"]
        )
        plt.xlabel("Row Index")
        plt.ylabel("Number of Missing Values")
        plt.title("Number of Missing Values by Row")
        plt.show()

    return missing_rates_df


# Function to calculate AIC for Linear Regression
def calculate_aic(n: int, mse: float, num_params: int) -> float:
    """
    Calculate the Akaike Information Criterion (AIC) for a linear regression model.
    :param n: Number of observations
    :param mse: Mean squared error of the model
    :param num_params: Number of parameters in the model
    :return: AIC score
    """
    aic = n * np.log(mse) + 2 * num_params
    return aic


def select_features_AIC(X_train, y_train, X_valid, y_valid, plot=True):
    """
    Select the best sub-model using the AIC score.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_valid: Validation data
    :param y_valid: Validation labels
    :param plot: Whether to plot the AIC scores
    :return: Best AIC score, list of best features
    """
    best_aic = np.inf
    best_features = None
    aic_scores = []

    total_features = X_train.shape[1]
    all_feature_indices = range(total_features)

    outer_tqdm = tqdm(reversed(range(1, total_features + 1)), desc="Outer Loop")
    for k in outer_tqdm:
        best_aic_k = np.inf
        inner_tqdm = tqdm(
            combinations(all_feature_indices, k), desc="Inner Loop", leave=False
        )

        for indices in inner_tqdm:
            X_train_subset = X_train[:, indices]
            X_valid_subset = X_valid[:, indices]

            model = LinearRegression()
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_valid_subset)

            n = len(y_valid)
            mse = mean_squared_error(y_valid, y_pred)
            num_params = k + 1
            aic = calculate_aic(n, mse, num_params)

            if aic < best_aic_k:
                best_aic_k = aic

            if aic < best_aic:
                best_aic = aic
                best_features = indices

        aic_scores.append(best_aic_k)
        outer_tqdm.set_postfix({"AIC (current k) ": best_aic_k, "AIC (best)": best_aic})

    if plot:
        plt.figure()
        aic_scores.reverse()
        plt.plot(range(1, total_features + 1), aic_scores)
        plt.xlabel("Number of Features")
        plt.ylabel("Best AIC")
        plt.title("Best AIC vs Number of Features")
        plt.show()

    return best_aic, best_features


def select_features_RFE(X_train, y_train, X_valid, y_valid, plot=True):
    # Feature Selection with RFE
    valid_auc_roc = []
    # keep the index of the best auc-roc score
    best_auc_roc_index = 0
    # keep the best auc-roc score
    best_auc_roc = 0
    # keep the ranking of the features in each iteration (Pd.DataFrame)
    ranking = pd.DataFrame()
    # keep the selected features in each iteration (Pd.DataFrame)
    selected_features = pd.DataFrame()
    model = model = LogisticRegression()
    for i in range(1, X_train.shape[1]):
        rfe = RFE(model, n_features_to_select=i)
        rfe.fit(X_train, y_train)
        valid_auc_roc.append(roc_auc_score(y_valid, rfe.predict(X_valid)))
        if valid_auc_roc[-1] > best_auc_roc:
            best_auc_roc = valid_auc_roc[-1]
            best_auc_roc_index = i
        ranking[f"iter{i}"] = rfe.ranking_
        selected_features[f"iter{i}"] = rfe.support_
    if plot:
        print(f"Best AUC-ROC score: {best_auc_roc} with {best_auc_roc_index} features")
        print(selected_features)
        plt.plot(range(1, X_train.shape[1]), valid_auc_roc)
        plt.xlabel("Number of features")
        plt.ylabel("AUC-ROC score")
        plt.show()

    return (
        best_auc_roc,
        selected_features,
    )


def select_features_forward(X_train, y_train, X_valid, y_valid, plot=True):
    """
    Perform forward feature selection using the AIC score.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_valid: Validation data
    :param y_valid: Validation labels
    :param plot: Whether to plot the AIC scores
    :return: Best AIC score, list of best features
    """
    best_aic = np.inf
    best_features = []
    aic_scores = []
    total_features = X_train.shape[1]
    selected_features = []

    for i in tqdm(range(total_features), desc="Selecting Features"):
        best_aic_i = np.inf
        best_feature_i = None

        for feature in set(range(total_features)) - set(selected_features):
            current_features = selected_features + [feature]
            X_train_subset = X_train[:, current_features]
            X_valid_subset = X_valid[:, current_features]

            model = LinearRegression()
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_valid_subset)

            n = len(y_valid)
            mse = mean_squared_error(y_valid, y_pred)
            num_params = len(current_features) + 1
            aic = calculate_aic(n, mse, num_params)

            if aic < best_aic_i:
                best_aic_i = aic
                best_feature_i = feature

        if best_feature_i is not None and best_aic_i < best_aic:
            best_aic = best_aic_i
            selected_features.append(best_feature_i)
            aic_scores.append(best_aic)

    if plot:
        plt.figure()
        plt.plot(range(1, len(selected_features) + 1), aic_scores)
        plt.xlabel("Number of Features")
        plt.ylabel("Best AIC")
        plt.title("Best AIC vs Number of Features")
        plt.show()

    return best_aic, selected_features


def select_features_backward(X_train, y_train, X_valid, y_valid, plot=True):
    """
    Perform backward feature selection using the AIC score.
    :param X_train: Training data
    :param y_train: Training labels
    :param X_valid: Validation data
    :param y_valid: Validation labels
    :param plot: Whether to plot the AIC scores
    :return: Best AIC score, list of best features
    """
    total_features = X_train.shape[1]
    selected_features = list(range(total_features))
    best_aic = np.inf
    aic_scores = []

    while len(selected_features) > 1:
        best_aic_i = np.inf
        worst_feature_i = None

        for feature in selected_features:
            current_features = list(selected_features)
            current_features.remove(feature)
            X_train_subset = X_train[:, current_features]
            X_valid_subset = X_valid[:, current_features]

            model = LinearRegression()
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_valid_subset)

            n = len(y_valid)
            mse = mean_squared_error(y_valid, y_pred)
            num_params = len(current_features) + 1
            aic = calculate_aic(n, mse, num_params)

            if aic < best_aic_i:
                best_aic_i = aic
                worst_feature_i = feature

        if best_aic_i < best_aic:
            best_aic = best_aic_i
            selected_features.remove(worst_feature_i)
            aic_scores.append(best_aic)
        else:
            break

    if plot:
        plt.figure()
        plt.plot(
            range(total_features, total_features - len(aic_scores), -1), aic_scores
        )
        plt.xlabel("Number of Features")
        plt.ylabel("Best AIC")
        plt.title("Best AIC vs Number of Features")
        plt.show()

    return best_aic, selected_features


def check_length_match(X, y):
    if len(X) != len(y):
        raise ValueError("Length of X and y should be the same.")


def compare_parsed_data(
    parsed_data_1: list,
    parsed_data_2: list,
    num_repete: int = 10,
    feature: str = "molecular weight",
) -> int:
    """
    Compare specific entries between two parsed data lists and print differences.
    :param parsed_data_1: First list of dictionaries to compare
    :param parsed_data_2: Second list of dictionaries to compare
    :param num_repete: Number of repetitions for comparison
    :return: Number of differences found
    """
    num_diff = 0
    diff_idx = []
    for i in range(num_repete):
        mw2 = parsed_data_1[len(parsed_data_1) - (num_repete - i)][feature]
        mw3 = parsed_data_2[i][feature]

        print(
            f"({feature})  |  {len(parsed_data_1) - (num_repete-i)}th data in result2 : {mw2:.2f}  |  {i}th data in result3 : {mw3:.2f}"
        )

        # record the row number of the difference
        if parsed_data_1[len(parsed_data_1) - (num_repete - i)] != parsed_data_2[i]:
            diff_idx.append(i)
            num_diff += 1

    if diff_idx:
        print(f"num_diff : {num_diff}")
        print(f"row number of the difference : {diff_idx}")

    return num_diff


def atom_feature_vector_to_dict_full_name(node: torch.Tensor) -> dict:
    """
    Convert an atom feature vector to a dictionary with full names.
    :param node: Atom feature vector
    :return: Dictionary of atom features
    """
    key_mapping = {
        "atomic_num": "Atomic Number",
        "chirality": "Chirality",
        "degree": "Degree of Connectivity",
        "formal_charge": "Formal Charge",
        "num_h": "Number of Hydrogen Atoms",
        "num_rad_e": "Number of Radical Electrons",
        "hybridization": "Hybridization Type",
        "is_aromatic": "Is Aromatic",
        "is_in_ring": "Is In Ring",
    }

    atom_dict = atom_feature_vector_to_dict(node)
    return {key_mapping.get(key, key): value for key, value in atom_dict.items()}


def bond_feature_vector_to_dict_full_name(edge: torch.Tensor) -> dict:
    """
    Convert a bond feature vector to a dictionary with full names.
    :param edge: Bond feature vector
    :return: Dictionary of bond features
    """
    key_mapping = {
        "bond_type": "Bond Type",
        "bond_stereo": "Bond Stereochemistry",
        "is_conjugated": "Is Conjugated",
    }

    bond_dict = bond_feature_vector_to_dict(edge)
    return {key_mapping.get(key, key): value for key, value in bond_dict.items()}


def plot_molecule_dgl(
    data: tuple, compound_name: str = None, smiles: str = None
) -> None:
    """
    Plot a molecule in DGL format.
    :param data: Tuple of DGL graph and label
    :param compound_name (optional): Name of the compound
    :param smiles (optional): SMILES representation of the compound
    """

    data, label = data
    # Convert DGL graph to NetworkX format
    G = dgl.to_networkx(data)

    # Node labels and colors based on atomic number
    atomic_number_to_symbol = {
        0: "H",
        1: "He",
        2: "Li",
        3: "Be",
        4: "B",
        5: "C",
        6: "N",
        7: "O",
        8: "F",
        9: "Ne",
        10: "Na",
        11: "Mg",
        12: "Al",
        13: "Si",
        14: "P",
        15: "S",
        16: "Cl",
        17: "Ar",
        18: "K",
        19: "Ca",
        33: "Se",
        51: "I",
    }

    node_color_map = {
        0: "grey",  # H
        4: "blue",  # B
        5: "orange",  # C
        6: "cyan",  # N
        7: "red",  # O
        8: "green",  # F
        9: "purple",  # Ne
        10: "lime",  # Na
        13: "teal",  # Si
        14: "pink",  # P
        15: "yellow",  # S
        16: "magenta",  # Cl
        33: "brown",  # Se
        51: "olive",  # I
    }

    # Edge styles and colors based on bond type
    bond_type_to_style = {0: "dotted", 1: "solid", 2: "solid", 3: "solid"}
    bond_color_map = {0: "black", 1: "grey", 2: "grey", 3: "black"}

    edge_features = data.edata["feat"]
    edge_styles = [
        (
            bond_type_to_style[edge_features[i, 0].item()],
            bond_color_map[edge_features[i, 0].item()],
        )
        for i in range(data.number_of_edges())
    ]

    # Extracting node colors and labels based on atomic number
    node_colors = [
        node_color_map[data.ndata["feat"][i, 0].item()]
        for i in range(data.number_of_nodes())
    ]
    labels = {
        i: atomic_number_to_symbol[data.ndata["feat"][i, 0].item()] for i in G.nodes()
    }

    # Plotting
    plt.figure(figsize=(6, 6))
    pos = nx.kamada_kawai_layout(G)

    plt.axis("off")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors)

    # Plotting edges and building legend
    drawn_combinations = set()
    for edge, (style, color) in zip(G.edges, edge_styles):
        if (style, color) not in drawn_combinations:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[edge],
                style=style,
                edge_color=color,
                label=f"{style} {color}",
                width=1,
                arrows=False,
            )
            drawn_combinations.add((style, color))
        else:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[edge],
                style=style,
                edge_color=color,
                width=3,
                arrows=False,
            )

    # Legend creation
    custom_lines = [
        Line2D([0], [0], color=color, linestyle=style, lw=2)
        for style, color in drawn_combinations
    ]
    plt.legend(
        custom_lines,
        [f"{style} {color}" for style, color in drawn_combinations],
        title="Bond Type",
    )

    nx.draw_networkx_labels(G, pos, labels=labels)

    if compound_name:
        plt.title(
            f"{compound_name}  -- (Class: {label.item()})",
            fontsize=14,
            pad=30 if smiles else 10,
        )
    else:
        plt.title(f"Class: {label.item()}", fontsize=14, pad=30 if smiles else 10)

    # Setting subtitle using text function
    if smiles:
        plt.gca().text(
            0.5,
            1.05,
            smiles,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="center",
        )

    plt.show()


# get property names from label
#   dataset: str in ["free_solv", "esol"]
#   labels: list of str. labels in the yaml file
#   output: a list of property names


def get_properties(dataset, labels, dir: str = "."):
    if dataset not in ["free_solv", "esol"]:
        print("Error: invalid dataset")
        return
    file_path = f"{dir}/results/{dataset}/{dataset}_properties.yaml"

    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    properties = []

    for label in labels:
        if data.get(label):
            properties += data.get(label)
        else:
            print(f"Warning: {label} is not a valid label")

    return properties


# register properties under a label
#   dataset: str in ["free_solv", "esol"]
#   label: str. label the property list
#   properties: list of str property names
def add_properties(dataset, label, properties, dir: str = "."):
    if dataset not in ["free_solv", "esol"]:
        print("Error: invalid dataset")
        return
    file_path = f"{dir}/results/{dataset}/{dataset}_properties.yaml"

    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    if not data:
        data = {}

    if label not in data.keys():
        data[label] = properties
    else:
        print("Warning: label already exist")

    with open(file_path, "w") as file:
        yaml.safe_dump(data, file)


# retrieve data from property names
#   dataset: str in ["free_solv", "esol"]
#   properties: list of str property names


def get_data(dataset, properties, dir: str = "."):
    if dataset not in ["free_solv", "esol"]:
        print("Error: invalid dataset")
        return
    file_path = f"{dir}/results/{dataset}/{dataset}.json"
    data = pd.read_json(file_path)
    return data[properties]


# add data into the database
#   dataset: str in ["free_solv", "esol"]
#   data: a pandas DataFrame object with columns of property names and rows of molecule names
def add_data(dataset, data):
    if dataset not in ["free_solv", "esol"]:
        print("Error: invalid dataset")
        return
    file_path = f"{dir}/results/{dataset}/{dataset}.json"
    old_data = pd.read_json(file_path)

    # check valid keys
    overlapping_keys = list(set(data.columns) & set(old_data.columns))
    new_keys = list(set(data.columns) - set(overlapping_keys))
    if overlapping_keys:
        print(f"Warning: overlapping keys not added: {overlapping_keys}")
        data = data[new_keys]

    old_data = old_data.join(data)
    old_data.to_json(file_path)

    print(f"Info: written properties {new_keys}")


import re


# extract property and function from gpt output
#   string: [property] : [code]
def extract_function(string):
    match = re.match(r"- (.+?): `(.+)?`", string)
    if match:
        property_name = match.group(1)
        code = match.group(2)
    else:
        property_name = ""
        code = ""
    return property_name, code


# populate the dictionary with entries
#   property_name : rdkit function code
def parse_tools_list(calculatable_property_list):
    rdkit_function_dict = {}

    for line in calculatable_property_list:
        property, code = extract_function(line)
        if property:
            rdkit_function_dict[property] = code

    return rdkit_function_dict


import rdkit

# make sure to add any other name bindings that exec() needs


# evaluate a property calculation
#   code: code string
#   mol: rdkit mol object
def evaluate_property_value(code, mol, ns):
    if code == "None":
        return None
    try:
        exec(code, ns)
        return ns["val"]
    except Exception as e:
        print("function execution error:", e)
        return None


# calculate the rdkit benchmark values for a single molecule
#   mol: rdkit mol object
#   func_dict: function dictionary that corresponds property name to code string
def get_rdkit_values(mol_rd, func_dict):
    mol_property_val = {}
    for property_name, code in func_dict.items():
        val = None
        if code:
            ns = {"rdkit": rdkit, "mol": mol_rd}
            val = evaluate_property_value(code, mol_rd, ns)
            if val == None:
                print(f"error executing {property_name}")
        mol_property_val[property_name] = val
    return mol_property_val


def write_function(
    dataset,
    llm_model,
    property_name,
    description,
    function_output,
    save_dir="../generated_results/",
):
    """
    Updates or creates a YAML file with the given dataset, LLM model, property, description, and function output.

    Parameters:
    dataset (str): The name of the dataset.
    llm_model (str): The LLM model used.
    property_name (str): The name of the property.
    description (str): The natural language description of the property.
    function_output (str): The function output for the property.
    save_dir (str): The directory where the YAML file will be saved.
    """
    yaml_file = os.path.join(save_dir, "functions.yaml")

    # Check if the YAML file exists and read it
    if os.path.exists(yaml_file):
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file) or {}
    else:
        data = {}

    # Update the structure with the new information
    if dataset not in data:
        data[dataset] = {}
    if llm_model not in data[dataset]:
        data[dataset][llm_model] = {}

    data[dataset][llm_model][property_name] = {
        "description": description,
        "function_output": function_output,
    }

    # Write the updated structure to the YAML file
    with open(yaml_file, "w") as file:
        yaml.safe_dump(data, file)


def read_function(
    dataset, llm_model, property_name, yaml_file="../generated_results/functions.yaml"
):
    """
    Reads from a YAML file and retrieves the description and function output for a specific dataset, LLM model, and property.

    Parameters:
    dataset (str): The name of the dataset.
    llm_model (str): The LLM model used.
    property_name (str): The name of the property.
    yaml_file (str): The path to the YAML file.

    Returns:
    tuple: A tuple containing the description and function output.
    """
    # Check if the YAML file exists
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"YAML file {yaml_file} not found.")

    # Read the YAML file
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # Navigate to the required data
    try:
        description = data[dataset][llm_model][property_name]["description"]
        function_output = data[dataset][llm_model][property_name]["function_output"]
    except KeyError as e:
        raise KeyError(f"Data not found for {e.args[0]}")

    return description, function_output
