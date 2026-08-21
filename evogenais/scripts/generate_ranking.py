#!python3

import glob
import gzip
import json
import os
import re
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem.Descriptors import MolWt
from sklearn.preprocessing import MinMaxScaler
from itertools import product
from typing import List
from functools import partial
import numexpr as ne
import multiprocessing
import shutil


# Helper functions

def merge_force_suffix(left, right, **kwargs):
    on_col = kwargs['on']
    on_col = [on_col] if not isinstance(on_col, (list, tuple)) else on_col
    suffix_tupple = kwargs['suffixes']

    def suffix_col(col, suffix):
        if col not in on_col:
            return str(col) + suffix
        else:
            return col

    left_suffixed = left.rename(columns=lambda x: suffix_col(x, suffix_tupple[0]))
    right_suffixed = right.rename(columns=lambda x: suffix_col(x, suffix_tupple[1]))
    del kwargs['suffixes']
    return pd.merge(left_suffixed, right_suffixed, **kwargs)


def parse_ranges(ctx):
    if len(ctx['main_config']['score_weighted_attributes']) != len(ctx['main_config']['score_weights']):
        raise ValueError("Number of attributes not equal number of score_weights.")

    if len(ctx['main_config']['score_weighted_attributes']) != len(ctx['main_config']['score_weighted_step_size']):
        raise ValueError("Number of attributes not equal number of score_weighted_step_size.")

    if len(ctx['main_config']['score_weights']) != len(ctx['main_config']['score_weighted_step_size']):
        raise ValueError("Number of score_weights not equal number of score_weighted_step_size.")

    start_vals = [float(weight.split('_')[0]) for weight in ctx['main_config']['score_weights']]
    end_vals = [float(weight.split('_')[1]) if len(weight.split('_')) > 1 else float(weight) + 1.
                for weight in ctx['main_config']['score_weights']]

    steps = [float(weight) if weight != 'none' and weight != '' else 1 for weight in ctx['main_config']['score_weighted_step_size']]

    attribs = {k: np.arange(start_vals[i], end_vals[i], steps[i])
               for i, k in enumerate(ctx['main_config']['score_weighted_attributes'])}

    weighted_combos = list(product(*list(attribs.values())))
    return attribs, weighted_combos


def data_weighted_scores(weighted_combos: List[tuple], attribs: dict, df_normalized: pd.DataFrame, chunksize: int):
    score_func = partial(get_weighted_score, weighted_combos=weighted_combos, df_normalized=df_normalized,
                     attribs=attribs)

    pool = multiprocessing.Pool()
    results = pool.map(score_func, list(df_normalized.iterrows()), chunksize=chunksize)
    df_weighted = pd.DataFrame(results)
    pool.close()
    pool.join()

    filter_col = [col for col in df_weighted if col.startswith('score_weighted_')]
    df_weighted = df_weighted.assign(score_weighted=df_weighted[filter_col].max(axis=1),
                                     temp=df_weighted[filter_col].idxmax(axis=1),
                                     best_weights=lambda x: (
                                         x['temp'].apply(lambda x: x.split('score_weighted_')[1])))
    df_weighted.drop(columns=['temp'], inplace=True)

    cols = df_weighted.columns.tolist()
    idx = cols.index("score_weighted")
    cols = cols[idx:] + cols[:idx]
    df_weighted = df_weighted[cols]
    return df_weighted


def get_weighted_score(data: tuple, weighted_combos: List[tuple], df_normalized: pd.DataFrame, attribs: dict):
    (idx, row) = data
    for weights in weighted_combos:
        weighted_score = 0
        total = sum(map(lambda x: abs(float(x)), list(weights)))
        name = ':'.join(list(f'{np.round(w, 2)}' for w in weights))

        if idx == 0:
            df_normalized[f'score_weighted_{name}'] = [0.0] * df_normalized.shape[0]

        for weight, attrib in zip(weights, list(attribs.keys())):
            ranking_weight = abs(weight)
            if row[attrib + '_scaled'] < 0:
                weighted_score = 0

            elif weight < 0:
                weighted_score += ((1 - row[attrib + '_scaled']) * ranking_weight)

            else:
                weighted_score += (row[attrib + '_scaled'] * ranking_weight)

        df_normalized.loc[idx, f'score_weighted_{name}'] = weighted_score / total

    return df_normalized.iloc[idx].to_dict()


# Main functions

def generate_ranking(ctx, docking_scenario_output_folders, vflp_folder_paths):
    # Check for VFVS output format. Currently only latest csv format of VFVS python version is supported
    if (ctx['main_config']['vfvs_summary_output_format'] == "csv"):

        all_res = []

        # Collect all VFVS data from all docking scenarios indicated in config file
        for index, docking_scenario_output_folder in enumerate(docking_scenario_output_folders):

            files = glob.glob(os.path.join(docking_scenario_output_folder, 'csv', '*', '*.csv.gz'))

            if len(files) < 1:
                print(f"No results in {docking_scenario_output_folder} found, continuing...")
                return

            # Todo: Test the workunit and task information when having different docking scenarios
            df_workunits = []
            pattern = re.compile(r'csv/(\d+)/(\d+)\.csv\.gz')

            for f in files:
                match = pattern.search(f)

                if match:
                    workunit = int(match.group(1))
                    task = int(match.group(2))
                else:
                    workunit = 0
                    task = 0

                df_workunit = pd.read_csv(f, compression='gzip', sep=',')
                df_workunit.insert(loc=3, column='workunit', value=workunit)
                df_workunit.insert(loc=4, column='task', value=task)
                df_workunits.append(df_workunit)

            df = pd.concat(df_workunits, axis=0, ignore_index=True)
            all_res.append(df)

        for index, df in enumerate(all_res):

            # If multiple docking scenarios are provided, a suffix is added to the column identifiers
            # Due to a problem in pandas df, a workaround is needed

            if index == 0:
                df_all = all_res[index]

            if index == 1:
                df_all = merge_force_suffix(df_all, all_res[index], on=['ligand', 'collection_key', 'attr_smi_orig'],
                                            suffixes=['_' + ctx['main_config']['docking_scenarios'][index - 1],
                                                      '_' + ctx['main_config']['docking_scenarios'][index]],
                                            how='inner')

            if index > 1:
                df_all = merge_force_suffix(df_all, all_res[index], on=['ligand', 'collection_key', 'attr_smi_orig'],
                                            suffixes=['', '_' + ctx['main_config']['docking_scenarios'][index]],
                                            how='inner')

    else:
        print(f"The option vfvs_summary_output_format has to be set to a supported format. Exiting...")
        exit(1)

    # Get additional attributes generated by VFLP
    data = {}

    # Iterate over all gzipped JSON files in the folder
    for file_path in glob.glob(vflp_folder_paths):

        match = re.search(r'VFLP([^/]+)/', file_path)
        if match:
            vflp_suffix = match.group(1)
        else:
            print(f"Did not find correct VFLP suffix from {file_path}. "
                  f"Make sure each VFLP instance begins with VFLP. Continuing...")
            continue

        with gzip.open(file_path, 'rb') as file:
            json_data = json.load(file)
            ligands = json_data['ligands']

            for ligand in ligands.values():
                tautomers = ligand['tautomers']

                for tautomer in tautomers.values():
                    remarks = tautomer['remarks']

                    if 'additional_attr' in remarks and 'smiles_current' in remarks:
                        attributes = remarks['additional_attr']
                        smiles_current = remarks['smiles_current']
                        value_smiles = smiles_current.split(": ", 1)[1]
                        key_smiles = 'attr_smi' + vflp_suffix

                        smiles_orig = remarks['smiles_original']
                        value_smiles_orig = smiles_orig.split(": ", 1)[1]
                        key_smiles_orig = 'attr_smi_orig'

                        if key_smiles_orig not in data:
                            smiles_orig_dict = {}
                            data[key_smiles_orig.strip()] = smiles_orig_dict

                        if key_smiles_orig in data[key_smiles_orig.strip()] and key_smiles in data[
                            key_smiles_orig.strip()]:
                            data[key_smiles_orig.strip()][key_smiles_orig.strip()].append(value_smiles_orig.strip())
                            data[key_smiles_orig.strip()][key_smiles.strip()].append(value_smiles.strip())
                        else:
                            data[key_smiles_orig.strip()][key_smiles_orig.strip()] = []
                            data[key_smiles_orig.strip()][key_smiles_orig.strip()].append(value_smiles_orig.strip())
                            data[key_smiles_orig.strip()][key_smiles.strip()] = []
                            data[key_smiles_orig.strip()][key_smiles.strip()].append(value_smiles.strip())

                        for attribute in attributes:
                            key_attr, value_attr = attribute.split(": ", 1)
                            key_attr += vflp_suffix

                            if key_attr not in data:
                                attribute_dict = {}
                                data[key_attr.strip()] = attribute_dict

                            if key_attr in data[key_attr.strip()] and key_smiles_orig in data[key_attr.strip()]:
                                data[key_attr.strip()][key_attr.strip()].append(value_attr.strip())
                                data[key_attr.strip()][key_smiles_orig.strip()].append(value_smiles_orig.strip())
                            else:
                                data[key_attr.strip()][key_attr.strip()] = []
                                data[key_attr.strip()][key_attr.strip()].append(value_attr.strip())
                                data[key_attr.strip()][key_smiles_orig.strip()] = []
                                data[key_attr.strip()][key_smiles_orig.strip()].append(value_smiles_orig.strip())

    for attribute_data in data.values():
        attribute_df = pd.DataFrame(attribute_data)
        df_all = pd.merge(df_all, attribute_df, on='attr_smi_orig', how='left')

    # Check if weighted score should be calculated and what properties are missing
    check_attributes = []

    if 'score_weighted' in ctx['main_config']['ranking_attributes']:
        check_attributes.extend(ctx['main_config']['score_weighted_attributes'])

    check_attributes.extend(
        [element for element in ctx['main_config']['ranking_attributes'] if element != 'score_weighted'])

    check_attributes = list(set(check_attributes))

    # Add filter attributes to the list of relevant attributes
    if ctx['main_config']['filter_mw'] == "true":
        check_attributes.append('mw')

    # Check if we need to calculate any additional molecular properties or scores

    if bool(set(check_attributes) - set(df_all.columns)) or ctx['main_config']['filter_mw'] == "true":

        attributes_generate = set(check_attributes) - set(df_all.columns)

        for i in df_all.index:

                smi = df_all['attr_smi_orig'].values[i]
                mol = smi2mol(smi)

                for attribute in attributes_generate:

                        if attribute == 'mw':
                                try:
                                        mw = MolWt(mol)
                                        df_all.at[i, 'mw'] = round(mw, 2)
                                except Exception:
                                        continue

        # Filter for molecular weight if set in config
        if ctx['main_config']['filter_mw'] == "true":
                df_all = df_all[(df_all['mw'] >= float(ctx['main_config']['filter_mw_min'])) & (
                                df_all['mw'] <= float(ctx['main_config']['filter_mw_max']))]
                df_all = df_all.reset_index(drop=True)

    df_final = df_all
    df_final['iteration'] = ctx['iteration'] - 1

    # Combine results with latest duplicates, if available
    if os.path.isfile(ctx['output_ranking_duplicates']):
        ranking_duplicates = pd.read_csv(ctx['output_ranking_duplicates'], compression='gzip', sep=',')
        ranking_duplicates_latest = ranking_duplicates[ranking_duplicates['iteration'] == ctx['iteration'] - 1]
        df_final = pd.concat([df_final, ranking_duplicates_latest]).reset_index(drop=True)

    return df_final


def prepare_output_dir(ctx):
    base_root = os.path.abspath(ctx.get("output_root", os.path.join(os.getcwd(), "output-files")))
    current_root = os.path.join(base_root, "current_iteration")
    iteration = int(ctx.get("iteration", 1))

    if iteration > 1:
        previous_root = os.path.join(base_root, f"iteration_{iteration - 1}")
        if os.path.exists(current_root):
            if os.path.exists(previous_root):
                shutil.rmtree(previous_root)
            os.replace(current_root, previous_root)

    os.makedirs(current_root, exist_ok=True)

    # expose both base and current roots
    ctx["output_root_base"] = base_root
    ctx["output_root"] = current_root

    # ensure runtime/config point to the current_iteration folder
    ctx["main_config"]["output_root"] = current_root
    ctx["main_config"]["ranking_file"] = os.path.join(current_root, "ranking_latest.csv.gz")

    # debug: show what paths are used
    print(f"[prepare_output_dir] base_root={base_root} current_root={current_root}", flush=True)

    return current_root


def process(ctx):
    prepare_output_dir(ctx)

    if ctx['iteration'] == 1:
        from .generate_molecules import generate_molecules
        print(f"Iteration {ctx['iteration']} - No ranking generation necessary, continuing...")

        if (ctx['main_config']['generative_model'] == "reinvent-randomized"):
            input_models = sorted(glob.glob(ctx['main_config']['model_folder_path'] + "/*"), key=os.path.getmtime)
            input_model = input_models[len(input_models) - 1]
            current_model = os.path.join(ctx['temp_dir'].name, 'model.' + str(ctx['iteration']))
            shutil.copyfile(input_model, current_model)

            # Generating temporary (current iteration) output folder path
            output_folder = os.path.join(ctx['output_root'], 'vfgm')

            # Replacing output of previous iteration by current iteration output
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            shutil.copy(current_model, os.path.join(output_folder, 'model.trained'))

        else:
            print(f"Generative model specified in config file not known or does not support option skip_first_training=true, returning...")
            return

        generate_molecules(ctx, None, current_model)

        return

    else:
        docking_scenario_output_folders = []

        output_root = ctx['output_root']
        os.makedirs(output_root, exist_ok=True)
        
        output_ranking_latest = os.path.join(output_root, 'ranking_latest.csv.gz')
        output_ranking_all_model = os.path.join(output_root, 'ranking_all.csv.gz')
        output_ranking_all = os.path.join(output_root, 'ranking_all.csv.gz')
        output_ranking_all_weighted = os.path.join(output_root, 'ranking_all_weighted.csv.gz')
        output_ranking_duplicates = os.path.join(output_root, 'ranking_latest_duplicates.csv.gz')

        # Pass output ranking all file path to ctx object, so we can add the already tested molecules to it
        ctx['output_ranking_duplicates'] = output_ranking_duplicates

        vfvs_pattern = ctx['main_config'].get('vfvs_output_glob')
        if not vfvs_pattern:
            raise KeyError("Missing configuration key: vfvs_output_glob")
        if not os.path.isabs(vfvs_pattern):
            vfvs_pattern = os.path.join(ctx['main_config']['_config_dir'], vfvs_pattern)

        for docking_scenario in ctx['main_config']['docking_scenarios']:
            pattern = os.path.join(vfvs_pattern, docking_scenario)
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(
                    f"No docking output folder found for scenario {docking_scenario!r} "
                    f"using pattern {pattern!r}"
                )
            docking_scenario_output_folders.extend(matches)

        vflp_pattern = ctx['main_config']['vflp_folder_paths']
        if not os.path.isabs(vflp_pattern):
            vflp_pattern = os.path.join(ctx['main_config']['_config_dir'], vflp_pattern)
        ranking = generate_ranking(ctx, docking_scenario_output_folders, vflp_pattern)

        if not os.path.isfile(output_ranking_all):
            ranking.to_csv(output_ranking_all, header=True, index=None, compression='gzip')
        else:
            ranking_previous = pd.read_csv(output_ranking_all, compression='gzip', sep=',')
            ranking_all = pd.concat([ranking_previous, ranking]).reset_index(drop=True)
            ranking_all.to_csv(output_ranking_all, header=True, index=None, compression='gzip')

        ranking_all = pd.read_csv(output_ranking_all, compression='gzip', sep=',')

        if ctx['main_config']['score_weighted'] == "true" or 'score_weighted' in ctx['main_config'][
            'ranking_attributes'] or any(re.search('scaled', s) for s in ctx['main_config'][
            'ranking_attributes']):

            # Normalize ranking attributes using MinMaxScaler
            scaler = MinMaxScaler()
            df_ranking_attributes = ranking_all[
                ctx['main_config']['score_weighted_attributes']].copy().add_suffix('_scaled')
            df_ranking_attributes_abs = df_ranking_attributes.abs()

            df_normalized = pd.DataFrame(scaler.fit_transform(df_ranking_attributes_abs),
                                         columns=df_ranking_attributes_abs.columns)

            # If cutoffs are defined, set scaled score of attribute to 0 or 1 if out of range
            if ctx['main_config']['score_weighted_cutoffs'][0] != "":

                for idx, cutoffs in enumerate(ctx['main_config']['score_weighted_cutoffs']):

                    cutoff_min = cutoffs.split("_")[0]
                    if cutoff_min != "@":
                        cutoff_min = float(cutoff_min)

                    cutoff_max = cutoffs.split("_")[1]
                    if cutoff_max != "@":
                        cutoff_max = float(cutoff_max)

                    attr = ctx['main_config']['score_weighted_attributes'][idx] + '_scaled'

                    if cutoff_max == "@" and cutoff_min == "@":
                        continue
                    elif cutoff_max == "@":
                        df_update = df_ranking_attributes[(df_ranking_attributes[attr] < cutoff_min)][
                            attr].to_frame()
                    elif cutoff_min == "@":
                        df_update = df_ranking_attributes[(df_ranking_attributes[attr] > cutoff_max)][
                            attr].to_frame()
                    else:
                        df_update = df_ranking_attributes[(df_ranking_attributes[attr] > cutoff_max) | (
                                df_ranking_attributes[attr] < cutoff_min)][attr].to_frame()

                    # If weight is negative set to 1, of weight is positive set to 0
                    df_update[attr].values[:] = -1

                    # Update dataframe with values that are out of the cutoff range
                    df_normalized[attr].update(df_update[attr])

            # Calculate score for weighted ranking (new, ranges of weights)
            attribs, weighted_combos = parse_ranges(ctx)
            df_normalized = data_weighted_scores(weighted_combos,
                                                 attribs,
                                                 df_normalized,
                                                 len(ctx['main_config']['score_weights']))

            ranking_all_weighted = pd.concat([ranking_all, df_normalized], axis=1)
            iteration_column = ranking_all_weighted.pop('iteration')
            ranking_all_weighted['iteration'] = iteration_column

            ranking_attributes_ascending_bool = [bool(value) for value in
                                                 ctx['main_config']['ranking_attributes_ascending']]

            # Remove duplicates
            ranking_all_weighted = ranking_all_weighted.drop_duplicates().reset_index(drop=True)

            # Create dense rank
            #ranking_all_weighted['rank'] = ranking_all_weighted.sort_values(
            #    by=ctx['main_config']['ranking_attributes'],
            #    ascending=ranking_attributes_ascending_bool).groupby(
            #    ctx['main_config']['ranking_attributes'], sort=False).ngroup() + 1

            # Normalize rank from 1 to 0
            #ranking_all_weighted['rank'] = 1.0 - (ranking_all_weighted['rank'] - 1) / ranking_all_weighted[
            #    'rank'].max()

            # Sort dataframe by ranking attributes
            ranking_all_weighted = ranking_all_weighted.sort_values(
                by=ctx['main_config']['ranking_attributes'],
                ascending=ranking_attributes_ascending_bool).reset_index(drop=True)

            ranking_all_weighted.to_csv(output_ranking_all_weighted, header=True, index=None, compression='gzip')
            ranking_latest = ranking_all_weighted

        else:
            ranking_attributes_ascending_bool = [bool(value) for value in
                                                 ctx['main_config']['ranking_attributes_ascending']]

            # Create dense rank
            #ranking_all['rank'] = ranking_all.sort_values(
            #    by=ctx['main_config']['ranking_attributes'],
            #    ascending=ranking_attributes_ascending_bool).groupby(
            #    ctx['main_config']['ranking_attributes'], sort=False).ngroup() + 1

            # Normalize rank from 1 to 0
            #ranking_all['rank'] = 1.0 - (ranking_all['rank'] - 1) / ranking_all['rank'].max()

            # Sort dataframe by ranking attributes
            ranking_all = ranking_all.sort_values(by=ctx['main_config']['ranking_attributes'],
                                                  ascending=ranking_attributes_ascending_bool).reset_index(
                drop=True)

            ranking_latest = ranking_all

        # Latest ranking and complete ranking are saved
        ranking_latest = ranking_latest.drop_duplicates().reset_index(drop=True)
        ranking_latest.to_csv(output_ranking_latest, header=True, index=None, compression='gzip')
        ranking_all = ranking_all.drop_duplicates().reset_index(drop=True)
        ranking_all.to_csv(output_ranking_all_model, header=True, index=None, compression='gzip')
        ranking_all.to_csv(output_ranking_all, header=True, index=None, compression='gzip')
