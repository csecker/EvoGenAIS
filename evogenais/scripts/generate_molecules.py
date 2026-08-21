#!python

import gzip
import os
import shutil
import glob
import pandas as pd
import numpy as np
import random
import requests
import time
from rdkit.Chem import FilterCatalog
from datetime import datetime
import json
import multiprocessing

from .generate_reinvent import rr_run, rr_sample_model
from .generate_stoned import stoned_run
from .filter_generated_smiles import filter_generated_smiles_worker


# Helper functions

def add_ligand_names(input_smi, prefix, starting_number, output_folder, file_format):
    # Get tranche from filename
    tranche_name = os.path.splitext(input_smi)[0]
    tranche_output = tranche_name + "." + file_format
    output_filepath_smi = os.path.join(output_folder, tranche_output)

    # Import input_smi as panda dataframe
    df = pd.read_csv(input_smi, header=None, delimiter=r"\s+")

    # Add ligand names
    df[1] = np.arange(starting_number, len(df) + starting_number)
    df[1] = df[1].astype(str).str.zfill(10)
    df[1] = prefix + "_" + df[1]

    # Write result to file
    df.to_csv(output_filepath_smi, header=None, index=None, sep='\t', mode='w')


def file_length(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        return len(lines)


def split_training_validation(input_smi, output_training_smi, output_validation_smi, ratio):
    ratio = float(ratio)
    with open(input_smi, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    numlines = int(len(lines) * ratio)

    with open(output_training_smi, 'w') as f:
        f.writelines(lines[numlines:])
    with open(output_validation_smi, 'w') as f:
        f.writelines(lines[:numlines])

    return str(numlines)


def wait_for_http_service(url, retry_interval=5, max_retries=50):
    retries = 0
    print("Test5", flush=True)

    while retries < max_retries:
        try:
            response = requests.get(url)
            print(response, flush=True)
            if response.status_code == 200:
                print(f"HTTP service is available at {url}", flush=True)
                return True
        except requests.ConnectionError:
            print(f"HTTP service not available. Retrying in {retry_interval} seconds...", flush=True)
            time.sleep(retry_interval)
        retries += 1
        print("Test6", flush=True)


def parse_config(filename):
    with open(filename, "r") as read_file:
        config = json.load(read_file)

    return config


def generate_molecules(ctx, task=None, model=None):
    print(f"-----------------------------------------------------------------")
    print(f"Sampling {model} for SMILES...", flush=True)
    print(f"")

    sampled_smi_can = []
    sampled_smi_properties = pd.DataFrame({'attr_smi_orig': []})

    rdkit_catalog_pickle = None
    syba = None
    gpusim_database = None
    gpusim_url = None

    if (ctx['main_config']['filter_rdkit_catalog'] == "true" and len(ctx['main_config']['filter_rdkit_catalog_sets']) > 0):
            params = FilterCatalog.FilterCatalogParams()
            for filter_set in ctx['main_config']['filter_rdkit_catalog_sets']:
                    params.AddCatalog(getattr(params.FilterCatalogs, filter_set))
            rdkit_catalog = FilterCatalog.FilterCatalog(params)
            rdkit_catalog_pickle = rdkit_catalog.Serialize()

    if (ctx['main_config']['gpusim_filter'] == "true"):
            gpusim_database = os.path.splitext(os.path.basename(ctx['main_config']['gpusim_db_filepath']))[0]
            gpusim_server = 'http://' + 'localhost' + ':' + ctx['main_config']['gpusim_port']
            print(gpusim_server, flush=True)
            gpusim_url = gpusim_server + '/similarity_search_json'

            if not wait_for_http_service(gpusim_server):
                    print(f"Could not reach gpusim http server, skipping ligand...", flush=True)
                    return

    # Setting timer
    start_time_sampling = datetime.now()

    # For certain models we want to go through the input library step by step, therefore setting iteration variable
    i = 0

    number_of_tranches_generate = ctx['iteration']
    character_string = "ABCDEFGHIJKLMNOPQRSTUVXYZ"
    base = 1
    while pow(base, 6) < number_of_tranches_generate:
        base += 1
    tranches_generate_list=list(map("".join, itertools.product(character_string[0:base], repeat=6)))

    tranches_iterations = tranches_generate_list[:number_of_tranches_generate]

    sample_file = os.path.join(ctx['temp_dir'].name,
                               tranches_iterations[ctx['iteration'] - 1] + '.sample.tmp')

    # Sampling is performed in multiple rounds with sample_interval_size until sample_size (or timelimit) is reached
    sampling_round = 1

    while len(sampled_smi_can) < int(ctx['main_config']['sample_size']):

            print(f"Generating novel molecules until {ctx['main_config']['sample_size']} are reached. Starting sampling round {sampling_round}...", flush=True)

            if (ctx['main_config']['generative_model'] == "reinvent-randomized"):
                    rr_sample_model(ctx['main_config']['reinvent_folder_path'],
                                                    ctx['main_config']['reinvent_environment'], model, sample_file,
                                                    ctx['main_config']['sample_interval_size'])

                    # Transform into canonical smiles, filter for unique smiles and check if smiles are valid
                    sampled_smi = open(sample_file, 'r').readlines()

            elif (ctx['main_config']['generative_model'] == "stoned"):
                    sampled_smi, i = stoned_run(ctx, task, i)

            else:
                    print(f"Selected generative model currently not known, exiting...")
                    exit(1)

            # Remove duplicates from raw sample
            sampled_smi = list(set(sampled_smi))
            print(f"{ctx['main_config']['generative_model']} has generated {len(sampled_smi)} molecules...", flush=True)

            # Saving config separately since some objects in ctx (e.g. s3) cannot be serialized
            cfg = {}
            cfg['main_config'] = ctx['main_config']
            cfg['temp_dir'] = ctx['temp_dir']

            # Filter smiles and return as canonical smiles (parallelized)
            pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
            results = pool.map(filter_generated_smiles_worker,
                               [(cfg, smi, rdkit_catalog_pickle, gpusim_database, gpusim_url) for smi in sampled_smi])
            canon_smis = [result[0] for result in results if result is not None]
            smis_properties = [result[1] for result in results if result is not None]
            pool.close()
            pool.join()

            # If a process returns None, remove from result, and remove duplicates
            canon_smis = [x for x in canon_smis if x is not None]
            canon_smis = list(set(canon_smis))

            print(f"{len(canon_smis)} unique molecules remaining after filtering...")

            # Add to complete list
            sampled_smi_can.extend(canon_smis)

            # Remove duplicates
            sampled_smi_can = list(set(sampled_smi_can))

            # Detect already screened molecules and add existing screening results to list of duplicates
            if (ctx['iteration'] > 1):

                if ctx.get("ranking_all") is None or isinstance(ctx['ranking_all'], str):
                    ranking_all_current = pd.read_csv("./output-files/ranking_all.csv.gz")
                else:
                    ranking_all_current = ctx['ranking_all']

                mask_attr_smi_orig = ranking_all_current['attr_smi_orig'].isin(sampled_smi_can)

                if mask_attr_smi_orig.any():

                    # duplicates_attr_smi = ranking_all_current.loc[mask_attr_smi].copy()
                    duplicates_attr_smi_orig = ranking_all_current.loc[mask_attr_smi_orig].copy()
                    # duplicates_attr_smi = duplicates_attr_smi.drop_duplicates(subset=['attr_smi']).reset_index(drop=True)
                    duplicates = duplicates_attr_smi_orig.drop_duplicates(subset=['attr_smi_orig']).reset_index(drop=True)
                    # duplicates = pd.concat([duplicates_attr_smi, duplicates_attr_smi_orig]).reset_index(drop=True)
                    # duplicates['iteration'] = ctx['iteration']
                    duplicates['ligand'] = duplicates['ligand'].str.replace(r'$', '_', regex=True)

                    # Remove duplicates from current sample list if they are not allowed
                    if (ctx['main_config']['allow_duplicates'] == "false"):
                        print(
                            f"Found {len(duplicates)} already screened ligand(s) in {len(sampled_smi_can)} sampled molecules.")
                        print(f"Removing duplicates from sample file...")

                        sampled_smi_can = [x for x in sampled_smi_can if
                                           x not in duplicates['attr_smi_orig'].tolist()]

                        print(f"Remaining novel ligands: {len(sampled_smi_can)}")
                        print(f"-----------------------------------------------------------------")

            # Enforce sample size if desired
            if (ctx['main_config']['sample_size_hard_limit'] == "true"):
                sampled_smi_can = sampled_smi_can[:int(ctx['main_config']['sample_size'])]

            # Save numerical properties from filtering
            smis_properties_df = pd.DataFrame(smis_properties)
            sampled_smi_properties = pd.concat([sampled_smi_properties, smis_properties_df], ignore_index=True)
            sampled_smi_properties = sampled_smi_properties.drop_duplicates(subset=['attr_smi_orig']).reset_index(drop=True)

            print(f"Current total number of sampled molecules after filtering: {len(sampled_smi_can)}\n")
            sampling_round += 1

            # If timelimit is reached, do not continue sampling
            end_time_sampling = datetime.now()
            difference_time_sampling = end_time_sampling - start_time_sampling
            timelimit = int(ctx['main_config']['sample_timelimit'])
            if timelimit != 0 and difference_time_sampling.seconds > timelimit:
                break

            # Saving numerical properties to file
            output_properties = os.path.join(ctx['output_root'], 'filtering_properties.csv.gz')

            if len(sampled_smi_properties) > 0:

                # Only keep and save properties that are actually in sampled smi can
                sampled_smi_properties_mask = sampled_smi_properties['attr_smi_orig'].isin(sampled_smi_can)
                sampled_smi_properties = sampled_smi_properties.loc[sampled_smi_properties_mask].copy()

                if os.path.exists(output_properties):
                    sampled_smi_properties_previous = pd.read_csv(output_properties, compression='gzip', sep=',')
                    sampled_smi_properties = pd.merge(sampled_smi_properties_previous, sampled_smi_properties, how='outer')
                    sampled_smi_properties = sampled_smi_properties.drop_duplicates(subset=['attr_smi_orig']).reset_index(drop=True)

                sampled_smi_properties.to_csv(output_properties, header=True, index=None, compression='gzip')

            # If no molecules have been sampled, continue
            if len(sampled_smi_can) == 0:
                print("No molecules remain after filtering. Starting another sampling round.")
                continue

            # Creating new tranche file and adding sampled molecules
            with open(os.path.join(ctx['temp_dir'].name,
                                   tranches_iterations[ctx['iteration'] - 1] + '.tmp'), 'w') as f:
                for line in sampled_smi_can:
                    f.write(f"{line}\n")

            os.listdir(ctx['temp_dir'].name)

            add_ligand_names(
                os.path.join(ctx['temp_dir'].name, tranches_iterations[ctx['iteration'] - 1] + '.tmp'),
                "", int(ctx['main_config']['ligand_names_starting']), ctx['temp_dir'].name, 'txt')

            # Generating temporary (current iteration) output folder path
            output_folder = ctx['output_root']

            # If folder does not exist, create
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            shutil.copyfile(
                os.path.join(ctx['temp_dir'].name, tranches_iterations[ctx['iteration'] - 1] + '.txt'),
                os.path.join(output_folder, tranches_iterations[ctx['iteration'] - 1] + '.txt'))


def process_iteration(ctx, task):

    print(f"Current iteration: {task['iteration']}")
    print(f"")

    task['collection_file_tmp'] = os.path.join(task['collection_temp_dir'].name,
                                               tranches_iterations[ctx['iteration'] - 1] + '.tmp')

    with gzip.open(task['collection_file'], 'rb') as f_in:
        with open(task['collection_file_tmp'], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # If we only want to screen the input library
    if (ctx['iteration'] == 1 and ctx['main_config']['screen_first_only'] == "true"):

        # Generating temporary (current iteration) output folder path
        output_folder = ctx['output_root']
        task['collection_file_named'] = os.path.join(output_folder, tranches_iterations[
            ctx['iteration'] - 1] + '.txt')

        if os.path.isfile(task['collection_file_tmp']):
            ctx['main_config']['ligand_names_starting'] = str(
                int(ctx['main_config']['ligand_names_starting']) + file_length(task['collection_file_tmp']))

        add_ligand_names(task['collection_file_tmp'], '', int(ctx['main_config']['ligand_names_starting']),
                         task['collection_temp_dir'].name, 'txt')
        task['collection_file_tmp_named'] = os.path.join(task['collection_temp_dir'].name,
                                                         tranches_iterations[
                                                             ctx['iteration'] - 1] + '.txt')

        # Generating output (append if already exists from previously processed collection of this job)
        os.makedirs(output_folder, exist_ok=True)
        try:
            shutil.copyfile(task['collection_file_tmp_named'], task['collection_file_named'])
        except FileExistsError:
            with open(task['collection_file_tmp_named']) as f_src:
                with open(task['collection_file_named'], 'a') as f_dest:
                    shutil.copyfileobj(f_src, f_dest)

        if (ctx['main_config']['generative_model'] == "reinvent-randomized"):
            # Copying input model for next iteration
            input_models = sorted(glob.glob(ctx['main_config']['model_folder_path'] + "/*"), key=os.path.getmtime)
            input_model = input_models[len(input_models) - 1]
            shutil.copy(input_model, os.path.join(output_folder, 'model.trained'))

        return

    # Setting some folder paths
    task['training_file'] = os.path.join(task['collection_temp_dir'].name, 'training' + '.smi')
    task['training_folder'] = os.path.join(task['collection_temp_dir'].name, 'training', '')
    task['validation_file'] = os.path.join(task['collection_temp_dir'].name, 'validation' + '.smi')
    task['validation_folder'] = os.path.join(task['collection_temp_dir'].name, 'validation', '')

    # Generating folder paths for previously generated ligand collections
    output_folder = ctx['output_root']
    previous_collection_files = glob.glob(os.path.join(output_folder, '*.txt'))
    for previous_collection_file in previous_collection_files:
        os.remove(previous_collection_file)
        print(f"Deleted previous collection file {previous_collection_file} from temporary storage...")

    # Splitting into training and validation set for certain models, for others not
    if (ctx['main_config']['generative_model'] == "reinvent-randomized"):
        print(f"-----------------------------------------------------------------")
        print(f'Splitting input SMILES into training and validation set and')
        print(f"saving size of validation set...")

        ctx['csn'] = split_training_validation(task['collection_file_tmp'], task['training_file'],
                                               task['validation_file'], ctx['main_config']['validation_set_ratio'])
        print(f"Size of validation set: {ctx['csn']}")
        print(f"")
    else:
        shutil.copyfile(task['collection_file_tmp'], task['training_file'])

    if (ctx['main_config']['generative_model'] == "reinvent-randomized"):
        rr_run(ctx, task)

    elif (ctx['main_config']['generative_model'] == "stoned"):
        generate_molecules(ctx, task)

    else:
        print(f"Selected generative model currently not known, exiting...")
        exit(1)
