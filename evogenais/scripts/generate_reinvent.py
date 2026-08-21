#!python

import os
import shutil
import glob
import sys
import subprocess


# Main functions

def rr_sample_model(reinvent_folder, env, model, output_smi, number_of_smiles):
    # Get the current working directory
    cwd = os.getcwd()

    # Convert input/output paths to absolute paths
    model = os.path.join(cwd, model)
    output_smi = os.path.join(cwd, output_smi)

    # Change working directory to reinvent folder (relative to tools folder)
    os.chdir(reinvent_folder)

    # Set command depending on environment type
    if env[0] == "conda" or env[0] == "micromamba" or env[0] == "mamba":
        cmd = [env[0], 'run', '-n', env[1], 'python', './sample_from_model.py', '-m', model, '-o', output_smi,
               '-n',
               number_of_smiles]

    elif env[0] == "virtualenv":
        python_reinvent = os.path.join(env[1], 'bin', 'python3')
        directories = glob.glob(os.path.join(env[1], 'lib', 'python3*', 'site-packages', '*'))
        for directory in directories:
            sys.path.append(directory)
        cmd = [python_reinvent, './sample_from_model.py', '-m', model, '-o', output_smi, '-n',
               number_of_smiles]

    else:
        print(f"The setting reinvent_environment has begin with either \"conda:\" or \"virtualenv:\","
              f" but is {env[0]}:{env[1]}. Exiting...")
        exit(1)

    subprocess.run(cmd)

    # Go back to previous working dir
    os.chdir(cwd)


# Create empty model (get tokens, vocabulary); CAUTION: give file paths in relation to reinvent_folder
def rr_create_model(reinvent_folder, env, input_smi, output_model):
        # Set command depending on environment type
        if env[0] == "conda" or env[0] == "micromamba" or env[0] == "mamba":
                cmd = [env[0], 'run', '-n', env[1], 'python', os.path.join(reinvent_folder, 'create_model' + '.py'), '-i',
                           input_smi, '-o', output_model]

        elif env[0] == "virtualenv":
                python_reinvent = os.path.join(env[1], 'bin', 'python3')
                directories = glob.glob(os.path.join(env[1], 'lib', 'python3*', 'site-packages', '*'))
                for directory in directories:
                        sys.path.append(directory)
                cmd = [python_reinvent, os.path.join(reinvent_folder, 'create_model' + '.py'), '-i',
                           input_smi, '-o', output_model]

        else:
                print(f"The setting reinvent_environment has begin with either \"conda:\" or \"virtualenv:\","
                          f" but is {env[0]}{env[1]}. Exiting...")
                exit(1)

        subprocess.run(cmd)


# Train a model; CAUTION: give file paths in relation to reinvent_folder
def rr_train_model(reinvent_folder, env, input_model, output_model, training_smi_folder, number_of_epochs,
                                   learning_mode, output_tensorboard, validation_smi_folder, csn):
        # NOTE: csn needs to be <= size of initial validation set (before randomization)

        # TO BE DONE: check that gpu is available, otherwise raise error

        # Get the current working directory
        cwd = os.getcwd()

        # Convert input/output paths to absolute paths
        input_model = os.path.join(cwd, input_model)
        output_model = os.path.join(cwd, output_model)
        training_smi_folder = os.path.join(cwd, training_smi_folder)
        output_tensorboard = os.path.join(cwd, output_tensorboard)
        validation_smi_folder = os.path.join(cwd, validation_smi_folder)

        # Change working directory to reinvent folder (relative to tools folder)
        os.chdir(reinvent_folder)

        # Set command depending on environment type
        if env[0] == "conda" or env[0] == "micromamba" or env[0] == "mamba":
                cmd = [env[0], 'run', '-n', env[1], 'python', './train_model.py', '-i', input_model, '-o', output_model, '-s',
                           training_smi_folder, '-e', number_of_epochs, '--lrm', learning_mode, '--csl', output_tensorboard,
                           '--csv', validation_smi_folder, '--csn', csn]

        elif env[0] == "virtualenv":
                python_reinvent = os.path.join(env[1], 'bin', 'python3')
                directories = glob.glob(os.path.join(env[1], 'lib', 'python3*', 'site-packages', '*'))
                for directory in directories:
                        sys.path.append(directory)
                cmd = [python_reinvent, './train_model.py', '-i', input_model, '-o', output_model, '-s',
                           training_smi_folder, '-e', number_of_epochs, '--lrm', learning_mode, '--csl', output_tensorboard,
                           '--csv', validation_smi_folder, '--csn', csn]

        else:
                print(f"The setting reinvent_environment has begin with either \"conda:\" or \"virtualenv:\","
                          f" but is {env[0]}:{env[1]}. Exiting...")
                exit(1)



        subprocess.run(cmd)

        # Go back to previous working dir
        os.chdir(cwd)


# Increase data source by randomizing SMILES; CAUTION: give file paths in relation to reinvent_folder
def rr_randomize_smiles(reinvent_folder, env, input_smi, N, output_smi_folder):
        # Get the current working directory
        cwd = os.getcwd()

        # Convert input/output paths to absolute paths
        input_smi = os.path.join(cwd, input_smi)
        output_smi_folder = os.path.join(cwd, output_smi_folder)

        # Change working directory to reinvent folder (relative to tools folder)
        os.chdir(reinvent_folder)

        # Set command depending on environment type
        if env[0] == "conda" or env[0] == "micromamba" or env[0] == "mamba":
                cmd = [env[0], 'run', '-n', env[1], 'python', './create_randomized_smiles.py', '-i', input_smi, '-o',
                        output_smi_folder, '-n', N]

        elif env[0] == "virtualenv":
                python_reinvent = os.path.join(env[1], 'bin', 'python3')
                directories = glob.glob(os.path.join(env[1], 'lib', 'python3*', 'site-packages', '*'))
                for directory in directories:
                        sys.path.append(directory)
                cmd = [python_reinvent, './create_randomized_smiles.py', '-i', input_smi, '-o',
                           output_smi_folder, '-n', N]

        else:
                print(f"The setting reinvent_environment has begin with either \"conda:\" or \"virtualenv:\","
                          f" but is {env[0]}:{env[1]}. Exiting...")
                exit(1)

        subprocess.run(cmd)

        # Go back to previous working dir
        os.chdir(cwd)


def rr_run(ctx, task):
    from .generate_molecules import generate_molecules

    print(f"-----------------------------------------------------------------")
    print(f"Randomizing SMILES strings of training set...")
    print(f"")

    rr_randomize_smiles(ctx['main_config']['reinvent_folder_path'], ctx['main_config']['reinvent_environment'],
                        task['training_file'], ctx['main_config']['reinvent_randomize_multiplier'],
                        task['training_folder'])

    print(f"-----------------------------------------------------------------")
    print(f"Randomizing SMILES strings of validation set...")
    print(f"")

    rr_randomize_smiles(ctx['main_config']['reinvent_folder_path'], ctx['main_config']['reinvent_environment'],
                        task['validation_file'], ctx['main_config']['reinvent_randomize_multiplier'],
                        task['validation_folder'])

    print(f"-----------------------------------------------------------------")
    print(f"Get current model or create new model...")

    if not os.path.exists(task['collection_temp_dir'].name + '/models'):
        os.makedirs(task['collection_temp_dir'].name + '/models')

    current_model = os.path.join(task['collection_temp_dir'].name, 'models', 'model.' + str(task['iteration']))

    if (((int(task['iteration']) == 1) and (len(os.listdir(ctx['main_config']['model_folder_path'])) == 0)) or
            ctx['main_config']['learning_scheme'] == "retraining"):
        print(f"")
        print(f"Since no input model exists, creating new model...")
        print(f"")
        rr_create_model(ctx['main_config']['reinvent_folder_path'], ctx['main_config']['reinvent_environment'],
                        task['collection_file_tmp'], current_model)

    elif (int(task['iteration']) == 1) or (ctx['main_config']['learning_scheme'] == "transfer") or (
            ctx['main_config']['learning_scheme'] == "explore"):
        input_models = sorted(glob.glob(ctx['main_config']['model_folder_path'] + "/*"), key=os.path.getmtime)
        input_model = input_models[len(input_models) - 1]
        shutil.copyfile(input_model, current_model)

    elif (ctx['main_config']['learning_scheme'] == "reinforce"):
        input_model = os.path.join('./models', 'model.2')
        shutil.copyfile(input_model, current_model)

    else:
        print(
            f"Learning scheme selected not available ({ctx['main_config']['learning_scheme']})."
            f"learning_scheme has to be set to either reinforce, transfer, explore or retraining. Exiting...")
        exit(1)

    current_model_trained = os.path.join(current_model + '.trained')

    if (ctx['main_config']['learning_scheme'] == "explore"):
        current_model_final = current_model

    else:
        print(f"Performing training on model {current_model} and saving to {current_model_trained}...")

        rr_train_model(ctx['main_config']['reinvent_folder_path'], ctx['main_config']['reinvent_environment'],
                       current_model, current_model_trained, task['training_folder'],
                       ctx['main_config']['reinvent_number_of_epochs'],
                       ctx['main_config']['reinvent_learning_mode'],
                       os.path.join(task['collection_temp_dir'].name, 'tensorboard'),
                       task['validation_folder'], ctx['csn'])

        print(os.listdir(task['collection_temp_dir'].name + '/models'))
        print(f"")

        trained_models = sorted(glob.glob(current_model_trained + ".*"), key=os.path.getmtime)
        current_model_final = trained_models[len(trained_models) - 1]

    # Generating temporary (current iteration) output folder path
    output_folder = ctx['main_config']['output_tmp']

    # Replacing output of previous iteration by current iteration output
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Copying model for next iteration
    if (ctx['main_config']['generative_model'] == "reinvent-randomized"):
        shutil.copy(current_model_final, os.path.join(output_folder, 'model.trained'))

    # After training/refining the model, we can sample it
    generate_molecules(ctx, task, current_model_final)
