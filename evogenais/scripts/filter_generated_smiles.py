#!python3

import pandas as pd
import os
import sys
from rdkit import Chem
import rdkit.Chem as rdc
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import BRICS
import rdkit.Chem.rdmolops as rdcmo
import json
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import FilterCatalog
import rdkit.Chem.Descriptors as rdcd
import rdkit.Chem.rdMolDescriptors as rdcmd
import rdkit.Chem.Descriptors as Descriptors
import requests
from pathlib import Path
from rdkit.Chem.Crippen import MolLogP


_filters = None
_dic = None

def _resolve_data_path(ctx, key):
    path = Path(ctx['main_config'][key]).expanduser()
    if not path.is_absolute():
        path = Path(ctx['main_config']['_config_dir']) / path
    return path.resolve()

def _load_filter_databases(ctx):
    global _filters, _dic
    if _filters is not None and _dic is not None:
        return
    mcf = pd.read_csv(_resolve_data_path(ctx, 'mcf_file'))
    pains = pd.read_csv(_resolve_data_path(ctx, 'wehi_pains_file'), names=['smarts', 'names'])
    combined = pd.concat([mcf, pains]).sort_values('smarts')
    _filters = [Chem.MolFromSmarts(x) for x in combined['smarts'].values]
    with _resolve_data_path(ctx, 'pains_file').open('r', encoding='utf-8') as handle:
        substructures = [line.rstrip().split(' ') for line in handle if line.strip()]
    _dic = {line[0]: line[1] for line in substructures}


# Helper functions

def gpusim_search_http(smi, dbname, url, similarity_cutoff, return_count):
    payload = {
        "smiles": smi,
        "similarity_cutoff": similarity_cutoff,
        "return_count": return_count,
        "dbnames": dbname
    }
    # Send the POST request using requests
    response = requests.post(url, data=payload)
    return response.text


def maximum_ring_size(mol):
    """
    Calculate maximum ring size of molecule
    """
    cycles = mol.GetRingInfo().AtomRings()
    if len(cycles) == 0:
        maximum_ring_size = 0
    else:
        maximum_ring_size = max([len(ci) for ci in cycles])
    return maximum_ring_size


def maximum_ringsystem_size(mol, includeSpiro=False):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon > 1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
        max_length_ringsystem = max(len(s) for s in systems)
    return max_length_ringsystem


def maximum_ring_membership(mol):
    # Get the ring information
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    # Identify bridgehead atoms
    bridgehead_atoms = set()
    for ring in atom_rings:
        if len(ring) == 2:
            bridgehead_atoms.update(ring)

    # Initialize a dictionary to count rings each atom is part of
    atom_ring_count = {atom_idx: 0 for atom_idx in range(mol.GetNumAtoms())}

    # Count the number of rings each atom is part of, excluding bridgeheads
    for ring in atom_rings:
        for atom_idx in ring:
            if atom_idx not in bridgehead_atoms:
                atom_ring_count[atom_idx] += 1

    # Find the maximum number of rings any non-bridgehead atom is part of
    max_ring_membership = max(atom_ring_count.values())

    return max_ring_membership


def filter_phosphorus(mol):
    """
    Check for presence of phopshorus fragment
    Return True: contains proper phosphorus
    Return False: contains improper phosphorus
    """
    violation = False

    if mol.HasSubstructMatch(rdc.MolFromSmarts("[P,p]")) == True:
        if mol.HasSubstructMatch(rdc.MolFromSmarts("*~[P,p](=O)~*")) == False:
            violation = True

    return violation


def substructure_violations(mol):
    """
    Check for substructure violates
    Return True: contains a substructure violation
    Return False: No substructure violation
    """
    violation = False

    forbidden_fragments = [
        "*1=**=*1",
        "*1*=*=*1",
        "*1~*=*1",
        "[F,Cl,Br]C=[O,S,N]",
        "[Br]-C-C=[O,S,N]",
        "[N,n,S,s,O,o]C[F,Cl,Br]",
        "[I]",
        "[S&X3]",
        "[S&X5]",
        "[S&X6]",
        "[B,N,n,O,S]~[F,Cl,Br,I]",
        "*=*=*=*",
        "*=[NH]",
        "[P,p]~[F,Cl,Br]",
        "SS",
        "C#C",
        "C=C=C",
        "C=C=N",
        "NNN",
        "[*;R1]1~[*]~[*]~[*]1",
        "OOO",
        "[#8]1-[#6]2[#8][#6][#8][#6]12",  # Epoxide group
        "N=C=O",  # Isocyanate group
        "C1CN1",  # Aziridine group
        "[#6](=[#8])[F,Cl,Br,I]",  # Acyl halides
        "[#6](=[#8])=[#6](-[#8])-[#6](=[#8])~[#8]",  # Quinone
        "N(-[#6])=[#7]-[#8]"  # Nitrosamine
    ]

    for ni in range(len(forbidden_fragments)):

        if mol.HasSubstructMatch(rdc.MolFromSmarts(forbidden_fragments[ni])) == True:
            # print('Violation frag: {} smi: {}'.format(forbidden_fragments[ni],  Chem.MolToSmiles(mol)) )
            violation = True
            break
        else:
            continue

    return violation


def passes_wehi_mcf(smi):
    mol = Chem.MolFromSmiles(smi)
    h_mol = Chem.AddHs(mol)
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    else:
        return True


def pains_filt(mol):
    for k, v in _dic.items():
        subs = Chem.MolFromSmarts(k)
        if subs != None:
            if mol.HasSubstructMatch(subs):
                mol.SetProp(v, k)
    return [prop for prop in mol.GetPropNames()]


def apply_filters(smi, ctx):
    _load_filter_databases(ctx)
    try:
        if 'Si' in smi or 'Sn' in smi:  # Atoms not appropriate for docking calculations
            return False

        mol = smi2mol(smi)

        # Added after GDB-13 was filtered to get rid charged molecules
        if rdcmo.GetFormalCharge(mol) != 0:
            return False
        # Added after GDB-13 was filtered to get rid radicals
        elif rdcd.NumRadicalElectrons(mol) != 0:
            return False
        # Filter by bridgehead atoms
        elif rdcmd.CalcNumBridgeheadAtoms(mol) > 2:
            return False
        # Filter by ring size
        elif maximum_ring_size(mol) > 8:
            return False
        # Filter by ring system size
        elif maximum_ringsystem_size(mol) > 14:
            return False
        # Filter by ring membership
        elif maximum_ring_membership(mol) > 2:
            return False
        # Filter by proper phosphorus
        elif filter_phosphorus(mol):
            return False
        elif substructure_violations(mol):
            return False
        elif Descriptors.NumRotatableBonds(mol) > 10:
            return False
        elif Descriptors.TPSA(mol) > 140:
            return False
        elif passes_wehi_mcf(smi) == False:
            return False
        elif len(pains_filt(mol)) != 0:
            return False
        else:
            return True
    except Exception:
        return False


# Main functions

def filter_generated_smiles_worker(args):
    ctx, smi, rdkit_catalog_pickle, gpusim_database, gpusim_url = args

    mol = smi2mol(smi)

    if mol is None or smi == '':
        return

    else:
        smi_properties = {}

        if (ctx['main_config']['gpusim_filter'] == "true"):

            if (ctx['main_config']['gpusim_filter_fragments'] == "true"):
                try:
                    fragments = list(BRICS.BRICSDecompose(mol, allNodes=None, minFragmentSize=2, returnMols=True))
                except Exception as e:
                    return

                fragments_cleaned = []
                for f in fragments:
                    f_cleaned = rdcmo.DeleteSubstructs(f, Chem.MolFromSmiles('[*]'))
                    fragments_cleaned.append(Chem.MolToSmiles(f_cleaned, canonical=True))

                fragments_found = []
                for f_clean in fragments_cleaned:
                    if f_clean is None or f_clean == '' or Chem.MolFromSmiles(f_clean) is None:
                        return
                    else:
                        try:
                            gpusim_result = json.loads(gpusim_search_http(f_clean, gpusim_database, gpusim_url,
                                                                          ctx['main_config']['gpusim_cutoff'],
                                                                          ctx['main_config']['gpusim_return_count']))
                        except Exception:
                            return

                        if gpusim_result['approximate_count'] == 0:
                            return
                        else:
                            fragments_found.append(gpusim_result['results'][0][1])

                smi_properties['fragments_smi'] = str(fragments_found)[1:-1].replace(", ", ".").replace("'", "")

            else:
                try:
                    gpusim_result = json.loads(gpusim_search_http(smi, gpusim_database, gpusim_url,
                                                                  ctx['main_config']['gpusim_cutoff'],
                                                                  ctx['main_config']['gpusim_return_count']))
                except Exception:
                    return

                if gpusim_result['approximate_count'] == 0:
                    return

        if (ctx['main_config']['filter_general'] == "true"):
            if not apply_filters(smi, ctx):
                return

        if (ctx['main_config']['filter_rdkit_catalog'] == "true"):
            rdkit_catalog = FilterCatalog.FilterCatalog(rdkit_catalog_pickle)
            if rdkit_catalog.HasMatch(mol):
                print("Warning: molecule failed filter", file=sys.stderr)
                # more detailed
                entry = rdkit_catalog.GetFirstMatch(mol)
                if entry:
                    print("Warning: molecule failed filter: reason %s" % (
                        entry.GetDescription()), file=sys.stderr)
                return

        if (ctx['main_config']['filter_mw'] == "true"):
            try:
                mw = MolWt(mol)
                if mw < float(ctx['main_config']['filter_mw_min']) or mw > float(
                        ctx['main_config']['filter_mw_max']):
                    return
                else:
                    smi_properties['mw_rdkit'] = mw
            except Exception:
                return

        if (ctx['main_config']['filter_logp_rdkit'] == "true"):
            try:
                logp = MolLogP(mol)
                if logp < float(ctx['main_config']['filter_logp_min']) or logp > float(
                        ctx['main_config']['filter_logp_max']):
                    return
                else:
                    smi_properties['logp_rdkit'] = logp
            except Exception:
                return

        canon_smi = mol2smi(mol, canonical=True)

        if canon_smi is not None:
                smi_properties['attr_smi_orig'] = canon_smi
                return canon_smi, smi_properties
