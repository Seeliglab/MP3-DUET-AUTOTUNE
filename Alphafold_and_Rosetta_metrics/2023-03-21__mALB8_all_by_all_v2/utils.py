import os
from os.path import basename
import pandas
import pandas as pd
import numpy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import ttest_ind

from glob import glob
from pathlib import Path
import re

#import sys; sys.path.insert(0, "/home/ajasja/projects/Stable_Heterodimers")
#import utils as u
#import sys; sys.path.insert(0, "/home/ajasja/projects/truncator/")
import truncator
from truncator import pymol_utils
import truncator as u


def parse_AF2_file_name(name):
    path = Path(name)
    res = {}
    res['full_name_path'] = name
    
    res['stem_name'] = path.stem
    res['dir'] = path.parent
    name = res['stem_name'] 

    #print(name)
    rank = get_token_value(name, 'rank_')
    model_number = rank = get_token_value(name, 'model_')
    #print(rank)

    find_ids_re =  r"(.*)___(.*)_scores_"


    match = re.match(find_ids_re, name)


    #print(match.group(1))
    res['base_name'] = name.replace('|','__')
    res['id1'] = match.group(1)
    res['id2'] = match.group(2)
    res['rank'] = int(rank)
    res['model_number'] = int(model_number)

    
    return res 
#parse_AF2_file_name('OUT_DIR/P7_P3_model_4_ptm_seed_0_prediction_results.json')

def load_AF2_json_data(file_name):
    data_dict = u.read_json(file_name)
    name_dict = parse_AF2_file_name(file_name)
    name_dict.update(data_dict)
    return name_dict

#load_AF2_json_data('OUT_DIR/P7_P3_model_4_ptm_seed_0_prediction_results.json')


def get_token_value(astr, token):
    """returns value next to token"""
    import re
    regexp = re.compile(f'{token}(\d*)')
    match = regexp.search(astr)
    return match.group(1)

#get_token_value('01a_monomer__r6__msad512', 'msad')

def print_layers(row, selections="core bdry surf".split(), colors="red orange yellow".split()):
    for sele, col  in zip(selections, colors):
        print_sele(row, sele, col)

def print_selection(row, selection):
    full_name = row.full_name
    
    print("delete all")
    print(f"load {full_name}") 
    if truncator.is_str(selection):
        selection = [selection]
    for sele in selection:
        print_sele(row, sele, None)
    print_layers(row)
    

    print("orient")   

def print_sele(row, name, color=None, remove_chains=False):
    try:
        sele = row[name+"_pymol_selection"]
        sele=sele.replace("rosetta_sele", name)
        if remove_chains:
            sele=sele.replace("chain A and", "").replace("chain B and", "")
        print(sele)
        if not color is None:
            print(f"color {color}, {name}")
    except:
        pass


def selection_to_count(sele_str):
    """Takes a pymol selection and returns the number of selected residues. Works only for selections geenrated using the simple pymol metric"""
    return sele_str.count(',')

