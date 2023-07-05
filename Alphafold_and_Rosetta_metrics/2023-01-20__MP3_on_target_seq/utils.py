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


#import sys; sys.path.insert(0, "/home/ajasja/projects/Stable_Heterodimers")
#import utils as u
#import sys; sys.path.insert(0, "/home/ajasja/projects/truncator/")
import truncator
from truncator import pymol_utils
import truncator as u


def parse_AF2_file_name(name):
    name = u.basename_noext(name)
    s = name.split('_')
    res = {}
    res['id1'] = s[0]
    res['id2'] = s[1]
    res['model'] = s[3]
    res['type'] = s[4]
    res['seed'] = s[6]
    res['base_name'] = u.basename_noext(name)[:-19]
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

