import csv
import os
import json
import numpy as np
import matplotlib.pyplot as plt

def get_block_info(filename, blocks_dict):
    '''
    Get all the blocks information from a specific filename

    Inputs:
    * filename (string): file to read from
    * blocks_dict (dict): district ids mapped to number of blocks

    Outputs:
    * blocks_dict (dict): district ids mapped to number of blocks

    '''
    try:
        file = open(filename)
    except:
        return
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        print(row[21])
        blocks_dict[row[21]] = blocks_dict.get(row[21], 0) + 1

def get_states_files(data_dir):
    '''
    Get the files for all the states corresponding to the blocks mapped to an elementary school

    Inputs:
    * data_dir (string): starting directory

    Outputs:
    * states_data_dict (dict): state names mapped to lists of file directories for a specific state
    '''
    listOfYears = os.listdir(data_dir)
    states_data_dict = {}
    for year in listOfYears:
        try:
            states = os.listdir(data_dir+'/'+year+'/')
        except:
            states = []
        for state in states:
            if state == '.DS_Store':
                continue
            if state not in states_data_dict:
                states_data_dict[state] = []
            states_data_dict[state].append(data_dir+'/'+year+'/'+state+'/'+'blocks_to_elementary.csv')
    return states_data_dict

def get_districts(states_data_dict):
    '''
    Get the number of blocks in every district for all states

    Inputs:
    * states_data_dict (dict): state names mapped to filenames

    Outputs:
    * the dict of all the districts mapped to number of blocks
    '''
    school_info = {}
    for state in states_data_dict:
        get_block_info(states_data_dict[state][1], school_info)
    return school_info

if __name__ == "__main__":
    #get the district with changed blocks
    states_data_dict = get_states_files('../../derived_data')
    data = get_districts(states_data_dict)

    print(data)

    #save file of new and changed blocks divided by state
    with open('/raw_results/district_info.json', 'w') as f:
        json.dump(data, f)

    print('Done')