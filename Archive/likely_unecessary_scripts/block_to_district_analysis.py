import csv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Block_Info:
    def __init__(self, block_id, school_id, district_id):
        '''
        Constructor for a Block_Info object
        '''
        self.block_id = block_id
        self.school_id = school_id
        self.district_id = district_id
    
    def get_school_id(self):
        '''
        Get school id of a block
        '''
        return self.school_id

    def get_block_id(self):
        '''
        Get block id of a block
        '''
        return self.school_id
    
    def get_district_id(self):
        '''
        Get district id of a block
        '''
        return self.district_id

def get_block_info(filename):
    '''
    Get all the blocks information from a specific filename

    Inputs:
    * filename (string): file to read from

    Outputs:
    * blocks_dict (dict): block ids mapped to Block_Info objects for that block
    '''
    try:
        file = open(filename)
    except:
        return
    csvreader = csv.reader(file)
    header = next(csvreader)
    #print(header[4], header[19], header[21])
    blocks_dict = dict()
    for row in csvreader:
        blocks_dict[row[4]] = Block_Info(row[4], row[19], row[21])
    return blocks_dict

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

def compare_block_assignments(states_data_dict):
    '''
    Get the districts with changed blocks for all states

    Inputs:
    * states_data_dict (dict): state names mapped to filenames

    Outputs:
    * list containing the dict of all the states mapped to new_blocks and changed_schools, set of all the new blocks, set of all the changed schools, set of all blocks
    '''
    all_districts_changed = set()
    changed_block_info = {}
    f = open('./district_info.json', "r")
    data1 = json.loads(f.read())

    for state in states_data_dict:
        years_info = []
        for year in states_data_dict[state]:
            blocks_dict = get_block_info(year)
            if blocks_dict is None:
                continue
            years_info.append(blocks_dict)
        changed_district_info = compare_block_assignments_state(years_info[0], years_info[1])
        for district in changed_district_info.keys():
            blocks_changed = changed_district_info[district]
            try:
                total = data1[district]
                changed_block_info[blocks_changed/total] = changed_block_info.get(blocks_changed/total, 0) + 1
            except:
                print('error')
        print(state)
        print('changed_districts: ', len(changed_district_info.keys()))
        all_districts_changed = all_districts_changed | set(changed_district_info.keys())
    return [changed_block_info, all_districts_changed]


def compare_block_assignments_state(year1, year2):
    '''
    Get the districtss with changed blocks between two snapshots for a specific state

    Inputs:
    * year1 (dict): block names mapped to Block_Info objects for snapshot 1
    * year2 (dict): block names mapped to Block_Info objects for snapshot 2

    Outputs:
    * dict containing the districts and count changed blocks
    '''
    districts = {}
    for block in (set(year1.keys()) | set(year2.keys())):
        if block not in year2.keys() or block not in year1.keys():
            try:
                district_id = year1[block].get_district_id()
            except:
                district_id = year2[block].get_district_id()
            #districts[district_id] = districts.get(district_id, 0) + 1
        elif year1[block].get_school_id() != year2[block].get_school_id():
            district_id = year1[block].get_district_id()
            districts[district_id] = districts.get(district_id, 0) + 1
    return districts

def ascii_histogram(data) -> None:
    """A horizontal frequency-table/histogram plot."""

    histogram_data = {}
    count = 0
    total = 0
    for frequency in data:
        total += len(data[frequency])
        count += len(data[frequency])
        if int(frequency) % 50 == 0:
            histogram_data[str(int(frequency) - 49) + '-' + str(frequency)] = total
            total = 0
    print(count)
        #histogram_data[int(frequency)] = len(data[frequency])

    for k in sorted(histogram_data):
        print(k, '+' * histogram_data[k])
        
if __name__ == "__main__":
    #get the district with changed blocks
    states_data_dict = get_states_files('../../derived_data')
    data = compare_block_assignments(states_data_dict)

    df = pd.DataFrame(data=data[0], index=[0])
    df = (df.T)
    df.to_excel('percentage_changed_schools_2.xlsx')
        
    #ascii_histogram(data[0])
    # #save file of new and changed blocks divided by state
    # with open('/raw_results/district_changed_data.json', 'w') as f:
    #     json.dump(data[0], f)

    #print total data
    print('----------------------------------------')
    print('Count of all changed districts', len(data[1]))