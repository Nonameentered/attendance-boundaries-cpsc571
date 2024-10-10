import csv
import os
import json

class Block_Info:
    def __init__(self, block_id, school_id):
        '''
        Constructor for a Block_Info object
        '''
        self.block_id = block_id
        self.school_id = school_id
    
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
    blocks_dict = dict()
    all_blocks = set()
    for row in csvreader:
        blocks_dict[row[4]] = Block_Info(row[4], row[19])
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
    Get the new and changed blocks for all states

    Inputs:
    * states_data_dict (dict): state names mapped to filenames

    Outputs:
    * list containing the dict of all the states mapped to new_blocks and changed_schools, set of all the new blocks, set of all the changed schools, set of all blocks
    '''
    changed_block_info = {}
    all_new_blocks = set()
    all_changed_schools = set()
    total_blocks = set()
    for state in states_data_dict:
        years_info = []
        for year in states_data_dict[state]:
            blocks_dict = get_block_info(year)
            if blocks_dict is None:
                continue
            years_info.append(blocks_dict)
            total_blocks = total_blocks | set(blocks_dict.keys())
        changed_info = compare_block_assignments_state(years_info[0], years_info[1])
        print(state)
        print('new_blocks: ', len(changed_info['new_blocks']))
        print('changed_blocks: ', len(changed_info['changed_schools']))
        changed_block_info[state] = changed_info
        all_new_blocks = all_new_blocks | set(changed_info['new_blocks'])
        all_changed_schools = all_changed_schools | set(changed_info['changed_schools'])
    return [changed_block_info, all_new_blocks, all_changed_schools, total_blocks]


def compare_block_assignments_state(year1, year2):
    '''
    Get the new and changed blocks between two snapshots for a specific state

    Inputs:
    * year1 (dict): block names mapped to Block_Info objects for snapshot 1
    * year2 (dict): block names mapped to Block_Info objects for snapshot 2

    Outputs:
    * dict containing the list of new_blocks and the list of changed_blcoks
    '''
    new_blocks = set()
    changed_schools = set()
    for block in (set(year1.keys()) | set(year2.keys())):
        if block not in year2.keys() or block not in year1.keys():
            new_blocks.add(block)
        elif year1[block].get_school_id() != year2[block].get_school_id():
            changed_schools.add(block)
    return {'new_blocks': list(new_blocks), 'changed_schools': list(changed_schools)}

if __name__ == "__main__":
    #get the segmentation between new and changed blocks
    states_data_dict = get_states_files('../../derived_data')
    data = compare_block_assignments(states_data_dict)
    
    #save file of new and changed blocks divided by state
    with open('/raw_results/state_block_data.json', 'w') as f:
        json.dump(data[0], f)

    #print total data
    print('----------------------------------------')
    print('Count of new blocks', len(data[1]))
    print('Count of blocks that changed schools', len(data[2]))
    print('Count of total blocks', len(data[3]))