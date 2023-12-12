import pathlib2
import itertools
from os import listdir
from os.path import join, splitext

def load_file(file_path, number_rows, separator):
    arr_info = []
 
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            line_info = line.split(separator)
            if len(line_info) != number_rows:
                print('error in line', line,  'number of rows different to', number_rows)
                exit(1)
            arr_info.append(line_info)
    
    return arr_info

def findsubsets(S):
    sets_list = []
    for i in range(1, len(S) + 1, 1):
        s = list(itertools.combinations(S, i))
        sets_list.extend(s)
    return sets_list

def create_reps(dataset_name):

    rows_number = 4
    separator = ' '
    int_to_char = {0:'1V', 1:'2V', 2:'V', 3:'C'}
    sets_list = findsubsets([0, 1, 2, 3])
    
    path_long_rep = join('long_rep', dataset_name)
    print('Path to long representation:', path_long_rep)
    
    for candidate in listdir(path_long_rep):
        print('\nCandidate:', candidate)
        path_candidate = join(path_long_rep, candidate)
        
        for pitch_score in listdir(path_candidate):
                
            if splitext(pitch_score)[1][1:] != 'ptch':
                continue 
            print('Score:', pitch_score)
            
            path_pitch_score = join(path_candidate, pitch_score)
            matrix_pitchs = load_file(path_pitch_score, rows_number, separator)
        
            for index,S in enumerate(sets_list):
            
                instrument_indexes = set()
                instrument_labels = ''
                
                for i in S:
                    instrument_indexes.add(i)
                    instrument_labels = instrument_labels + int_to_char[i] + '-'
                if index < 9:
                    instrument_labels = '0' + str(index + 1) + '. ' + instrument_labels[0: len(instrument_labels) - 1]
                else:
                    instrument_labels = str(index + 1) + '. ' + instrument_labels[0: len(instrument_labels) - 1]
                
                print('Processing subset:', instrument_labels)
                pathlib2.Path(join('long_rep', 'midi_pitch', instrument_labels, candidate)).mkdir(parents=True, exist_ok=True)
                pathlib2.Path(join('long_rep', 'class_pitch', instrument_labels, candidate)).mkdir(parents=True, exist_ok=True)
                
                output_file = join('long_rep', 'midi_pitch', instrument_labels, candidate, splitext(pitch_score)[0]+'.midi_pitch')
                with open(output_file, 'w') as ff:
                    for row in matrix_pitchs:
                        
                        new_row = []
                        for i,elem in enumerate(row):
                        
                            if i in instrument_indexes:
                                new_row.append(elem + ' ')
                                
                        ff.write(''.join(new_row)+'\n')
                        
                output_file = join('long_rep', 'class_pitch', instrument_labels, candidate, splitext(pitch_score)[0]+'.class_pitch')
                with open(output_file, 'w') as ff:
                    for row in matrix_pitchs:
                        
                        new_row = []
                        for i,elem in enumerate(row):
                        
                            if i in instrument_indexes:
                                if elem != 'r':
                                    elem = int(elem)%12
                                
                                new_row.append(str(elem) + ' ')
                                
                        ff.write(''.join(new_row)+'\n')
                        
    print(':)')

create_reps('HM_s')