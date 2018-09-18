import subprocess
import os
import numpy as np
import pandas as pd




def runAFNI(inputstring):
    print(inputstring)
    subprocess.run(inputstring,shell=True, check=True)
    
    
def splitall(path):
    """
    splitall function from: https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch04s16.html
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def walklevel(some_dir, level=1):
    """
    works just like os_walk but one can specify how many levels it will go in directory tree
    from: https://stackoverflow.com/questions/229186/os-walk-without-digging-into-directories-below
    """
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def find_animal_folders(info, main_folder):        
    for animal in info.columns:
        for folderName, subfolders, filenames in walklevel(main_folder, 1):
            for file in filenames:
                if (file == 'subject' and 'Pilot data' not in folderName):
                    if animal in open(os.path.join(folderName,file)).read().lower():
                        info.loc['folder', animal] = folderName
    return info


def convertAll(folder):
    runAFNI(f'Bru2 -a {folder}/subject')