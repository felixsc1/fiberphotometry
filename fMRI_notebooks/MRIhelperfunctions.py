import subprocess
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import papermill as pm
import shutil

from os.path import expanduser
home = expanduser("~")

# Creates an empty class, to group a bunch of variables.
# See https://www.oreilly.com/library/view/python-cookbook/0596001673/ch01s08.html
# doesnt work with papermill! non json seriliazable error
class Settings:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)




def runAFNI(inputstring, printout=True):
    if printout:
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


def find_animal_folders(info, raw_folder):        
    for animal in info.columns:
        for folderName, subfolders, filenames in walklevel(raw_folder, 1):
            for file in filenames:
                if (file == 'subject' and 'Pilot data' not in folderName):
                    if animal in open(os.path.join(folderName,file)).read().lower():
                        info.loc['folder', animal] = folderName
    return info

def getinfo(folders):
    info = pd.read_excel(f"{folders['main']}/{folders['excel']}", convert_float=True)
    info.columns = map(str.lower, info.columns)
    info.set_index('scan', inplace=True)
    # info = info.append({'animal-id': 'folder'}, ignore_index=True)

    # def get_subject_folders(main_folder, info):
    #     for animal in info.columns

    info = find_animal_folders(info, folders['raw'])
    return(info)


def convertAll(folder_in, folder_out):
    """
    requires original paravision folder structure with subject file.
    requires Bru2 from github in PATH
    converts all 2dseqs, except tripilots, with original dimensions as .nii to folder_out
    """
    runAFNI(f'Bru2 -a -o {folder_out} {folder_in}/subject')
    
    

def check_and_convert(folders, animal):
    """
    Is there already a folder for this animal in the main analysis folder?
    If not, create it and convert all 2dseqs of this animal and move them to analysis subfolder
    """
    animal_output_folder = os.path.join(folders['analysis'],animal)
    if not os.path.exists(animal_output_folder):
        os.makedirs(animal_output_folder)
        convertAll(folders['animal'], animal_output_folder)

        
        
def simple_coreg(template, scanA, scanB, out_dir):
    """
    scanA will be coregistered to template, identical transformation will be applied to scanB (no motion between A and B!)
    """
    parameters = os.path.join(out_dir,'1dparams.1D')
    scanAloc = f"{out_dir}/A_coreg.nii"
    scanBloc = f"{out_dir}/B_coreg.nii"
    runAFNI(f"3dAllineate -base {template} -source {scanA} -prefix {scanAloc} -cost ls -zclip -interp quintic -final wsinc5 -twopass -twoblur 2 -fineblur 0.5 -nmatch 80% -conv 0.01 -1Dparam_save {parameters}")
    runAFNI(f"3dAllineate -1Dparam_apply {parameters} -source {scanB} -prefix {scanBloc} -cost ls -zclip -interp quintic -final wsinc5 -twopass -twoblur 2 -fineblur 0.5 -nmatch 80% -conv 0.01 -master {template}")
    
    return scanAloc, scanBloc
    

def coreg_epi(template, scanA, scanB, out_dir):
    """
    difference to simple_coreg: also does -tshift,
    volreg off as global bolus inflow could be mistaken for movement.
    WARNING: AFNI gives error for some reason
    """
    runAFNI("python2.7 " + home + "/abin/align_epi_anat.py -dset1to2 -dset1 " + scanA + " -dset2 " + template + \
                                " -child_dset1 " + scanB + \
                                " -output_dir " + out_dir + \
                                " -dset1_strip None -dset2_strip None" \
                                " -overwrite" \
                                " -big_move" \
                                " -volreg off" \
                                " -Allineate_opts '-maxrot 10 -maxshf 3 -conv 0.005 -twofirst -twoblur 0.8 -source_automask+2 -final wsinc5'" \
                                " -tshift on -tshift_opts '-tzero 0 -quintic'" \
                                " -suffix .coreg" \
                                " -save_vr" \
                                " -cost ls")


def save_nifti(data, dimensions, folders, animal, prefix):
    """
    Input:  data is flattened np array.
            dimension contains matrix size and nifti affine matrix.
            folders contains output folder.
            animal is animal ID.
            prefix will be combined with animal ID to filename
    Output: .nii file in output folder. filename/location is returned
    """
    data_reshaped = data.reshape(dimensions.x, dimensions.y, dimensions.z, -1)
    data_img = nib.Nifti1Image(data_reshaped, dimensions.affine)
    out_file = os.path.join(folders['out'],f'{prefix}_{animal}.nii')
    nib.save(data_img, out_file)
    return out_file


def run_module(folders, moduleName, moduleInputs, overwrite = True):
    """
    Expects that there is a notebook called "moduleName.ipynb" in the folders['notebooks'] folder for the analysis.
    moduleInputs are the parameters forwarded to the notebook.
    overwrite:  When set to True: will always run analysis for every animal, even when it has already run.
    When set to False: Will only run analysis for animals that don't have an output notebook in their data folder yet (will print warning). Warning: Even with overwrite = True, some AFNI steps may not overwrite already existing files, or even crash the analysis. Better delete the files manually beforehand.
    """
    print(f'running {moduleName} analysis...')
    out_notebook = os.path.join(folders['animal'],f'{moduleName}_output.ipynb')
    if (os.path.exists(out_notebook) and not overwrite):
        print(f'animal already analyzed (delete {out_notebook} to re-run analysis)')
        return 
    pm.execute_notebook(
       os.path.join(folders['notebooks'],f'{moduleName}.ipynb'),
       os.path.join(folders['animal'],f'{moduleName}_output.ipynb'),
       parameters = moduleInputs
    )
    

def folder_check_create(base, new):
    """
    Checks if a folder with name 'new' (string) already exists within folder 'base' (full path)
    if not, create it.
    returns the newly created folder (full path)
    """
    newfullfolder = os.path.join(base,new)
    if not os.path.exists(newfullfolder):
        os.makedirs(newfullfolder)
    return newfullfolder

def copy_notebook_outputs(folders, animal, modality, info):
    """
    creates a new folder with name of modality inside groupstats folder (if needed)
    opens notebook of animal which is expected to be in the animals data folder and called 'modality'_output
    for each entry in the notebooks papermill data, will copy the corresponding file and make an entry of its location in info dataframe 
    """
    group_modality_folder = folder_check_create(folders['group'], modality)
    folders[f"group_{modality}"] = group_modality_folder

    notebook = pm.read_notebook(os.path.join(folders['animal'],f'{modality}_output.ipynb'))
    for key, value in notebook.data.items():
        info.loc[key, animal] = value
        path, file = os.path.split(value)
        output = os.path.join(group_modality_folder, file)
        shutil.copy(value, output)
        
    return info, folders


def select_for_group_average(folder, info, map_names, exclude='xxx343434343', row='genotype'):
    """
    Looks at all nii files in folder,
    first filters by prefix string e.g. CBV_map and string to exclude (e.g. male)
    then groups the results based on a row in the info dataframe (e.g. genotype)
    output are two lists group1, group2 with the full path of all files in a group.
    
    Limitations: Must have exactly 2 groups, must be encoded with 1 or 2.
    """
    allfiles = os.path.join(folder,'*.nii')

    files = [fn for fn in glob.glob(allfiles)
            if exclude not in os.path.basename(fn) and
            os.path.basename(fn).startswith(map_names)]

    x = info.loc[row,:].to_dict()

    x_1 = {k : v for k,v in x.items() if v == 1.0}
    x_2 = {k : v for k,v in x.items() if v == 2.0}

    group1 = []
    group2 = []
    for file in files:
        for filtered in x_1.keys():
            if filtered in file:
                group1.append(file)
        for filtered in x_2.keys():
            if filtered in file:
                group2.append(file)
    return group1, group2


def average_map(map_name, out_folder, group1, group2):
    """
    OBSOLETE: using 3dttest++ also calculates the mean maps for each group as well as difference map.
    created with the WET (write everything twice) principle :(
    """
    outfile1 = os.path.join(out_folder,f'mean{map_name}_group1.nii')
    filelist_formatted = " ".join(map(str, group1))
    runAFNI(f"3dMean -prefix {outfile1} {filelist_formatted}", printout=False)
    print(f"created {outfile1}")
    
    outfile2 = os.path.join(out_folder,f'mean{map_name}_group2.nii')
    filelist_formatted = " ".join(map(str, group2))
    runAFNI(f"3dMean -prefix {outfile2} {filelist_formatted}", printout=False)
    print(f"created {outfile2}")
    
    # also create the difference map:
    outfile3 = os.path.join(out_folder,f'mean{map_name}_group1_MINUS_group2.nii')
    runAFNI(f"3dcalc -prefix {outfile3} -a {outfile1} -b {outfile2} -expr 'a-b'", printout=False)
    print(f"created {outfile3}")


def ttest_2groups(outfolder, map_name, group1, group2, blur=0.0):
    outfile = os.path.join(outfolder,f'mean{map_name}_Ttest.nii')
    filelist_formatted1 = " ".join(map(str, group1))
    filelist_formatted2 = " ".join(map(str, group2))
    runAFNI(f"3dttest++ -prefix {outfile} -setA {filelist_formatted1} -labelA group1 -setB {filelist_formatted2} -labelB group2 -toz  -exblur {blur} -overwrite", printout=False)
    print(f"created {outfile}")
    
    
def custom_detrend(x, baseline):
    """
    For very common problem:
    Linear (or polynomial) detrending of time-series x.
    Fit is only performed for the initial baseline period (units for bsl are just array points, not seconds!)
    but extrapolated fit curve is subtracted over whole time series x (mean is re-added).
    Afni tools etc only perform fits on whole time series.
    """ 
    t = np.arange(0,x.size)
    t_bsl = np.arange(0,baseline)
    fit = np.polyfit(t_bsl, x[0:baseline], 1)
    trendline = np.polyval(fit, t)
    x_detrended = x - trendline + np.mean(x)
    return x_detrended, trendline