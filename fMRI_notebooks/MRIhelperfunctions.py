import subprocess, os, glob
import numpy as np
import pandas as pd
import nibabel as nib
import papermill as pm
import shutil
from lmfit.models import ExpressionModel
import http.client, urllib
import re
import fnmatch
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

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

                         
def check_and_convert_single(folders):
    """
    Is there already a folder for this animal in the main analysis folder?
    If not, create it and convert all 2dseqs of this animal and move them to analysis subfolder
    """
    folders['analysis'] = os.path.join(folders['animal'],'analysis')
    folders['scan'] = os.path.join(folders['analysis'],folders['scanNumber'])
    actualinput = os.path.join(folders['animal'],folders['scanNumber'])
    folders['raw'] = f"{folders['scan']}/X{folders['scanNumber']}P1.nii"
    if not os.path.exists(folders['scan']):
        os.makedirs(folders['scan'])
        convertAll(actualinput, folders['scan'])
    return folders
                       
        
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

                         
def get_single_timepoint(scan):
    outfile = os.path.join(os.path.dirname(scan), 'single_timepoint.nii')             
    runAFNI(f"3dTcat -overwrite -prefix {outfile} {scan}'[0]'")
    return outfile
                     
                         
from os.path import expanduser
def coreg_epi_to_template(template, scan):
    single_timepoint = get_single_timepoint(scan)
    scanpath = os.path.dirname(scan)
    os.chdir(scanpath) # somehow it would otherwise always store files in wrong folder with python2.7 code
    homepath = expanduser("~")
    runAFNI("python2.7 " + homepath + "/abin/align_epi_anat.py -dset2to1 -dset1 " + template + " -dset2 " + single_timepoint + \
                                    " -child_dset2 " + scan + \
                                    " -dset1_strip None -dset2_strip None"  \
                                    " -overwrite" \
                                    " -big_move" \
                                    " -volreg on -volreg_method 3dvolreg" \
                                    " -volreg_opts '-Fourier -zpad 1 -dfile dfile.1D'" \
                                    " -Allineate_opts '-maxrot 10 -maxshf 3 -conv 0.005 -twofirst -twoblur 0.8 -source_automask+2 -final wsinc5'" \
                                    " -tshift on -tshift_opts '-tzero 0 -quintic'" \
                                    " -dset2_base 0" \
                                    " -volreg_base 0" \
                                    " -suffix .volreg" \
                                    " -cost ls")
    
    outfile = scan[:-4] + '.volreg+orig.BRIK'   # <-- remove .nii part                     
    return outfile
                         
                         
                         
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
    Looks at all *.nii files in folder,
    first filters by prefix string e.g. CBV_map and string to exclude (e.g. male)
    then groups the results based on a row in the info dataframe (e.g. genotype)
    output are two lists group1, group2 with the full path of all files in a group.
    
    Limitations: Must have exactly 2 groups, genotype row in info df must be encoded with 1 or 2.
    """
    allfiles = os.path.join(folder,'*.nii')

    files_a = [fn for fn in glob.glob(allfiles) if os.path.basename(fn).startswith(map_names)]
    
    files=[]
    for file in files_a:
        if any([animal in os.path.basename(file) for animal in exclude]):
            print(f'skipping {os.path.basename(file)}')
            continue
        else:
            files.append(file)
    
#         files = [fn for fn in glob.glob(allfiles)
#             if not exclusion_logic(exclude, fn) and
#             os.path.basename(fn).startswith(map_names)]
    
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
    created according to the WET (write everything twice) principle :(
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
    if blur == 0:  # e.g. for 2D datasets, which don't allow exblur option
        runAFNI(f"3dttest++ -prefix {outfile} -setA {filelist_formatted1} -labelA group1 -setB {filelist_formatted2} -labelB group2 -toz -overwrite", printout=False)
    else:
        runAFNI(f"3dttest++ -prefix {outfile} -setA {filelist_formatted1} -labelA group1 -setB {filelist_formatted2} -labelB group2 -toz  -exblur {blur} -overwrite", printout=False)
    print(f"created {outfile}")
    
    
def custom_detrend(x, baseline):
    """
    For very common problem:
    Linear (or polynomial) detrending of time-series x.
    Fit is only performed for the initial baseline period (units for bsl are just array points, not seconds!)
    but extrapolated fit curve is subtracted over whole time series x (mean is re-added).
    AFNI and co only perform fits on whole time series.
    """ 
    t = np.arange(0,x.size)
    t_bsl = t[0:baseline]
    fit = np.polyfit(t_bsl, x[0:baseline], 1)
    trendline = np.polyval(fit, t)
    x_detrended = x - trendline + np.mean(x)
    return x_detrended, trendline

def custom_detrend_exp(x, baseline, endpoints):
    """
    Variation of custom_detrend()
    """ 
    t = np.arange(0,x.size)
    t_bsl = t[0:baseline]
    t_end = t[-endpoints:]
    t_combine = np.append(t_bsl, t_end)
    x_bsl = x[0:baseline]
    x_end = x[-endpoints:]
    x_combine = np.append(x_bsl, x_end)
    
    x_log = np.log(x_combine)
    fit = np.polyfit(t_combine, x_log, 1)
    trendline = np.polyval(fit, t)
    x_detrended = x - np.exp(trendline) + np.mean(x)
    return x_detrended, trendline




def exp_fits(x,x0,xend,t_res):
    """
    fits an exponential signal increase starting from x0 to xend, optionally with an exponential washout term of the contrast agent.
    output contains various parameters computed from the fit
    """
    # model = ExpressionModel('B*(1 - exp(-ktrans * (x)))')
    model = ExpressionModel('B*(1 - exp(-ktrans * (x))) * exp(-w * x)', nan_policy='propagate') # with washout

    params = model.make_params()
    params['ktrans'].set(value=0.001,min=0.0001, max=0.1)
    params['B'].set(value=0.1, min=0.01, max=50)
    params['w'].set(value=0.0001,min=0, max=0.001)


    signal = x[x0:xend]-x[x0]

    time = np.arange(0, signal.size, 1)*t_res

    result = model.fit(signal, params, x=time)

    slope = result.best_values['B']*result.best_values['ktrans']  #derivative of model function at point x=0
#     integral = np.sum(result.best_fit)
    maximum = np.max(result.best_fit)
    TTP = np.argmax(result.best_fit) * t_res
    
    return result, slope, maximum, TTP, signal, time

def pushover(message, recipient = 'Huawei'):

    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
      urllib.parse.urlencode({
        "token": "axB8uLxeVed3b32AVcvd6QfdLzEfS4",
        "user": "uWqdKSNezFzvHjcSEFcxTWySEzCtur",
        "device": recipient,
        "message": message,
      }), { "Content-type": "application/x-www-form-urlencoded" })
    conn.getresponse()


def extract_roi(data, roi, name='ROIdata'):
    """
    Inputs must be full directories to the files.
    Optional name of file can be given.
    Output is stored in same directory as data.
    Returns path to the roi file.
    """
    folder = os.path.dirname(data)
    filename = name + '.1D'
    outfile = os.path.join(folder,filename)
    runAFNI(f'3dROIstats -mask {roi} {data} > {outfile}', printout=False)
    return outfile


def create_clean_csv(path, csvname='roi_all.csv', measurement=''):
    """
    Only works for 1.D ROI files generated with AFNI 3dROIstats AND
    ROI files MUST be generated (named correctly) with name_rois() function here.
    
    For emergencies, the regular expression parts can be commented out, 
    to just get an unsorted excel sheet should work with every kind of data then.
    
    1. Searches for all the .1D files in path.
    2. Puts them all into one big dataframe
    3. adds column for group 1 or group 2 (only 2 group data supported)
    4. adds column 'animal' with animal ID
    5. adds column 'measurement' with type of measurement
    6. stores dataframe as .csv and returns the dataframe
    
    """
    frame = []
    allFiles = glob.glob(path + "/*.1D")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_, index_col=0, sep='\t', header=0)
        if fnmatch.fnmatch(file_, '*group1*'):
            df['group'] = 1
        elif fnmatch.fnmatch(file_, '*group2*'):
            df['group'] = 2
        else:
            df['group'] = np.nan
            print(os.path.splitext(os.path.splitext(file_)[0]))

        list_.append(df)
    frame = pd.concat(list_)
    frame.reset_index(level=0, inplace=True) # prevents that filename is the index, instead makes it a column called 'File'
    
    animalre = re.compile(r'(glut1_(.*))(.nii)')
    measurere = re.compile(fr'({measurement}(.*))(_glut)')
    i=0
    for line in frame['File']:
        line = os.path.basename(line)
        x = animalre.search(line)
        y = measurere.search(line)
        frame.loc[i,'animal'] = x.group(1)
        frame.loc[i,'measurement'] = y.group(1)
        i+=1

    del frame['Sub-brick']
    del frame['File']

    csvfile=os.path.join(path,csvname)+'.csv'
    frame.to_csv(csvfile, index=False)
    print('created', csvfile)
    return frame, csvfile



def MR_boxplots(plotframe):
    """
    dataframe input must have quite specific format. only works for GLUT1_Analysis notebook.
    currently only works for one ROI (could be easily selected as second parameter input)
    Must have group columns with 1 or 2 (see function create_clean_csv above).
    """
    individualFrames={}
    roiname = plotframe.columns[0]
    for measurement in plotframe['measurement'].unique():
        individualFrames[measurement] = plotframe.loc[plotframe['measurement'] == measurement]
        individualFrames[measurement]['group1'] = individualFrames[measurement].loc[plotframe['group'] == 1, roiname]
        individualFrames[measurement]['group2'] = individualFrames[measurement].loc[plotframe['group'] == 2, roiname]

        cat1 = individualFrames[measurement]['group1'].dropna()
        cat2 = individualFrames[measurement]['group2'].dropna()
        tstat, pval = ttest_ind(cat1, cat2)
        
        plt.figure()
        boxplot = individualFrames[measurement].boxplot(column=['group1', 'group2'])
        plt.title(measurement + ' - ' + roiname)
        txt = f"t = {round(tstat,3)}, P = {round(pval,3)}"
        plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)