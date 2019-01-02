import os, fnmatch
import subprocess
import numpy as np
import pandas as pd

def splitall(path):
    """splits path into constituents"""
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


def runAFNI(inputstring, printout=True):
    if printout:
        print(inputstring)
    subprocess.run(inputstring,shell=True, check=True)
    
    
def CreateStimParadigm(paradigm):
    StimOnsets=list(range(paradigm['baseline'],paradigm['baseline']+paradigm['ISI']*paradigm['Nblocks'],paradigm['ISI']))
    
    # for plotting vertical lines, get all stim pulses
    pulses_per_block = np.arange(0,paradigm['stim_duration'],1/paradigm['frequency'])
    stim_times_all=np.empty(0)
    for block in range(paradigm['Nblocks']):
        to_append = (paradigm['baseline']+block*paradigm['ISI']) + pulses_per_block
        stim_times_all = np.append(stim_times_all, to_append)    
    stim_times_all = stim_times_all - paradigm['initial_cutoff']*paradigm['TR']
    
    # format for 3dDeconvolve
    stimString=' '.join(map(str,StimOnsets))
    stimulus_times = f"'1D: {stimString}'"
    print('stimulus times:',StimOnsets)
    return stimulus_times, StimOnsets, stim_times_all, pulses_per_block


def volreg(folders):
    """
    Never used???
    Expects that a file single_timepoint has been created.
    """
    infile = folder['main']
    outfile = f"{infile}_volreg.nii"
    runAFNI(f'3dvolreg -prefix {infile}.volreg -base single_timepoint+orig -Fourier -zpad 1 -tshift 0 -1Dfile dfile.{infile}.1D {infile}')
    

def initial_cutoff(scan, timepoints):
    scanpath = os.path.dirname(scan)
    outfile = os.path.join(scanpath, "data_cut.nii")
    runAFNI(f"3dTcat -prefix {outfile} {scan}'[{timepoints}..$]'", printout=False)
    return outfile


def filter_blur(scan,highpass,blurFWHM):
    cleaneddata = os.path.join(os.path.dirname(scan),'CleanedData.nii')
    runAFNI(f"3dTproject -input {scan} -passband {1/highpass} 99999 -blur {blurFWHM} -prefix {cleaneddata}", printout=False)
    return cleaneddata



def Deconvolve(scan,paradigm):
    scanpath = os.path.dirname(scan)
    os.chdir(scanpath) # just easier than working with full paths below
    runAFNI('3dDeconvolve -input ' + scan + \
                        ' -polort 0' \
                        ' -nodmbase' \
                        ' -overwrite' \
                        ' -num_stimts 1' \
                        ' -stim_times 1 ' +  str(paradigm['stim_times']) + \
                        " 'SPMG3(" + str(paradigm['stim_duration']) + ")'" \
                        ' -stim_times_subtract ' + str(paradigm['initial_cutoff']*paradigm['TR']) + \
                        ' -stim_label 1 stim' \
                        ' -iresp 1 HRF_SPM_el' \
                        ' -fout -tout -x1D XSPM.xmat.1D -xjpeg XSPM.jpg' \
                        ' -fitts fittsSPM_' \
                        ' -errts errtsSPM' \
                        ' -bucket statsSPM_' \
                        ' -cbucket regcoeffsSPM')


def find(pattern, path):
    """
    from: https://stackoverflow.com/questions/1724693/find-a-file-in-python
    Usage example:  find('*.txt', '/path/to/dir')
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
    
      
    
def find_cleaned_data(folders):
    if 'cleaned' not in folders:
        result = find('CleanedData.*', folders['scan'])
        return result[0]
    else:
        return folders['cleaned']
    
    
    
def averageblocks(inputdf, paradigm, averaging):
    """
    Is also used in CalciumGroupAnalysis (just needs some modification)
    """
    starttimes = np.array(paradigm['stim_onsets'])-paradigm['initial_cutoff']
    baseline = averaging['baseline']
    duration = averaging['time'] + paradigm['stim_duration']
    
    idx = pd.IndexSlice
    blocks_mean=[]
    block_time = np.arange(-baseline,duration) 
    blockIndex = pd.Index(block_time, name='Time [s]')
    
    for i in range(inputdf.columns.size):
        block_single = []
        
        for n in starttimes:
            #shifting the baseline of each block to zero
            _temp = inputdf.iloc[n-baseline:n+duration,i].reset_index(drop=True)
            _temp -= _temp.iloc[:baseline].mean()
            block_single.append(_temp)
        
        blocks = pd.concat(block_single, axis=1, ignore_index = True)
        blocks['time'] = blockIndex
        blocks.set_index('time',drop=True,inplace=True)           
            
        blocks_mean.append(blocks.mean(axis=1))
        
    blocks_output = pd.concat(blocks_mean, axis=1, ignore_index = True)
    blocks_output['time'] = blockIndex
    blocks_output.set_index('time',drop=True,inplace=True)
    return blocks_output