import os, glob, fnmatch
import scipy.io as sio
import pandas as pd
import numpy as np
from lmfit.models import ExpressionModel
import ipywidgets as widgets
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt
import dill as pickle

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('retina')

def get_filepaths(mainpath, fileending='*.mat', subfolders=True):
    """
    Finds the files in ALL subfolders.
    """
    filepaths = []
    
    if subfolders:
        for folderName, _subfolders, filenames in os.walk(mainpath):
            for file in filenames:
                if fnmatch.fnmatch(file, fileending):
                    filepaths.append(os.path.join(os.path.abspath(folderName),file))
    
    else:   
        filepaths = glob.glob(os.path.join(os.path.abspath(mainpath),fileending))


    print('files found: \n', filepaths)
  
    return filepaths

    
    
def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range") 
    
    
def append_all_data(filepaths):
    alldata=[]
    N_files=len(filepaths)
    for i in range(N_files):
        alldata.append(sio.loadmat(filepaths[i]))
    return alldata, filepaths, N_files

def extract_mat_data(alldata, filepaths, N_files):
    '''
    probably very ugly way of doing things. Switch to pickle format in future.
    '''
        
    stim_events = alldata[0][os.path.split(filepaths[0])[1][:-4]+' Channel 1']['stim'][0][0][0]

    #Create data frames
    time_sec=alldata[0][os.path.split(filepaths[0])[1][:-4]+' Channel 1']['time'][0][0][0] # must be same for all files, so read it from first mat file
    index = pd.Index(time_sec)
    columns = pd.MultiIndex.from_product([['channel1','channel2'],['mean','SD','stim']],names=['channel','datatype'])
    # df0 = pd.DataFrame(np.zeros((index.size,columns.size)),index = index, columns = columns)

    dfs=[]
    measurements=[]
    for i in range(N_files):
        measurementname = get_nth_key(alldata[i],4)[:-10]
        df = pd.DataFrame(np.zeros((index.size,columns.size)),index = index, columns = columns)

        df['channel1','mean'] = alldata[i][get_nth_key(alldata[i],3)]['mean'][0][0][0][0:index.size]
        df['channel1','SD'] = alldata[i][get_nth_key(alldata[i],3)]['SD'][0][0][0][0:index.size]

        df['channel2','mean'] = alldata[i][get_nth_key(alldata[i],4)]['mean'][0][0][0][0:index.size]
        df['channel2','SD'] = alldata[i][get_nth_key(alldata[i],4)]['SD'][0][0][0][0:index.size]
        measurements.append(measurementname)
        dfs.append(df)

    dffinal=pd.concat(dfs, axis=1, names=['channel','datatype','scan'], ignore_index=False)
    return stim_events, dffinal, measurements


def normalize_by_bsl(inputdf, bsl_duration=10, targetSD=1):
    '''
    Optional:  Probably not the best way... calculates SD during baseline of each scan and puts them on an equal value  (targetSD).
    Idea behind that is that percent changes are then comparable across different SNR values.
    '''
    xx = inputdf.loc[0:bsl_duration,:].describe()
#     print(xx)
#     x = xx.loc['mean',:]/xx.loc['std',:]
    x = xx.loc['std',:]
    print(x)
#     x.hist()
#     np.histogram(x)
    plt.figure()
    plt.boxplot(x)
    plt.show()
    
    k=0
    for i in x:
        factor = targetSD/i
        inputdf.iloc[:,k]=inputdf.iloc[:,k]*factor
        k+=1

#     inputdf=inputdf.mean(axis=1)
    return inputdf




def CreateModel(stim_events):
    '''
    creates lmfit model object with a gamma variate at every stimulus timepoint.
    parameter values seem to work well both for neurons and astrocytes
    (in case of astrocyte/isoflurane the first gamma variate will just cover the whole time course and all others get almost zero amplitude)
    '''
    
    
    script = """
def gammavar(ymax, tmax, a, x, t0):
    x_input = x * (x > t0) + (t0 + tmax) * (x <= t0)
    return (exp(log(ymax)+a*(1+log((x_input-t0)/tmax)-(x_input-t0)/tmax))) * (x > t0)
"""

    model = ExpressionModel(f'gammavar(ymax0, tmax0, a0, x, 0.02)',
                            init_script=script,
                            intependent_vars=['x'])


    k=1
    for i in stim_events[1:]:
        model += ExpressionModel(f'gammavar(ymax{k}, tmax{k}, a{k}, x, {i})',
                            init_script=script,
                            intependent_vars=['x'])
        k += 1
           
    params = model.make_params()
    params['ymax0'].set(value=1, min=0.002, max=10)
    params['tmax0'].set(value=2, min=0.1, max=8)
    params['a0'].set(1, min=0.05, max=5)

    for i in range(1,len(stim_events)):
        params[f'tmax{i}'].set(expr='tmax0')   
        params[f'a{i}'].set(expr='a0')     
        params[f'ymax{i}'].set(0.8, min=0.002, max=10)  
    
    return model, params


def CreateModel_simple(stim_events):
    
    script = """
def gammavar(ymax, tmax, a, x, t0):
    x_input = x * (x > t0) + (t0 + tmax) * (x <= t0)
    return (exp(log(ymax)+a*(1+log((x_input-t0)/tmax)-(x_input-t0)/tmax))) * (x > t0)
"""
    
    model = ExpressionModel(f'gammavar(ymax0, tmax0, a0, x, 0.02)',
                            init_script=script,
                            intependent_vars=['x'])
    
    params = model.make_params()
    params['ymax0'].set(value=1, min=0.002, max=10)
    params['tmax0'].set(value=2, min=0.1, max=20)
    params['a0'].set(1, min=0.05, max=5)
    
    return model, params
    

def RunFit(stim_events, time, signal, singlegamma):
    if singlegamma:
        model, params = CreateModel_simple(stim_events)
    else:
        model, params = CreateModel(stim_events)
    result = model.fit(signal, params, x=time)
    return result



def RunFit_multi(inputdf, stim_events, singlegamma=False):
    '''
    The main function that takes data frame as input and runs gamma variate fits to each column
    '''
    time = inputdf.index.values
    result = {}
    smallresult = {}
    for column in inputdf:
        print(f'fitting {column}')
        signal = inputdf[column]
        fullresult = RunFit(stim_events, time, signal, singlegamma)
        result[column] = fullresult
        reduced_result = [fullresult.best_values, fullresult.data, fullresult.best_fit]
        smallresult[column] = reduced_result
    print('-----')
    return result, smallresult


def get_descriptive_values(result):
    allvalues = []
    for name, fit in result.items():
        time = fit.data.index.values
        scan = {}
        scan['ID'] = name
        scan['ttp_single'] = fit.best_values['tmax0']
        scan['alpha'] = fit.best_values['a0']
        scan['ttp_overall'] = time[np.argmax(fit.best_fit)]
        scan['mean_amplitude'] = np.sum(fit.best_fit)/np.count_nonzero(fit.best_fit)
        scan['max_amplitude'] = np.max(fit.best_fit)

        allvalues.append(scan)
    #     amplitude
    #     for y in range(1,len(fit.components)):
    #         amplitude = fit.best_values[f'ymax{y}']
    #         if amplitude > 0.003:
    #             amplitudes.append(amplitude)
    #         else:
    #             amplitudes.append(0)

    parameters = pd.DataFrame(allvalues)
    print(scan.keys())
    return parameters


def plot_fit(resultobject,channel, scanID):
    time = resultobject[channel][scanID].data.index.values
    signal = resultobject[channel][scanID].data.values
    plt.figure(figsize=[40,30])
    plt.plot(time, signal)
    plt.plot(time, resultobject[channel][scanID].best_fit)
    plt.rcParams.update({'font.size': 30})

def interactive_plot(resultobject):
    interact(plot_fit, resultobject=fixed(resultobject), channel=list(resultobject.keys()), scanID=list(resultobject['ch1'].keys()))
    
    
    
def savedata(datatostore, folder, filename):
    '''
    just add any kind of variables in a list like this:
    # dataToStore = [results_small, parameters]
    # g.savedata(dataToStore, file_path, 'calciumdata.p')
    
    protip: import dill as pickle <-- works the same way, but supports more file types.
    '''
    pickle.dump(datatostore, open(os.path.join(folder, filename),'wb'))

    
def add_mean_SD(df, whichaxis=1):
    """
    simply adds mean and standard deviation across all columns to any dataframe
    safety checks if mean/SD columns already exist.
    """
    if {'mean', 'SD'}.issubset(df):
        print("mean/SD already added")
        return df
    else:
        df['mean'] = df.mean(axis=whichaxis)
        df['SD'] = df.std(axis=whichaxis)
        return df


def excel_multi_sheets(in_library, path, filename='Results.xlsx'):
    """
    example usage:
    forexcel={} # below, add the sheet-name and corresponding dataframe in order of your choice.
    forexcel['RCaMP_traces'] = normalized['ch1']
    forexcel['RCaMP_stats'] = parameters['ch1']
    """
    outfile = os.path.join(path,filename)
    writer = pd.ExcelWriter(outfile)
    for key, value in in_library.items():
        value.to_excel(writer, sheet_name=key)
    writer.save()
    print(f"stored data under {outfile}")
    
    
    
def check_if_arrays_equal(a, b):
    if a.shape != b.shape:
        print("The blocks don't have the same number of pulses as in the calcium data")
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            print("The timings don't match with that of the calcium")
            return False
    return True    
    
    
    
# ==============================================================    
# mainly BOLD related stuff below:
# ==============================================================    


def load_afni_rois(file_path, IDs='', file_ending='*ROIdata.1D'):
    """
    Should work for any number of ROI files, and any number of ROIs (but all files should have same number of rois).
    IDs (optional) is list of animal/scan names, will be combined with ROI name. Must have same lenght as ROIfiles. Can also be used to override the automatic name given from ROIfile.
    In in the future ID should be in the ROIdata.1D filename!
    """
    ROIfiles = get_filepaths(file_path, file_ending)
    dataframes = []
    keys = []
    i=0
    for files in ROIfiles:
        dataframes.append(pd.read_table(files))
        name=os.path.basename(ROIfiles[i][:-len(file_ending)+1]) #cut off *ROIdata.1D = 10 chars 
        del dataframes[-1]['File']
        del dataframes[-1]['Sub-brick']
        for ROI in range(dataframes[-1].columns.size):
            try:
                keys.append(IDs[i] + ' ' + dataframes[-1].columns[ROI])
            except:
                keys.append(name + dataframes[-1].columns[ROI])
        i+=1

    frames = pd.concat(dataframes, axis=1)
    frames.columns = keys  
    return frames


def averageblocks(inputdf,starttimes,baseline,duration):
    """
    Goal: This function should work for any dataframe (calcium, or BOLD signals)
    probably only works for 1s TR BOLD at the moment!
    """
    columnnames = inputdf.columns.values
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
    blocks_output.columns = columnnames
    return blocks_output
                                     
# BOLDaverages = averageblocks(frames,stimList,10,50)
# BOLDaverages.columns =keys



def averageblocks_BOLD(inputdf, normalized, stim_events):
    """
    calls generic averageblocks() above, reading some additional parameters from the calcium, 
    to make both averages comparable (same timings etc.)
    """
    baseline = int(np.absolute(np.round(normalized['ch1'].index[0])))
    duration = int(np.round(normalized['ch1'].index[-1]))
    
    starttimes = stim_events
    BOLDaverages = averageblocks(inputdf,starttimes,baseline,duration)
    
    return BOLDaverages
