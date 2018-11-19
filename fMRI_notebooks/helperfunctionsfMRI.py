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


def runAFNI(inputstring):
    print(inputstring)
    subprocess.run(inputstring,shell=True, check=True)
    
    
def CreateStimParadigm(paradigm):
    stimList=list(range(paradigm['baseline'],paradigm['baseline']+paradigm['ISI']*paradigm['Nblocks'],paradigm['ISI']))
    stimString=' '.join(map(str,stimList))
#     stimulus_times = "'1D: "+stimString+"'"
    stimulus_times = f"'1D: {stimString}'"
    print('stimulus times:',stimList)
    return stimulus_times


def volreg(folders):
    """
    Expects that a file single_timepoint has been created.
    """
    infile = folder['main']
    outfile = f"{infile}_volreg.nii"
    runAFNI(f'3dvolreg -prefix {infile}.volreg -base single_timepoint+orig -Fourier -zpad 1 -tshift 0 -1Dfile dfile.{infile}.1D {infile}')