#!/home/aic/anaconda3/bin/python

##!/usr/bin/python3

#usage: run script within a folder where there are all the scan number subfolders e.g. 34 35 36 etc..  
# put the converted file in the same folder as the 2dseq unline the nrmal convert_bruker_batch.py

# reads the parametes from method file of each scan to convert EPI it to BRIK using to3d
# WARNING: Only tested with normal 2D EPI.
# currently assuming zero interslice distance. 

import os   #assuming this script is run from the scan folder
import re  #regular expressions, to search the text.
import subprocess #see manual https://docs.python.org/3/library/subprocess.html
#import shutil #only needed when copying/moving files

#os.chdir('D:\\MRDATA\\20171222_FSc_iso\\16')





#the actual conversion script
def conversion(methodfilewithpath):

#####Part 1: read the parameters from method file


    method = open(methodfilewithpath) #using os methods for script to work under windows&linux
    # method = open(os.path.join(cwd,'method')) #using os methods for script to work under windows&linux


    method_fulltext = method.read()


    TR1 = re.compile(r'##\$PVM_RepetitionTime=(\d+)')
    Matrix1 = re.compile('##\$PVM_Matrix=\\( 2 \\)\s(\d+)\s(\d+)')
    Repetitions1 = re.compile(r'##\$PVM_NRepetitions=(\d+)')
    Slices1 = re.compile('##\$PVM_SPackArrNSlices=\\( 1 \\)\s(\d+)')
    Slice_thickness1 = re.compile(r'##\$PVM_SliceThick=(\d+\.\d+)')
    FOV1 = re.compile('##\$PVM_Fov=\\( 2 \\)\s(\d+)\s(\d+)')
    Segments1 = re.compile(r'##\$NSegments=(\d+)')

    TR = TR1.search(method_fulltext)
    TR = TR.group(1)

    Matrix = Matrix1.search(method_fulltext)
    if Matrix is None:
        return
    else:
        Matrix = Matrix.group(1,2)

    Repetitions = Repetitions1.search(method_fulltext)
    Repetitions = Repetitions.group(1)

    Slices = Slices1.search(method_fulltext)
    if Slices is None:
        return
    else:
        Slices = Slices.group(1)

    Slice_thickness = Slice_thickness1.search(method_fulltext)
    Slice_thickness = Slice_thickness.group(1)

    FOV = FOV1.search(method_fulltext)
    FOV = FOV.group(1,2)

    Segments = Segments1.search(method_fulltext)
    Segments = Segments.group(1)

####part 2: calculate parameters and create the to3d command

    originalpath = os.getcwd()
    scanpath = os.path.dirname(methodfilewithpath)
    scannumber = os.path.basename(scanpath)
    os.chdir(os.path.join(scanpath,'pdata','1'))

    #example
    #to3d -epan -view orig -prefix $argv[1] -time:zt 9 960 1500 alt+z -xFOV 8L-R -yFOV 6A-P -zSLAB 2.25I-S 3D:0:0:128:96:8640:"$argv[1].2dseq"

    TR = str(int(TR)*int(Segments)) #for multi-shot EPI
    xFOV = str(float(FOV[0])/2)
    yFOV = str(float(FOV[1])/2)
    zSLAB = str(float(Slices)*float(Slice_thickness)/2)
    read = Matrix[0]
    phase = Matrix[1]
    images = str(int(Repetitions)*int(Slices))

    to3d = 'to3d -epan -view orig -prefix E' + scannumber + ' -time:zt ' \
            + Slices + ' '      \
            + Repetitions + ' ' \
            + TR + ' alt+z -xFOV ' \
            + xFOV + 'L-R -yFOV '  \
            + yFOV + 'A-P -zSLAB ' \
            + zSLAB + 'I-S 3D:0:0:'\
            + read + ':' + phase + ':' \
            + images + ':' + '"*2dseq"'
            
    print(to3d)
    #to3d= 'to3d -epan -view orig -prefix ' + filename + ' -time:zt ' + Slices 960 1500 alt+z -xFOV 8L-R -yFOV 6A-P -zSLAB 2.25I-S 3D:0:0:128:96:8640:"$argv[1].2dseq"


#####part3: run AFNI to3d

    subprocess.run(to3d,shell=True, check=True) #test later if shell=true can be removed.

	### copy the files to common folder  (to save disk space eventually use move command)
	# shutil.copy('E' + scannumber + '+orig.BRIK.Z',copytofolder)
	# shutil.copy('E' + scannumber + '+orig.HEAD',copytofolder)
	
    #shutil.move('E' + scannumber + '+orig.BRIK.Z',copytofolder)
	#shutil.move('E' + scannumber + '+orig.HEAD',copytofolder)


    if int(Repetitions) == 1:
        print('WARNING! Scan had only one repetition!')

    os.chdir(originalpath) #get back to the origin folder, or the main loop may get messed up.
    return



### part 0: walk the directory tree, and run stuff below only for folders containing a method file.

cwd = os.getcwd()
#os.mkdir('all_converted')
#copytofolder = os.path.join(cwd,'all_converted')

#in following quite messy code to determine if file has already been converted, if so skip without error
import glob
i = 0
### the main loop of the batch program
for folderName, subfolders, filenames in os.walk(cwd):
    for file in filenames:
        if file == 'method':
            x = glob.glob(os.path.join(folderName,'pdata/1/E*.BRIK*'))
            print(x)
            try:
                y = os.path.isfile(os.path.join(folderName,x[0]))
                if y:
                    print('scan '+ folderName + ' already converted.')
            except:
                conversion(os.path.join(folderName,file))
                i += 1
                print('converting')
print('In total '+ str(i) + ' scans converted.')

