# Analysis tools for fiberphotometry and fMRI

The main tool to load the LabView data is *CalciumAnalysis.ipynb*

## CalciumAnalysis.ipynb manual

1. Double check and adjust the settings in the *Settings* cell.<br><br>
2. Run the whole script by pressing *Run* above. Or run cell-by-cell by pressing CTRL+ENTER.
<br>Starting the script will open a file dialog (possibly hidden behind other windows). Select the Labview .lvm file to analyze.
(Only tested for "CalciumRecordings2channels.vi", 2 signal channels + stimulation channel.)<br><br>
3. Check for error messages below each cell, and make necessary changes.
   <br><br> Typical data issues:
   - Problem: detected stimulus frequency/block lenght/ ON/OFF time is wrong.
       -  Cause: Beginning or end of recording contains incomplete stim block that messes up the regular paradigm. 
           -  Solution: Set *CutStart* or *CutEnd* accordingly to remove any incomplete stim. blocks.
 -  Problem: Detected modulation frequency is wrong (e.g. both channels show 980Hz)
     -   Cause: There was too much laser crosstalk.
         -  Solution: Manually set a value for *mod_ch1* or *mod_ch2*.
  -  Problem: Not all or too many signal drops / jumps are detected.
      -  Cause: Unusually high/low noise in the signal. 
          -  Solution: adjust the advanced parameters *xSTD*, i.e. how many standard deviations are considered to be a jump. and *timetoconsider* i.e. how many seconds before and after each time point should be compared. *Tip*: The output of the cell *Fixing jumps in data* shows the time in seconds where each jump is detected. Compare by eye whether these time points make sense. Also it may need a lot of trial and error, better go directly to the *Fixing jumps* cell below, and play around with different values and re-run only that cell.
  -  Problem: Detrending is overfitting the time course or not removing enough baseline oscillations.
      -  Cause: For long stimulation blocks or lots of physiological noise the automaticpolynomial degree may be wrong.
          -  Solution: Go to the *Detrending / Baseline drift removal* cell, check the output *polynomial degree x. Locate the line #pnum=n in the code above, uncomment the line and enter a value larger or smaller than in the output (must be some integer value).
<br><br>
4. To to clear all previous values and prevent errors, run whole script again after changing a parameter (Button with two arrows above, confirm when asked to restart Kernel)
<br><br>
5. Outputs (created in the same folder as the .lvm file):
    -  A .mat file with the same filename as the .lvm file selected. See Matlab cell below for description of its contents. 
    -  Plots are not stored. Manually format them using plotly and store as image.
    -  Two text files, *regressor1_married.1D* and *regressor2_married.1D* are created. Can be used as inputs for AFNI 3DDeconvolve. (Currently fixed at 1s resolution, like usual fMRI TR, and starting from (first stimulus event - *bsl*) selected in the Settings)


Limitations: <br>
- Although it should work for any modulation frequency, the filter settings have only been tested for 1230Hz (channel1) and 980Hz (channel2).<br>
- The automatic stimulus detection and block averaging only works for regular block stimulation, i.e. constant stim.freq., block duration, and inter-stimulus interval during the whole scan.
- Currently its not possible to cut out time periods in the middle of the time course (only start or end), but bad blocks can be removed from the average.


>©2018 AIC 
