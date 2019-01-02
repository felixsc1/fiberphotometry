import shelve, copy
import os
import ipywidgets as widgets



def MainGUI(mainfolder='/mnt/c/Users/felix/Documents/exampledata_python'):
    
    # first we need to check if shelve file exists.
    s = shelve.open(os.path.join(mainfolder,'folders.db'), writeback=True)
    init_shelve(s)

    my_tabs={}

    # 1. instructions tab (converted markdown to html on some website)
    _w_instructions = widgets.HTML(
        value='''
        <h3 id="preparingthedata">1. Preparing the data</h3>

        <p>Folder structure:</p>

        <ul>
        <li><p><em>Main Folder contains these subfolders and files:</em></p>

            <ul>
            <li><em>Rawdata</em></li>

            <li><em>Templates</em></li>

            <li>Animal<em>Scan</em>IDs.xlsx</li></ul></li>

        <li><p>In the <em>Rawdata</em> folder each animal has a subfolder containing all the raw scans (simply the folders given by ParaVision)</p></li>

        <li><p><em>Templates</em> folder contains templates for coregistration for each MR sequence (just pick one of the animals in the group). See examples.</p></li>

        <li><p><em>Excel</em> file contains all animal ID's and their corresponding scan numbers for each modality (see example file).</p></li>
        </ul>
        ''')
    my_tabs['0. Instructions'] = widgets.Box([_w_instructions])

    # 2. folders tab
    style = {'description_width': 'initial'}
    w_elements = {}
    for key in s:
        w_elements[key] = widgets.Text(value=copy.deepcopy(s[key]['path']), description=s[key]['description'],
                             layout = widgets.Layout(width=s[key]['width']), style=style)
        
        
#     w_elements['main_folder'] = widgets.Text(value=s['main_folder'], description='Main Folder:',
#                              layout = widgets.Layout(width='90%'), style=style)
#     w_elements['notebook_folder'] = widgets.Text(value=s['notebook_folder'], description='Notebook (script) Folder:',
#                              layout = widgets.Layout(width='90%'), style=style)
#     w_elements['excel_file'] = widgets.Text(value=s['excel_file'], description='Excel file name:',
#                              layout = widgets.Layout(width='50%'), style=style)
#     w_elements['epi_roi'] = widgets.Text(value=s['epi_roi'], description='EPI ROI:',
#                              layout = widgets.Layout(width='90%'), style=style)
    my_tabs['1. Folders/Files'] = widgets.VBox(list(w_elements.values()))


    # 3. Convert / Coreg
    
    _w_convert = widgets.HTML(
        value = '''
                <ol>
        <li><p>Press the convert button</p></li>

        <li><p>Go in the <em>Analysis</em> folder, select represantive animals for coregistration &amp; ROI template:</p>

        <ol>
        <li>Inspect animals in AFNI.</li>

        <li>Select and copy one <em>single_timepoint.nii</em> EPI file into <em>Main Folder/Templates</em></li>

        <li>Rename it according to your choice in the <em>Folders/Files</em> tab</li></ol></li>

        <li><p>Now that we have templates, press the coregistration button.</p></li>


        <li><p>Outlier animals:</p>

        <ol>
        <li>By default, ROI template in the <em>Templates</em> folder will be used.</li>

        <li>To use an individual ROI file instead, draw ROIs on a <em>single_timepoint.nii</em> of that animal. Save the ROI file under the same name as default ROI template in the main folder of the animal.</li>

        <li>Note: Keep number of ROIs and ROI names consistent with default ROI.</li></ol></li>
        </ol>
                ''')

    my_tabs['2. Convert/Coreg'] = widgets.Box([_w_convert])

    my_tabs['3. Analyses'] = widgets.Box([_w_convert])

    
    
    # Tab interface
    tab_dict = dict(enumerate(my_tabs.keys()))

    interface_tabs = widgets.Tab(
        children=list(my_tabs.values()), _titles=tab_dict)

    display(interface_tabs)

    return s, w_elements




def update_shelve(s, w):
    '''
    no longer hardcoded, now only init_shelve has to be changed and will determine GUI etc.
    '''
    for key in w:
        s[key]['path'] = copy.deepcopy(w[key].value)
    return


def update_shelve_outdated(s, w):
    '''
    Hardcoded contents, only works with jupyter widgets in VascularFunctionalBattery notebook.
    Update the Shelf values in case user changed them, also set values for first call.
    '''
    'writes directly in file, no return value needed'
    
    for key in s:
        if key == 'main_folder':
            update_single_shelve_element(s, w, key, os.path.dirname(s.dict._datfile))
        else:
            update_single_shelve_element(s, w, key, 'insert path / filename')
    
#     if 'notebook_folder' in s:
#         try:
#             s['notebook_folder'] = _w_notebook_folder.value
#         except: # in case GUI has not been created yet, during first call
#             pass
#     else:
#         s['notebook_folder'] = 'path/to/scripts'

#     if 'main_folder' in s:
#         s['main_folder'] = _w_raw_folder.value
#     else:
#         s['main_folder'] = os.path.dirname(s.dict._datfile)

#     if 'excel_file' in s:
#         s['excel_file'] = _w_excel_file.value
#     else:
#         s['excel_file'] = 'AnimalID.xlsx'
        
#     if 'epi_roi' in s:
#         s['epi_roi'] = _w_epi_roi.value
#     else:
#         s['epi_roi'] = '/path/to/ROIs_EPI.nii'

    return

def init_shelve(s):
    '''
    this defines all the user-set variables, also read by the GUI.
    '''
    if 'main_folder' not in s:
        s['main_folder'] = {'path':os.path.dirname(s.dict._datfile),
                            'description':'Main Folder:',
                            'width':'90%'
                            }
    if 'notebook_folder' not in s:
        s['notebook_folder'] = {'path':os.getcwd(),
                            'description':'Notebook (script) Folder:',
                            'width':'90%'
                            }
    if 'excel_file' not in s:
        s['excel_file'] = {'path':'AnimalIDs.xlsx',
                            'description':'Excel file name:',
                            'width':'50%'
                            }
    if 'epi_roi' not in s:
        s['epi_roi'] = {'path':os.path.join(os.path.join(os.path.dirname(s.dict._datfile),'Templates'),'EPI_ROI.nii'),
                            'description':'EPI ROI:',
                            'width':'90%'
                            }
    return


def update_single_shelve_element(shelf, w, key, default_value):
    '''
    w=widget dictionnary
    '''
    if key in shelf:
        try:
            shelf[key] = w[key]
        except: # in case GUI has not been created yet, during first call
            pass
    else:
        shelf[key] = default_value
    return