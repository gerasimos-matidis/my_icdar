__author__ = "Gerasimos Matidis"
# This script creates and uses a graphic interface in order to pass arguments 
# for creating datasets from big images (use by the file "ds_creation.py")
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askdirectory, askopenfile
import sys

def get_arguments_by_gui():
    # start creating the GUI
    root = Tk()
    root.title("Set the parameters to create dataset")

    # initialize the frame
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.option_add("*Font", "Times 12")


    # function that opens a browsing window where the user can select the input image
    def ask_input_image():
        global input_image_path
        f = askopenfile(initialdir="datasets")
        input_image_path = f.name
        input_path_entry.insert(0, input_image_path)

    # Browse button to select the input image
    input_path_label = Label(mainframe, text="Select input image ", width=40, 
        anchor="e")
    input_path_label.grid(column=0, row=0)
    input_path_entry = Entry(mainframe, width=60, font="Arial 10",  borderwidth=3)
    input_path_entry.grid(column=3, row=0, sticky=W)
    input_path_button = Button(mainframe, text="Browse", font="Arial 10", padx=1, 
        command=ask_input_image)
    input_path_button.grid(column=4, row=0, sticky=W)

    # function that opens a window where the user can select the target image
    def ask_target_image():
        global target_image_path
        f = askopenfile()
        target_image_path = f.name
        target_path_entry.insert(0, target_image_path)

    # Browse button to select the target image
    target_path_label = Label(mainframe, text="Select target image ", width=40, 
        anchor="e")
    target_path_label.grid(column=0, row=1)
    target_path_entry = Entry(mainframe, width=60, font="Arial 10",  borderwidth=3)
    target_path_entry.grid(column=3, row=1, sticky=W)
    target_path_button = Button(mainframe, text="Browse", font="Arial 10", padx=1, 
        command=ask_target_image)
    target_path_button.grid(column=4, row=1, sticky=W)

    def ask_output_directory():
        global output_directory
        output_directory = askdirectory(initialdir="datasets")
        output_directory_entry.insert(0, f"{output_directory}")

    # Browse button to select the output directory for the new dataset
    output_directory_label = Label(mainframe, text='select output directory for the '
        'new dataset ', width=40, anchor="e")
    output_directory_label.grid(column=0, row=2)
    output_directory_entry = Entry(mainframe, width=60, font="Arial 10",  
        borderwidth=3)
    output_directory_entry.grid(column=3, row=2, sticky=W)
    output_directory_button = Button(mainframe, text="Browse", font="Arial 10", 
        padx=1, command=ask_output_directory)
    output_directory_button.grid(column=4, row=2, sticky=W)

    # Dropdown menu to define sampling method to create patches
    sampling_methods_dict = {
        'Independent Patches': 'independent_patches',
        'Overlapped Patches': 'overlapped_patches',
        'Random Patches': 'random_patches'
    }
    Sampling_method_label = Label(mainframe, text="Choose a sampling method ", 
        width=40, anchor="e")
    Sampling_method_label.grid(column=0, row=3)
    Sampling_method_options = ['Independent Patches', 'Overlapped Patches', 
        'Random Patches'] # list of the available fMRIPrep versions
    clicked_sampling_method = StringVar()
    clicked_sampling_method.set(Sampling_method_options[0])
    Sampling_method_menu = OptionMenu(mainframe, clicked_sampling_method, 
        *Sampling_method_options)
    Sampling_method_menu.grid(column=3, row=3, sticky="NSWE")

    # Dropdown menu to define patch size
    patch_size_label = Label(mainframe, text="Choose patch size ", width=40, 
        anchor="e")
    patch_size_label.grid(column=0, row=4)
    patch_size_options = [64, 128, 256, 512] # list of the available fMRIPrep versions
    clicked_patch_size = StringVar()
    clicked_patch_size.set(patch_size_options[2])
    patch_size_menu = OptionMenu(mainframe, clicked_patch_size, *patch_size_options)
    patch_size_menu.grid(column=3, row=4, sticky="NSWE")

    # Text box to define the minimum percentage of the less-dominating class. The 
    # images, where the respective percentage is lower, will be later discarded
    percentage_label = Label(mainframe, text='Define a minimum percentage limit '
        'for the less-appeared class in an image', width=50, anchor="e")
    percentage_label.grid(column=0, row=5)
    percentage_entry = Entry(mainframe, width=60, font="Arial 10",  borderwidth=3)
    percentage_entry.insert(0, 0.10)
    percentage_entry.grid(column=3, row=5, sticky=W)

    # Text box to define number of patches
    patches_num_label = Label(mainframe, text='Define number of patches to create '
        '(only with "Random Patches" method) ', width=50, anchor="e")
    patches_num_label.grid(column=0, row=6)
    patches_num_entry = Entry(mainframe, width=60, font="Arial 10",  borderwidth=3)
    patches_num_entry.grid(column=3, row=6, sticky=W)

    # Text box to define patches step
    patches_step_label = Label(mainframe, text='Define patches step (only with '
        '"Overlapped" method) ', width=50, anchor="e")
    patches_step_label.grid(column=0, row=7)
    patches_step_options = [32, 64, 128, 256] # list of the available fMRIPrep versions
    clicked_patches_step = StringVar()
    patches_step_menu = OptionMenu(mainframe, clicked_patches_step, 
        *patches_step_options)
    patches_step_menu.grid(column=3, row=7, sticky="NSWE")

    def on_submit():
        
        sampling_method = sampling_methods_dict[clicked_sampling_method.get()]
        patch_size = int(clicked_patch_size.get())
        percentage = float(percentage_entry.get())
        global args
        args = {
        'input image path': input_image_path,
        'target image path': target_image_path,
        'output directory': output_directory,
        'sampling method': sampling_method,
        'patch size': patch_size,
        'minimum percentage': percentage
        } 
        
        print('\n\33[4mParameters to create the dataset\33[m\n')
        print('Input image path: ', input_image_path)
        print('Target image path', target_image_path)
        print('Output directory of the new dataset: ', output_directory)
        print('\nSampling Method: ', clicked_sampling_method.get())
        print('Patch Size: ', patch_size)
        print('Minimum percentage limit for the less-appeared class in an image: ', 
            "{:.0%}".format(percentage))
        if sampling_method == 'random_patches':
            patches_number = int(patches_num_entry.get())
            args['patches number'] = patches_number
            print('Number of patches to create: ', patches_number)
        if sampling_method == 'overlapped_patches':
            patches_step = int(clicked_patches_step.get())
            args['patches step'] = patches_step
            print('Patches Step: ', clicked_patches_step.get(), '\n')
        
        # TODO: I must set error raisers for cases that some required arguments were
        # not given

        # Close the window
        root.destroy()

    def cancel():
        root.destroy()

    # create the "Submit" and "Cancel" buttons
    submit_button = Button(mainframe, text="Submit", font="Arial 10 bold", 
        bg="cyan", padx=10, borderwidth=4, command=on_submit)
    submit_button.grid(column=4, row=8, sticky="E")
    cancel_button = Button(mainframe, text="Cancel", font="Arial 10 bold", 
        padx=10, borderwidth=4, command=cancel)
    cancel_button.grid(column=6, row=8, sticky="E")
    root.mainloop()
    
    return args

    