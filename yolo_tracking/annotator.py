import argparse
import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json, copy
import random

from matplotlib.widgets import Button

class Matcher:
    def __init__(self, annot_filename, desired_width = 200, desired_height = 200):
        # params
        self.annot_filename = self.validate_filename(annot_filename)
        self.desired_width = desired_width
        self.desired_height = desired_height
        
        # other attributes
        self.actual_IDs = {
                            'blur': [], 
                            'unrecognizable': []
                          }
        self.count_ID = 0
        self.processed_image_paths = []     
        self.if_new_ID = False

        #button info
        self.button_data = [
            {"label": "Same Face", "position": [0.15, 0.05, 0.1, 0.05],"path":None},
            {"label": "Different Face", "position": [0.35, 0.05, 0.11, 0.05],"path":None},
            {"label": "Too blur", "position": [0.55, 0.05, 0.1, 0.05],"path":None},
            {"label": "Unrecognisable", "position": [0.75, 0.05, 0.13, 0.05],"path":None},
            {"label": "Acceptable Face", "position": [0.15, 0.05, 0.1, 0.05],"path":None},            
        ]

    def validate_filename(self, filename):
        i = 1
        ori_filename = copy.deepcopy(filename)
        while os.path.isfile(filename):
            i += 1
            temp = os.path.splitext(ori_filename)[0]
            filename = f'{temp}_{i}.json'
        return filename
        
        
    def loop(self, root_dir, sample_num):
        '''
        root_dir
        ├── (ID 1)
        │   └── img 1.1
        │   └── img 1.2   
        │        
        ├── (ID 2)
        │   └── img 2.1
        │   └── img 2.2  
        │        
        ├── (ID 3)
        │   └── img 3.1
        │   └── img 3.2 
        
        
        then we flatten to:
        alist = [img 1.1, img 1.2, img 2.1, img 2.2, .....]
        
        then nested for loop for mathcing
        '''
        
        # list all directoy in root_dir
        sub_dirs = os.listdir(root_dir)
        sub_dirs = [os.path.join(root_dir, sub_dir) for sub_dir in sub_dirs]
        
        # flatten to get all image paths
        all_image_paths = []
        all_pseudo_ids  = []
        current_id      = 0
        for sub_dir in sub_dirs:
            # get image paths
            paths = os.listdir(sub_dir) # get all paths in sub_dir
            paths = [os.path.join(sub_dir, path) for path in paths] # join the sub_dir with paths
            temp = copy.deepcopy(paths[:3]) # extract first 3 images, since these images are crucial for identification
            paths = copy.deepcopy(paths[3:]) # remaining images will be randomly sampled
            random.shuffle(paths) # shuffle the remaining images
            paths = temp + paths # combine together
            paths = paths[:sample_num] # extract N number of samples only
            paths = copy.deepcopy(paths)
            all_image_paths += paths

            # get pseudo_ids
            pseudo_ids = [current_id] * len(paths)
            all_pseudo_ids += pseudo_ids
            
        
        #--------------------------------------------------------------------
        # start looping
        skip = False # skip if not new id
        previous_pseudo_id = None
        for i in range(len(all_image_paths)):
            # save annotations every iteration            
            self.save_annotations()

            # get current id
            current_pseudo_id = all_pseudo_ids[i]
            if (previous_pseudo_id is not None) and skip:
                if current_pseudo_id == previous_pseudo_id:
                    continue
                else:
                    skip = False
            
            # get 1st image path
            image_path1 = all_image_paths[i]

            # if had been processed
            if image_path1 in self.processed_image_paths:
                continue
            
            # prompt this image
            self.display_one_image(image_path1)            

            # if not new ID
            if not self.if_new_ID:
                skip = True
                previous_pseudo_id = current_pseudo_id
                continue
                
            # create a new ID, and store this image
            self.count_ID += 1
            self.actual_IDs[self.count_ID] = [image_path1]            

            # compare with 2nd image
            for j in range(i+1, len(all_image_paths)):
                # get 2nd image path
                image_path2 = all_image_paths[j]
                self.display_two_images(image_path1, image_path2)

                self.save_annotations()
            

    def save_annotations(self):
        with open(self.annot_filename, "w") as fp:
            json.dump(self.actual_IDs , fp, indent=4)

    def display_one_image(self, image_path):
        # set background 
        plt.style.use("dark_background")
        # Load the first image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        # Resize both images
        image = cv2.resize(image, (self.desired_width, self.desired_height))
        dummy_image = np.ones((image.shape))

        # Create a Matplotlib figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10,5))

        # Display the first image in the first subplot
        axes[0].imshow(image)
        axes[0].set_title(os.path.basename(os.path.basename(image_path)))
        axes[0].axis('off')  # Turn off axis labels

        # Display the second image in the second subplot
        axes[1].imshow(dummy_image)
        axes[1].set_title('None')
        axes[1].axis('off')  # Turn off axis labels        

        #--------------------------------------------------------------------
        #button operation start here
        
        self.if_new_ID = False
        
        #Acceptable face----------------------------------------------------------------------------------
        aFaceBtn = plt.axes(self.button_data[4]["position"])
        btn1 = Button(aFaceBtn,label= self.button_data[4]["label"],color='b',hovercolor=u'0.1')
        def aFace(event):
            print("this face acceptable")
            plt.close()
            self.processed_image_paths.append(image_path) # dont have to check this path again
            self.if_new_ID = True
        
        btn1.on_clicked(aFace)

        #Blur face----------------------------------------------------------------------------------
        bFaceBtn = plt.axes(self.button_data[2]["position"])
        btn3 = Button(bFaceBtn,label= self.button_data[2]["label"],color='b',hovercolor=u'0.1')
        def bFace(event):
            print("too blur cannot la")
            plt.close()
            self.actual_IDs['blur'].append(image_path)
            self.processed_image_paths.append(image_path) # dont have to check this path again
            self.if_new_ID = False

        btn3.on_clicked(bFace)

        #unrecognise face----------------------------------------------------------------------------
        uFaceBtn = plt.axes(self.button_data[3]["position"])
        btn4 = Button(uFaceBtn,label= self.button_data[3]["label"],color='b',hovercolor=u'0.1')
        def unRecog(event):
            print("cannot cannot this one")
            plt.close()
            self.actual_IDs['unrecognizable'].append(image_path)
            self.processed_image_paths.append(image_path) # dont have to check this path again
            self.if_new_ID = False
          
        btn4.on_clicked(unRecog)      
            
        #End of button
        #--------------------------------------------------------------------------------------------------

        # Get the current figure manager
        mgr = plt.get_current_fig_manager()

        # Set the window's geometry to center it
        mgr.window.geometry(f"+0+0") #(f"(wxh)+x+y), where w = width, h = height of the windows in pixel on screen, x = x position, y = y position on screen, 0,

        # Display the Matplotlib figure
        plt.show()     

        
    def display_two_images(self, image_path1, image_path2):
        # set background 
        plt.style.use("dark_background")
        # Load the first image
        image1 = cv2.imread(image_path1)
        image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)

        # Load the second image
        image2 = cv2.imread(image_path2)
        image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)

        # Resize both images
        resized_image1 = cv2.resize(image1, (self.desired_width, self.desired_height))
        resized_image2 = cv2.resize(image2, (self.desired_width, self.desired_height))

        # Create a Matplotlib figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10,5))

        # Display the first image in the first subplot
        axes[0].imshow(resized_image1)
        axes[0].set_title(os.path.basename(os.path.basename(image_path1)))
        axes[0].axis('off')  # Turn off axis labels

        # Display the second image in the second subplot
        axes[1].imshow(resized_image2)
        axes[1].set_title(os.path.basename(os.path.basename(image_path2)))
        axes[1].axis('off')  # Turn off axis labels      

        #--------------------------------------------------------------------
        #button operation start here
        
        #Same face------------------------------------------------------------------------------------
        sFaceBtn = plt.axes(self.button_data[0]["position"])
        btn1 = Button(sFaceBtn,label= self.button_data[0]["label"],color='b',hovercolor=u'0.1')
        def sFace(event):
            print("Same Face")  
            plt.close()
            self.actual_IDs[self.count_ID].append(image_path2)
            self.processed_image_paths.append(image_path2) # dont have to check this path again           

        btn1.on_clicked(sFace)
        
        #different face------------------------------------------------------------------------------
        dFaceBtn = plt.axes(self.button_data[1]["position"])
        btn2 = Button(dFaceBtn,label= self.button_data[1]["label"],color='b',hovercolor=u'0.1')
        def dFace(event):
            print("Macam Not neh")
            plt.close()            

        btn2.on_clicked(dFace)

        #Blur face----------------------------------------------------------------------------------
        bFaceBtn = plt.axes(self.button_data[2]["position"])
        btn3 = Button(bFaceBtn,label= self.button_data[2]["label"],color='b',hovercolor=u'0.1')
        def bFace(event):
            print("too blur cannot la")
            plt.close()
            self.actual_IDs['blur'].append(image_path2)
            self.processed_image_paths.append(image_path2) # dont have to check this path again           

        btn3.on_clicked(bFace)
        
        #unrecognise face----------------------------------------------------------------------------
        uFaceBtn = plt.axes(self.button_data[3]["position"])
        btn4 = Button(uFaceBtn,label= self.button_data[3]["label"],color='b',hovercolor=u'0.1')
        def unRecog(event):
            print("cannot cannot this one")
            plt.close()
            self.actual_IDs['unrecognizable'].append(image_path2)
            self.processed_image_paths.append(image_path2) # dont have to check this path again
         
        btn4.on_clicked(unRecog)

            
        #End of button
        #--------------------------------------------------------------------------------------------------

        # Get the current figure manager
        mgr = plt.get_current_fig_manager()

        # Set the window's geometry to center it
        mgr.window.geometry(f"+0+0") #(f"(wxh)+x+y), where w = width, h = height of the windows in pixel on screen, x = x position, y = y position on screen, 0,

        # Display the Matplotlib figure
        plt.show()

# get root direectory from user
parser = argparse.ArgumentParser()
parser.add_argument('--root-dir', type=str, default='runs/track/exp/crops',
                    help='the root directory that stores all pseudo ID folders')
parser.add_argument('--sample-num', type=int, default=10,
                    help='number of images sampled from each pseudo-ID folder')
opt = parser.parse_args()

# our root directory
root_dir = opt.root_dir
sample_num = opt.sample_num

# validate
assert os.path.isdir(root_dir), f'Directory "{root_dir}" does not available!'
assert sample_num >= 5, f'argument --sample-num must be at least 5'

sub_dirs = []
for path in os.listdir(root_dir):
    sub_dir = os.path.join(root_dir, path)
    sub_dirs.append(sub_dir)

matcher = Matcher(annot_filename="annotation.json")
for sub_dir in sub_dirs:
    matcher.loop(sub_dir, sample_num)
    matcher.save_annotations()
