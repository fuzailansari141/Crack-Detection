# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:57:46 2023

@author: Fuzail Ansari
"""
#%%
import os
import cv2
import torch
import tkinter
import warnings
import torchvision
import numpy as np
import customtkinter
import tkinter as tk
from PIL import Image
import tkinter.messagebox
from tkinter import filedialog
warnings.filterwarnings("ignore")
from torchvision import transforms as T
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue") 
#%%
class App(customtkinter.CTk):
    width=1100
    height=750
    def __init__(self):
        super().__init__()

        # configure window
        self.title("Crack vision")
        self.geometry(f"{self.width}x{self.height}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        
        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=9, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="MENU",font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Process Image:",font=customtkinter.CTkFont(weight="bold"))
        self.logo_label.grid(row=1, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame,text="      Crack Detection     ",command=self.process_crack_detection)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame,text="      Crack Severity       ", command=self.process_classification)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame,text="Segmentation & Overlay", command=self.process_segmentation_overlay)
        self.sidebar_button_4.grid(row=4, column=0, padx=20, pady=10)
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame,text="Bounding &  Innerwidth", command=self.bounding)
        self.sidebar_button_5.grid(row=5, column=0, padx=20, pady=10)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Process Directory:",font=customtkinter.CTkFont(weight="bold"))
        self.logo_label.grid(row=6, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_7 = customtkinter.CTkButton(self.sidebar_frame,text="Process Images", command=self.process_images)
        self.sidebar_button_7.grid(row=7, column=0, padx=20, pady=10)
    
        self.toplevel_window=None
        
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:",font=customtkinter.CTkFont(weight="bold"),anchor="w")
        self.scaling_label.grid(row=12, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=13, column=0, padx=20, pady=(10, 20))     
        
        # load and create background image
        self.bg_image = customtkinter.CTkImage(Image.open("5605708.jpg"),
                                                size=(2000,1050))
        self.bg_image_label = customtkinter.CTkLabel(self,text="",image= self.bg_image)
        self.bg_image_label.grid(row=0, column=1,rowspan=4)
        
        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=30,fg_color="#E0E0E0",bg_color="#F7F7F7",width=200,height=200)
        self.home_frame.grid(row=0, column=1,rowspan=4, padx=(18, 0), pady=(18, 0),sticky="n")
        # load and create background image
        self.bg_image1 = customtkinter.CTkImage(Image.open("vector icon_3791336.png"),
                                                size=(80,60))
        # load and create background image
        self.bg_image2 = customtkinter.CTkImage(Image.open("Lovepik_com-400278553-uploading-a-linear-icon.png"),
                                                size=(25,25))
        # Add the button to home_frame
        self.label1 = customtkinter.CTkLabel(self.home_frame, corner_radius=50,width=20,text="" ,font=customtkinter.CTkFont(size=20, weight="bold"),image= self.bg_image1)
        self.label1.grid(row=1, column=0, padx=20, pady=10)
        
        # Add the button to home_frame
        self.button_1 = customtkinter.CTkButton(self.home_frame, corner_radius=50,width=150,text="Upload Image" ,font=customtkinter.CTkFont(weight="bold"),image= self.bg_image2,command=self.load_image)
        self.button_1.grid(row=2, column=0, padx=20, pady=10)

        # load and create background image
        self.bg_image3 = customtkinter.CTkImage(Image.open("pngfind.com-folder-icon-png-60207.png"),
                                                size=(80,60))        
        # Add the button to home_frame
        self.label2 = customtkinter.CTkLabel(self.home_frame, corner_radius=50,width=20,text="" ,font=customtkinter.CTkFont(size=20, weight="bold"),image= self.bg_image3)
        self.label2.grid(row=1, column=1, padx=30, pady=10)

        # Add the button to home_frame
        self.button_2 = customtkinter.CTkButton(self.home_frame, corner_radius=50,width=150,text="Upload Folder" ,font=customtkinter.CTkFont(weight="bold"),image= self.bg_image2,command=self.load_folder)
        self.button_2.grid(row=2, column=1, padx=30, pady=10)
        
        # create home frame
        self.frame = customtkinter.CTkFrame(self, corner_radius=10,width=500, height=300)
        self.frame.grid(row=2, column=1, padx=(18, 0), pady=(18, 0))
        
        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, corner_radius=20,bg_color="#F7F7F7",width=600,height=200,activate_scrollbars=True)
        self.textbox.grid(row=1, column=1, padx=(18, 0), pady=(130, 0))
        
        self.textbox.configure(state="disabled")
#%% 
    # Function for Scaling
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
#%%
    #Function for printing text in the textbox
    def redirect_print(self,text):
        self.textbox.configure(state="normal")
        self.textbox.insert(tk.END, text + "\n")
        self.textbox.see(tk.END)
        self.textbox.configure(state="disabled")     
#%%
    # Function to load the image
    def load_image(self):
        global file_path,img
        self.redirect_print("Loading Image....\n")
        app.update()
        self.frame.destroy()
        # create Image frame
        self.frame = customtkinter.CTkFrame(self, corner_radius=10,width=500, height=300)
        self.frame.grid(row=2, column=1, padx=(18, 0), pady=(18, 0))
        file_path = tk.filedialog.askopenfilename(title="Select Image", filetypes=(("Image files","*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if file_path:
            img=cv2.imread(file_path)
            self.my_image = customtkinter.CTkImage(light_image=Image.open(file_path),
                                      dark_image=Image.open(file_path),
                                      size=(400, 250))
            self.image_label = customtkinter.CTkLabel(self.frame, text="", image=self.my_image)
            self.image_label.grid(row=0, column=1, padx=20, pady=10)            
            self.redirect_print("Image Loaded\n")
            app.update()
#%%            
    def load_folder(self):
        self.redirect_print("Loading Folder....\n")
        app.update()
        global image_file_names,folder_path
        self.frame.destroy()
        # create Multi Image frame
        self.frame = customtkinter.CTkFrame(self, corner_radius=10,width=500, height=300)
        self.frame.grid(row=2, column=1, padx=(18, 0), pady=(18, 0))
        
        self.img_frame1 = customtkinter.CTkFrame(self.frame, corner_radius=10,width=200, height=150,fg_color="#114B5F",bg_color="#114B5F")
        self.img_frame1.grid(row=1, column=0, padx=(10, 10), pady=(10, 10))
        
        self.img_frame2 = customtkinter.CTkFrame(self.frame, corner_radius=10,width=200, height=150,fg_color="#114B5F",bg_color="#114B5F")
        self.img_frame2.grid(row=1, column=1, padx=(10, 10), pady=(10, 10))
        
        self.img_frame3 = customtkinter.CTkFrame(self.frame, corner_radius=10,width=200, height=150,fg_color="#114B5F",bg_color="#114B5F")
        self.img_frame3.grid(row=1, column=2, padx=(10, 10), pady=(10, 10))
        
        self.img_frame4 = customtkinter.CTkFrame(self.frame, corner_radius=10,width=200, height=150,fg_color="#114B5F",bg_color="#114B5F")
        self.img_frame4.grid(row=2, column=0,padx=(10, 10), pady=(10, 10))
        
        self.img_frame5 = customtkinter.CTkFrame(self.frame, corner_radius=10,width=200, height=150,fg_color="#114B5F",bg_color="#114B5F")
        self.img_frame5.grid(row=2, column=1, padx=(10, 10), pady=(10, 10))
        
        self.img_frame6 = customtkinter.CTkFrame(self.frame, corner_radius=10,width=200, height=150,fg_color="#114B5F",bg_color="#114B5F")
        self.img_frame6.grid(row=2, column=2, padx=(10, 10), pady=(10, 10))
        
        self.image_list=[]
        
        folder_path = filedialog.askdirectory(title="Select Folder") 
        
        # Get a list of all the image file names in the folder
        image_file_names = os.listdir(folder_path)
        image_file_names = [filename for filename in image_file_names if filename.lower().endswith((".JPG",".PNG",".png",".jpg", ".JPEG", ".jpeg",".tif", ".TIF"))]
        self.redirect_print(f"Total no. of Images: {len(image_file_names)}\n")
        app.update()
        
        # Loop for Loading Multi Image Frame
        for i in range(min(6, len(image_file_names))):
            image_path = os.path.join(folder_path, image_file_names[i])
            image = Image.open(image_path)
            self.image_list.append(image)
            
            # self.label_sample.configure(text="Sample Images:")
            
            if len(self.image_list) >= 1:
                image = self.image_list[0]
                self.my_image = customtkinter.CTkImage(light_image=image,
                                                      dark_image=image,
                                                      size=(225,150))
                self.image_label1 = customtkinter.CTkLabel(self.img_frame1, text="", image=self.my_image)
                self.image_label1.grid(row=0, column=0)
                app.update()
            
            if len(self.image_list) >= 2:
                image = self.image_list[1]
                self.my_image = customtkinter.CTkImage(light_image=image,
                                                      dark_image=image,
                                                      size=(225,150))
                self.image_label1 = customtkinter.CTkLabel(self.img_frame2, text="", image=self.my_image)
                self.image_label1.grid(row=0, column=0)
                app.update()
            else:
                pass

            if len(self.image_list) >= 3:            
                image = self.image_list[2]
                self.my_image = customtkinter.CTkImage(light_image=image,
                                                      dark_image=image,
                                                      size=(225,150))
                self.image_label1 = customtkinter.CTkLabel(self.img_frame3, text="", image=self.my_image)
                self.image_label1.grid(row=0, column=0)
                app.update()
            else:
                pass
            
            if len(self.image_list) >= 4:
                image = self.image_list[3]
                self.my_image = customtkinter.CTkImage(light_image=image,
                                                      dark_image=image,
                                                      size=(225,150))
                self.image_label1 = customtkinter.CTkLabel(self.img_frame4, text="", image=self.my_image)
                self.image_label1.grid(row=0, column=0)
                app.update()
            else:
                pass
            
            if len(self.image_list) >= 5:
                image = self.image_list[4]
                self.my_image = customtkinter.CTkImage(light_image=image,
                                                      dark_image=image,
                                                      size=(225,150))
                self.image_label1 = customtkinter.CTkLabel(self.img_frame5, text="", image=self.my_image)
                self.image_label1.grid(row=0, column=0)
                app.update()
            else:
                pass
            
            if len(self.image_list) >= 6:
                image = self.image_list[5]
                self.my_image = customtkinter.CTkImage(light_image=image,
                                                      dark_image=image,
                                                      size=(225,150))
                self.image_label1 = customtkinter.CTkLabel(self.img_frame6, text="", image=self.my_image)
                self.image_label1.grid(row=0, column=0)
                app.update()
            else:
                pass
        self.redirect_print("Folder Loaded\n")
        app.update()
#%%            
    def process_images(self):
        self.redirect_print("Processing....\n")
        app.update()
        i=1
        # Loop through each image file name and apply the code
        for file in image_file_names:
            self.redirect_print(f"Making Folder for Image no.{i} for saving outputs....\n")
            app.update()
            # Load the image
            img_path = os.path.join(folder_path, file)
            file_name=img_path.split("/")[-1].split(".")[0]
            if os.path.exists(file_name):
                # Delete all the files inside the folder
                for item in os.listdir(file_name):
                    item_path = os.path.join(file_name, item)
                    os.remove(item_path)
            # Delete the existing folder
                os.rmdir(file_name)
            os.mkdir(file_name)
            self.redirect_print(f"Detecting Crack for Image no.{i}....\n")
            app.update()
            #Crack Classification Model(Positive Or Negative)
            model1 = load_model("crack_classification_model.h5")
            
            # Load the image and resize it to the expected size
            img1 = image.load_img(img_path, target_size=(420, 420))
            img_arr = image.img_to_array(img1)
            img_arr = np.expand_dims(img_arr, axis=0)
        
            # Make a prediction
            prediction = model1.predict(img_arr)
        
            # Define the label based on the prediction
            if prediction[0] > 0.5:
                self.redirect_print(f"Crack Detected for Image no.{i}\n")
                app.update()
                # Classification code goes here
                #Crack Classification Model(Minor or Moderate Or Major)
                self.redirect_print(f"Checking Severity of Crack for Image no.{i}....\n")
                app.update()                
                model3 = load_model("crack_severity_classifier_model.h5")
                # Classify new images
                def classify_image(image_path):
                    image1 = cv2.imread(image_path)
                    image = cv2.resize(image1, (150, 150))
                    image = np.expand_dims(image, axis=0) # Add a batch dimension to the image for feeding it into the neural network
                    image = image / 255.0  # Normalize pixel values
                    prediction = model3.predict(image)
                    predicted_label = categories[np.argmax(prediction)]
                    return predicted_label,image1
                
                categories = ["Minor", "Moderate", "Major"]
                predicted_label,img = classify_image(img_path)
                self.redirect_print(f"Severity of Crack for Image no.{i} is {predicted_label}\n")
                app.update()               
                cv2.putText(img, predicted_label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imwrite(f"{file_name}/Severity.jpg",img)
                self.redirect_print(f"Segmenting Crack for Image no.{i}....\n")
                app.update() 
                # Define the model architecture
                model2 = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=False)
                model2.classifier = DeepLabHead(2048, 1)
            
                # Load the saved state dictionary
                saved_model_path = "crack_segmentation_model.pth"
                state_dict = torch.load(saved_model_path, map_location=torch.device("cpu"))
                model2.load_state_dict(state_dict)
                model2.eval()
            
                # Define the transform to preprocess the image
                transform = T.Compose([
                    T.ToPILImage(),
                    T.Resize((256)),
                    T.ToTensor()
                ])
            
                # Load the new image
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
                # Apply the saved model to the new image
                with torch.no_grad():
                    input_tensor = transform(img).unsqueeze(0)
                    output = model2(input_tensor)
                    output = output['out'][0].cpu().detach().numpy().transpose(1,2,0)
            
                # Normalize the output to the range [0, 255]
                output_norm = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
                # Apply adaptive histogram equalization
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_eq = clahe.apply(output_norm)
            
                # Increase contrast
                alpha = 2.0
                beta = 50
            
                img_contrast = cv2.convertScaleAbs(img_eq, alpha=alpha, beta=beta)
            
                # Resize the mask to match the RGB image dimensions
                mask = cv2.resize(img_contrast, (img.shape[1], img.shape[0]))
            
                coord=np.where(mask==255)
            
                # Create a new array with the same shape as mask, filled with zeros
                new_array = np.zeros_like(mask)
                new_array[coord] = 255
                cv2.imwrite(f"{file_name}/Mask.jpg",new_array)
            
                # Convert the mask to 3 color channels to match the RGB image
                mask = cv2.cvtColor(new_array, cv2.COLOR_GRAY2BGR)
            
                # Apply a color map to the mask to make it visible on the RGB image
                color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                
                color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Overlay the color mask on the RGB image
                result = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
                cv2.imwrite(f"{file_name}/Overlay.jpg",result)
                self.redirect_print(f"Crack Segmented and overlayed for Image no.{i}\n")
                app.update() 
                
                self.redirect_print(f"Making bounding box and analysing crack for Image no.{i}....\n")
                app.update() 
                # Convert the mask to grayscale
                gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
                # Find contours in the grayscale image
                contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
                crack_intensity = (np.sum(mask==255) / (np.sum(mask!=255) + np.sum(mask==255)))* 100
            
                wi=[]
                he=[]
                # Draw the contours and rectangles on the original image
                for cnt in contours:
                    cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
                    # Get the bounding rectangle of the contour
                    x, y, w, h = cv2.boundingRect(cnt)
                    wi.append(w)
                    he.append(h)
                    # Draw the rectangle
                    cv2.rectangle(result, (x, y), (x+w, y+h), (0,255,255), 2)
                        
                text=f"Height:{max(he)}, Width:{max(wi)}, Crack_per:{round(crack_intensity, 2)}%"
            
                # Using cv2.putText() method
                cv2.putText(result,text, (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
            
                if len(contours)==1:
                    # Find the coordinates of white pixels in the mask
                    y, x = np.where(new_array == 255)
                    
                    # Find the unique row indices
                    y_set = set(y)
                    
                    # Compute the width of each column
                    width = []
                    for n in y_set:
                        width.append(np.sum(y == n))
                    
                    # Find the maximum width and its corresponding row index
                    max_width = max(width)
                    max_width_index = width.index(max_width)
                    max_width_row = list(y_set)[max_width_index]
                    # print("max width:",max_width)
                    
                    # Find the start and end points of the maximum width line within the white region
                    start_point = (np.min(x[y == max_width_row]), max_width_row)
                    end_point = (np.max(x[y == max_width_row]), max_width_row)
                    
                    # Draw the maximum width line within the white region
                    cv2.line(result, start_point, end_point, (0, 0, 255), 2)  # Draw red line
                    cv2.putText(result, f"Single Crack,Max inner width: {max_width}, No. of Cracks: {len(contours)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imwrite(f"{file_name}/Bounding.jpg",result)
                    self.redirect_print(f"bounding box and analysing completed for Image no.{i}....\n")
                    app.update() 
                    i+=1
                else:
                    cv2.putText(result, f"Multiple Crack, No. of Cracks: {len(contours)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    cv2.imwrite(f"{file_name}/Bounding.jpg",result)
                    self.redirect_print(f"bounding box and analysing completed for Image no.{i}....\n")
                    app.update()
                    i+=1
            else:
                img=cv2.imread(img_path)
                cv2.putText(img, "No Crack Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                cv2.imwrite(f"{file_name}/No_crack.jpg",img)
                self.redirect_print(f"No Crack Detected for Image no.{i}\n")
                app.update()
                i+=1
                
        self.redirect_print("Sucessfully Completed all the Process & and Saved all the results in the folder!\n")
        app.update()

#%%
    # Function to process image for crack detection
    def process_crack_detection(self):
        try:
            self.redirect_print("Detecting Crack....\n")
            app.update()
            #Crack Classification Model(Positive Or Negative)
            model1 = load_model("crack_classification_model.h5")
            
            # Load the image and resize it to the expected size
            img1 = image.load_img(file_path, target_size=(420, 420))
            img_arr = image.img_to_array(img1)
            img_arr = np.expand_dims(img_arr, axis=0)
        
            # Make a prediction
            prediction = model1.predict(img_arr)
        
            # Define the label based on the prediction
            if prediction[0] > 0.5:
                cv2.namedWindow("Crack Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Crack Detection", 800, 600)
                cv2.putText(img, "Crack Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                cv2.imshow("Crack Detection", img)
                self.redirect_print("Crack Detected\n")
                app.update()
                app.update()
                cv2.waitKey(0)
                
            else:
                cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Result", 800, 600)
                cv2.putText(img, "No Crack Detected", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
                cv2.imshow("Result", img)
                self.redirect_print("No Crack Detected\n")
                app.update()
                cv2.waitKey(0)
        except Exception as e:
            self.redirect_print(f"Error: {e}\n")
            
#%%
    # Function to process image for classification
    def process_classification(self):
        
        self.redirect_print("Checking Severity of Crack....\n")
        app.update()
        
        #Crack Classification Model(Minor or Moderate Or Major)
        model3 = load_model("crack_severity_classifier_model.h5")
        # Classify new images
        def classify_image(image_path):
            # Read the image using OpenCV
            image1 = cv2.imread(image_path)
            
            # Resize the image to a fixed size (150x150) for processing
            image = cv2.resize(image1, (150, 150))
            
            # Add a batch dimension to the image for feeding it into the neural network
            image = np.expand_dims(image, axis=0)
            
            # Normalize the pixel values of the image to the range [0, 1]
            image = image / 255.0
            
            # Make a prediction on the normalized image using a pre-trained neural network model (model3)
            prediction = model3.predict(image)
            
            # Get the predicted label by finding the index with the highest probability in the prediction array
            predicted_label = categories[np.argmax(prediction)]
            
            # Return the predicted label and the original image
            return predicted_label, image1
        
        # List of categories for the crack severity levels: "Minor", "Moderate", "Major"
        categories = ["Minor", "Moderate", "Major"]
        
        # Call the classify_image function with the file_path of the input image
        predicted_label, img = classify_image(file_path)
        
        # Redirect the predicted label to some output (e.g., console, log, etc.)
        self.redirect_print(f"Predicted label: {predicted_label}\n")
        
        # Update the app (probably a GUI application) to show the predicted label
        app.update()
        
        # Create a window using OpenCV to display the image with the predicted label
        cv2.namedWindow("Crack Severity Classification", cv2.WINDOW_NORMAL)
        
        # Resize the window to a specific size for display
        cv2.resizeWindow("Crack Severity Classification", 800, 600)
        
        # Add the predicted label as text on the image using OpenCV
        cv2.putText(img, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the image with the predicted label in the created window
        cv2.imshow("Crack Severity Classification", img)
        
        # Wait for a key press (0 means to wait indefinitely) and close the window when a key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
#%%   
    # Function to process image for segmentation and overlay
    def process_segmentation_overlay(self):
        global result,mask,new_array
        
        self.redirect_print("Segmentating Crack....\n")
        app.update()
        # Segmentation and Overlay code goes here
        # Define the model architecture
        model2 = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=False)
        model2.classifier = DeepLabHead(2048, 1)
    
        # Load the saved state dictionary
        saved_model_path = "crack_segmentation_model.pth"
        state_dict = torch.load(saved_model_path, map_location=torch.device("cpu"))
        model2.load_state_dict(state_dict)
        model2.eval()
    
        # Define the transform to preprocess the image
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256)),
            T.ToTensor()
        ])
    
        # Load the new image
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        # Apply the saved model to the new image
        with torch.no_grad():
            input_tensor = transform(img).unsqueeze(0)
            output = model2(input_tensor)
            output = output['out'][0].cpu().detach().numpy().transpose(1,2,0)
    
        # Normalize the output to the range [0, 255]
        output_norm = cv2.normalize(output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_eq = clahe.apply(output_norm)
    
        # Increase contrast
        alpha = 2.0
        beta = 50
    
        img_contrast = cv2.convertScaleAbs(img_eq, alpha=alpha, beta=beta)
    
        # Resize the mask to match the RGB image dimensions
        mask = cv2.resize(img_contrast, (img.shape[1], img.shape[0]))
    
        #Get the coordinates of white pixels
        coord=np.where(mask==255)
    
        # Create a new array with the same shape as mask, filled with zeros
        new_array = np.zeros_like(mask)
        new_array[coord] = 255
    
        # Convert the mask to 3 color channels to match the RGB image
        mask = cv2.cvtColor(new_array, cv2.COLOR_GRAY2BGR)
    
        # Apply a color map to the mask to make it visible on the RGB image
        color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
        color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Overlay the color mask on the RGB image
        result = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
        
        # Display the image
        cv2.namedWindow("Segmentation & Overlay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Segmentation & Overlay", 800, 600)
        cv2.imshow("Segmentation & Overlay", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.redirect_print("Segmentation Compeleted\n")
        app.update()
#%%   
    def bounding(self):
        
        self.redirect_print("Making Bounding Box....\n")
        app.update()
        # Convert the mask to grayscale
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
        # Find contours in the grayscale image
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #Calculate the intensity of the crack
        crack_intensity = (np.sum(mask==255) / (np.sum(mask!=255) + np.sum(mask==255)))* 100
    
        wi=[]
        he=[]
        # Draw the contours and rectangles on the original image
        for cnt in contours:
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
            # Get the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            wi.append(w)
            he.append(h)
            # Draw the rectangle
            cv2.rectangle(result, (x, y), (x+w, y+h), (0,255,255), 2)
                
        text=f"Height:{max(he)}, Width:{max(wi)}, Crack_per:{round(crack_intensity, 2)}%"
    
        # Using cv2.putText() method
        cv2.putText(result,text, (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
        self.redirect_print(f"{text}+\n")
    
        if len(contours)<=3:
            # Find the coordinates of white pixels in the mask
            y, x = np.where(new_array == 255)
            
            # Find the unique row indices
            y_set = set(y)
            
            # Compute the width of each column
            width = []
            for i in y_set:
                width.append(np.sum(y == i))
            
            # Find the maximum width and its corresponding row index
            max_width = max(width)
            max_width_index = width.index(max_width)
            max_width_row = list(y_set)[max_width_index]
            
            # Find the start and end points of the maximum width line within the white region
            start_point = (np.min(x[y == max_width_row]), max_width_row)
            end_point = (np.max(x[y == max_width_row]), max_width_row)
            
            # Draw the maximum width line within the white region
            cv2.line(result, start_point, end_point, (0, 0, 255), 2)  # Draw red line
            cv2.putText(result, f"Single Crack,Max width: {max_width}, No. of Cracks: {len(contours)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            self.redirect_print(f"Single Crack, Maximum inner width:{max_width}, No. of Cracks: {len(contours)}\n")
            
            # Display the image
            cv2.namedWindow("Bounding & Inner Width", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Bounding & Inner Width", 800, 600)
            cv2.imshow("Bounding & Inner Width", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.putText(result, f"Multiple Crack, No. of Cracks: {len(contours)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.namedWindow("Bounding", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Bounding", 800, 600)
            cv2.imshow("Bounding", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

#%%
if __name__ == "__main__":
    app = App()
    app.mainloop()