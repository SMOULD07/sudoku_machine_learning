import customtkinter as ctk
import cv2
from tkinter import filedialog, messagebox, Menu
from PIL import Image
import torch

import keras
import torch
import numpy as np
from resolution import *
from reconnaisance2 import *
from extraction import *



class Application(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("images/Sudoku")
        self.geometry("1000x1000")

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.initUI()

        self.timer_id = None
        self.selected_file_path = None



    def initUI(self):
        # Load the background image
        self.bg_image_original = Image.open("images/fon.jpg")

        # Create a CTkImage to display the background image
        self.bg_image_ctk = ctk.CTkImage(light_image=self.bg_image_original, size=(800, 600))
        self.bg_label = ctk.CTkLabel(self, image=self.bg_image_ctk)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Bind the window's <Configure> event to the update_bg_image_size function
        self.bind("<Configure>", self.update_bg_image_size)

        self.update_bg_image_size()

        # Transparent frame for widgets
        self.frame = ctk.CTkFrame(self, fg_color="transparent")
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.label_text = ctk.CTkLabel(self.frame, text="Bienvenue", font=("Times New Roman", 24, "bold"),
                                       fg_color="white", bg_color="transparent", text_color="black")
        self.label_text.pack(pady=10)

        self.label_image = ctk.CTkLabel(self.frame, text="")
        self.label_image.pack(pady=10)

        try:
            cam_icon = Image.open("images/cam.jpg")
            cam_icon = cam_icon.resize((100, 100), Image.LANCZOS)
            self.cam_icon_ctk = ctk.CTkImage(light_image=cam_icon, size=(100, 100))
        except Exception as e:
            self.cam_icon_ctk = None

        self.btn_open_camera = ctk.CTkButton(self.frame, text="", command=self.open_camera, image=self.cam_icon_ctk,
                                             width=100, height=100, fg_color="black", hover_color="#2c3e50",
                                             corner_radius=10)
        self.btn_open_camera.pack(pady=10)

        self.btn_browse = ctk.CTkButton(self.frame, text="Importer", command=self.browse_file, width=200, height=50,
                                        fg_color="black", hover_color="#2c3e50", corner_radius=10)
        self.btn_browse.pack(pady=10)
        self.btn_Resolution = ctk.CTkButton(self.frame, text="Résoudre",command=lambda: self.resultat(self.selected_file_path), width=200,
                                            height=50, fg_color="black", hover_color="#2c3e50", corner_radius=10)
        self.btn_Resolution.pack(pady=10)

        self.create_menu()

    def update_bg_image_size(self, event=None):
        # Resize the background image to match the window size
        width = self.winfo_width()
        height = self.winfo_height()
        resized_bg_image = self.bg_image_original.resize((width, height), Image.LANCZOS)
        self.bg_image_ctk = ctk.CTkImage(light_image=resized_bg_image, size=(width, height))

        # Update the image on the label
        self.bg_label.configure(image=self.bg_image_ctk)
        self.bg_label.image = self.bg_image_ctk

    def create_menu(self):
        self.menu_bar = Menu(self)
        self.config(menu=self.menu_bar)

        file_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Ouvrir", command=self.browse_file)
        file_menu.add_command(label="Quitter", command=self.quit)

        about_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="À propos", menu=about_menu)
        about_menu.add_command(label="Info", command=self.show_info)

        language_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Langue", menu=language_menu)
        language_menu.add_command(label="Français", command=lambda: self.change_language('fr'))
        language_menu.add_command(label="English", command=lambda: self.change_language('en'))

    def open_camera(self):
        self.update_frame()
        self.btn_browse.configure(state="disabled")

    def update_frame(self):
        ret, frame = self.vid.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ctk.CTkImage(light_image=image, size=(400, 300))
            self.label_image.configure(image=image)
            self.label_image.image = image

        self.timer_id = self.after(30, self.update_frame)

    def browse_file(self):
        self.selected_file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.bmp")])
        if self.selected_file_path:
            image = Image.open(self.selected_file_path)
            image = image.resize((400, 300), Image.LANCZOS)
            photo = ctk.CTkImage(light_image=image, size=(400, 300))
            self.label_image.configure(image=photo)
            self.label_image.image = photo
            if self.timer_id:
                self.after_cancel(self.timer_id)
            self.btn_browse.configure(state="normal")


    def show_info(self):
        messagebox.showinfo("Information",
                            "Notre projet vise à développer un modèle de machine learning capable de résoudre automatiquement des grilles de Sudoku, en combinant des techniques avancées d'intelligence artificielle et de reconnaissance de motifs pour offrir des solutions rapides et précises.")

    def change_language(self, lang):
        if lang == 'fr':
            self.label_text.configure(text="Bienvenue")
            self.btn_browse.configure(text="Importer")
            self.btn_Resolution.configure(text="Résoudre")
        else:
            self.label_text.configure(text="Welcome")
            self.btn_browse.configure(text="Import")
            self.btn_Resolution.configure(text="Resolve")
    
    def resultat(self, chemin):
        if not chemin:
            messagebox.showerror("Erreur", "Veuillez importer une image.")
            return

        model_reconnaisance = model.Net()
        model_reconnaisance.load_state_dict(torch.load('/home/etud/Bureau/projet_Sudoku/reconnaisance2/mnist_model.pth'))
        model_reconnaisance.eval()

        model_resolution = keras.models.load_model('/home/etud/Bureau/projet_Sudoku/resolution/sudoku_model.h5')
        grille, x, y = get_sudoku_predictions(chemin, model_reconnaisance)
        grille = np.array(grille).reshape(9, 9)

        # Chargement du modèle de résolution 
        predi = outils.solve_sudoku(grille, model_resolution)
        
        # Chemin de l'image de résultat
        result_image_path = "/home/etud/Bureau/projet_Sudoku/images/sudoku_result.png"
        draw_predictions_on_image(chemin, x, y, predi, result_image_path)
        print(predi)

        # Afficher l'image modifiée dans l'interface Tkinter
        result_image = Image.open(result_image_path)
        result_image = result_image.resize((400, 300), Image.LANCZOS)
        result_photo = ctk.CTkImage(light_image=result_image, size=(400, 300))
        self.label_image.configure(image=result_photo)
        self.label_image.image = result_photo



app = Application()
app.mainloop()
