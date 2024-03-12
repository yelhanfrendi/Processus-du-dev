import streamlit as st
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import io
import os
import cv2
import numpy as np


dossier_images_originales = 'images_originales'
dossier_images_traitees = 'images_traitees'

def convert_to_bw(image):
    """Convertir l'image en noir et blanc."""
    return image.convert('L')


def rotate_img_cv2(image_array):
    """Rotate the image 180 degrees using OpenCV."""
    rotated_image = cv2.rotate(image_array, cv2.ROTATE_180)
    return rotated_image


def apply_sepia(image):
    """Appliquer un filtre sépia à l'image."""
    sepia_filter = Image.new("RGB", image.size)
    for py in range(image.size[1]):
        for px in range(image.size[0]):
            r, g, b = image.getpixel((px, py))
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            tr = min(255, tr)
            tg = min(255, tg)
            tb = min(255, tb)
            sepia_filter.putpixel((px, py), (tr, tg, tb))
    return sepia_filter

def rajouter_text(image):
    # Superposer le texte sur l'image traitée
    draw = ImageDraw.Draw(image)
    # Définir la police et la taille du texte
    font = ImageFont.truetype("arial.ttf", 20)  # Modifier le chemin de la police si nécessaire

    # Obtenir le texte à superposer sur l'image depuis la zone de saisie de texte
    texte = st.text_input("Entrez le texte à superposer sur l'image")

    # Option pour la position du texte
    position_option = st.selectbox("Choisissez la position du texte", ["Haut", "Milieu", "Bas"])

    # Calculer la position du texte en fonction de l'option sélectionnée
    if position_option == "Haut":
        position = (10, 10)  # Haut : Position fixe à partir du coin supérieur gauche
    elif position_option == "Milieu":
        width, height = image.size
        position = (10, height // 2)  # Milieu : Verticalement au milieu, à 10 pixels de la gauche
    elif position_option == "Bas":
        width, height = image.size
        position = (10, height - 30)  # Bas : Position fixe à partir du coin inférieur gauche, en supposant que 30 pixels suffisent pour la hauteur du texte

    # Dessiner le texte sur l'image
    draw.text(position, texte, fill="red", font=font)
    return image

def reframe_img_cv2(img, scale_percent):
    """
    Redimensionne l'image selon un pourcentage donné.
    
    Args:
    - img : image source en format array Numpy
    - scale_percent : pourcentage du redimensionnement (0.1 pour 10%, 1 pour 100%, etc.)
    
    Returns:
    - Image redimensionnée comme array Numpy
    """
    # Calculer les nouvelles dimensions
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)

    # Redimensionner l'image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  
    return resized



def sauvegarder_image(dossier, image, nom_fichier):
    """Sauvegarder une image dans un dossier spécifique."""
    if not os.path.exists(dossier):
        os.makedirs(dossier)
    chemin_complet = os.path.join(dossier, nom_fichier)
    image.save(chemin_complet)

def apply_blur(image):
    """Appliquer un effet de flou à  l'image."""
    return image.filter(ImageFilter.BLUR)

def crop_center(img, new_width, new_height):
    """
    Découpe la partie centrale de l'image selon les dimensions spécifiées.

    Args:
    - img: Image source sous forme de tableau NumPy.
    - new_width: Nouvelle largeur de l'image.
    - new_height: Nouvelle hauteur de l'image.

    Returns:
    - Partie centrale découpée de l'image originale.
    """
    y, x = img.shape[0], img.shape[1]
    startx = x // 2 - (new_width // 2)
    starty = y // 2 - (new_height // 2)
    
    # Vérification pour s'assurer que startx et starty ne sont pas négatifs
    startx = max(startx, 0)
    starty = max(starty, 0)

    # Si l'image est en couleur
    if len(img.shape) == 3:
        return img[starty:starty + new_height, startx:startx + new_width, :]
    # Si l'image est en niveaux de gris
    else:
        return img[starty:starty + new_height, startx:startx + new_width]



st.title('Application de téléchargement et de traitement d\'image')



uploaded_file = st.file_uploader("Choisissez une image à  télécharger", type=['jpg', 'png', 'jpeg'])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image originale', use_column_width=True)
    
    traitement_options = ['Rajouter un texte','Rotation', 'Redimensionner', 'Noir et Blanc', 'Sépia', 'Flou','Découpe']
    choix_traitements = st.multiselect('Choisissez les traitements à appliquer:', traitement_options)

    if choix_traitements:
        image_modifiee = image  # Image initiale à modifier
        
        for traitement in choix_traitements:
            if traitement == 'Rotation':
                image_modifiee = rotate_img_cv2(np.array(image_modifiee))
                image_modifiee = Image.fromarray(image_modifiee)  # Conversion de NumPy array à PIL Image
                st.image(image_modifiee, caption='Après rotation', use_column_width=True)
                
            elif traitement == 'Redimensionner':
                facteur = st.slider('Choisissez un facteur de redimensionnement:', 0.1, 1.0, 0.5)
                image_modifiee = reframe_img_cv2(np.array(image_modifiee), facteur)
                image_modifiee = Image.fromarray(image_modifiee)  # Conversion de NumPy array à PIL Image
                st.image(image_modifiee, caption='Après redimensionnement', use_column_width=True)

            elif traitement == 'Découpe':
                pourcentage = st.slider('Choisissez le pourcentage de l\'image à conserver:', 10, 100, 50)
                
                # Convertir PIL Image en tableau NumPy si nécessaire
                if isinstance(image_modifiee, Image.Image):
                    image_modifiee = np.array(image_modifiee)
                
                # Vérifiez si l'image est en niveaux de gris (2D) ou en couleur (3D)
                if len(image_modifiee.shape) == 3:  # Image en couleur
                    height, width, _ = image_modifiee.shape
                else:  # Image en niveaux de gris
                    height, width = image_modifiee.shape

                new_width = int(width * pourcentage / 100)
                new_height = int(height * pourcentage / 100)

                image_modifiee = crop_center(image_modifiee, new_width, new_height)
                # Après le découpage, convertissez l'image_modifiee en PIL Image pour l'affichage
                image_modifiee = Image.fromarray(image_modifiee)
                st.image(image_modifiee, caption='Après conversion en Noir et Blanc', use_column_width=True)
                
            elif traitement == 'Noir et Blanc':
                image_modifiee = convert_to_bw(image_modifiee)
                st.image(image_modifiee, caption='Après conversion en Noir et Blanc', use_column_width=True)
                
            elif traitement == 'Sépia':
                if isinstance(image_modifiee, np.ndarray):
                    image_modifiee = Image.fromarray(image_modifiee)
                image_modifiee = apply_sepia(image_modifiee)
                st.image(image_modifiee, caption='Après application du filtre Sépia', use_column_width=True)  
            elif traitement == 'Flou':
                image_modifiee = apply_blur(image_modifiee)
                st.image(image_modifiee, caption='Après application du flou', use_column_width=True)
            elif traitement == 'Rajouter un texte':
                image_modifiee = rajouter_text(image_modifiee)
                st.image(image_modifiee, caption='Après application de rajouter un texte', use_column_width=True)                
        # Option pour télécharger l'image finale
        img_byte_arr = io.BytesIO()
        image_modifiee.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        st.download_button(label='Télécharger l\'image finale',
                           data=img_byte_arr,
                           file_name='image_finale.png',
                           mime='image/png')
