import os
import numpy as np
from skimage import io, color, transform
import matplotlib.pyplot as plt
import pyttsx3

plt.close('all')
imagen = io.imread(r"poema.png")
imagen = imagen[650:,:]



output_dir = "abecedario1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



# Si es RGBA pasar a RGB
if imagen.shape[2] == 4:
    imagen = imagen[:, :, :3]
else:
    img_gris = np.copy(imagen)
    
img_gris = color.rgb2gray(imagen)
fil, col = img_gris.shape
img_bin = np.copy(img_gris)



# Binarización de imagen 
for i in range(fil):
    for j in range(col):
        if img_gris[i,j] >= 0.8:                  
            img_bin[i,j] = 0
        else:
            img_bin[i,j] = 1                      
            
            
            
def hablar():
    engine = pyttsx3.init()
    engine.setProperty('rate', 125) 

    voices = engine.getProperty('voices')
    for voice in voices:
        if "sabina" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break
    else:  
        for voice in voices:
            if "spanish" in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        
    return engine


            
# Encuentra índices de inicio y fin de una letra (en filas) y centra verticalmente
def fil_cent(original_img):
    proyeccion = np.sum(original_img, axis=1)
    fil, col = original_img.shape
    inicio_ind = 0                                        
    final_ind = fil
    umbral = 0.1 * np.max(proyeccion)                     
    indices_texto = np.where(proyeccion > umbral)[0]      
    
    if len(indices_texto) > 0:                            
        inicio_ind = np.min(indices_texto)
        final_ind = np.max(indices_texto) + 1             
    
    return inicio_ind, final_ind




def fil_seg(original_img):
    proyeccion = np.sum(original_img, axis=1)
    fil, col = original_img.shape
    letra_indices = []
    dentro_letra = False
    
    for i, valor in enumerate(proyeccion):
        if valor > 0 and not dentro_letra:
            inicio_ind = i
            dentro_letra = True
        elif valor == 0 and dentro_letra:
            if (fil - i) > 6:
                if proyeccion[i + 6] != 0:
                    pass
                elif proyeccion[i + 6] == 0:
                    final_ind = i
                    letra_indices.append((inicio_ind, final_ind))
                    dentro_letra = False
            elif (fil - i) < 5:
                final_ind = i
                letra_indices.append((inicio_ind, final_ind))
                dentro_letra = False
    
    renglones = []
    for inicio_ind, final_ind in letra_indices:
        centro = (inicio_ind + final_ind) // 2
        renglon = original_img[centro - 50 : centro + 60, :]
        if renglon.size > 0:                                       
            renglones.append(renglon)
    
    return renglones





filas_segmentadas = fil_seg(img_bin)
renglones = np.stack(filas_segmentadas, axis=0) 


# Segmenta las letras dentro de un renglón de teto
def col_seg(renglon):
    proyeccion = np.sum(renglon, axis = 0)                        
    letra_indices = [] 
    dentro_letra = False 
   
    for i, valor in enumerate(proyeccion): 
        if valor > 0 and not dentro_letra:          
            inicio_ind = i 
            dentro_letra = True 
        elif valor == 0 and dentro_letra:         
            final_ind = i 
            letra_indices.append((inicio_ind,final_ind)) 
            dentro_letra = False 
            
            if proyeccion[i + 54] != 0:
                pass
            elif proyeccion[i + 54] == 0:
                inicio_ind = 0
                final_ind = 28
                letra_indices.append((inicio_ind,final_ind))
            
    
    letras_columnas = []
    for inicio_ind, final_ind in letra_indices:           
        if inicio_ind != 0:
            yini, yfin = fil_cent(renglon[:,inicio_ind:final_ind])
            
        
        letra_img = renglon[yini:yfin, inicio_ind:final_ind]
        letra_img = transform.resize(letra_img, (28, 28), anti_aliasing=True)
        y,x = letra_img.shape

        
        for i in range(y):
            for j in range(x):
                if letra_img[i,j] >= 0.8:
                    letra_img[i,j] = 1
                else:
                    letra_img[i,j] = 0

        letra_img = letra_img.flatten()
        letras_columnas.append(letra_img)
    return letras_columnas
            



#----------------- INSTAR -----------------
# Training
input_dir = "abecedario1"
imagenes_vectores = []

# Recorrer todas las imágenes en la carpeta
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(input_dir, filename)                        # Crea la ruta completa al archivo de img
        img = io.imread(img_path)
        img_resized = transform.resize(img, (28, 28), anti_aliasing=True)   # Redimensiona y suaviza
        y,x = img_resized.shape                                             # Obtiene dimensiones

        # Binarizar el arreglo
        for i in range(y):
            for j in range(x):
                if img_resized[i,j] >= 0.8:
                    img_resized[i,j] = 1
                else:
                    img_resized[i,j] = 0
        
        img_vector = img_resized.flatten()                                  
        imagenes_vectores.append(img_vector)                                




alpha = 0.1                                                                 
epsilon = 0.017                                                            
patrones = np.vstack(imagenes_vectores)                                     
pesos1 = np.vstack(imagenes_vectores)                                       
patrones = patrones.astype(np.float64)
pesos1 = pesos1.astype(np.float64)
S = pesos1.shape[0]                                                         
pesos2 = np.full((S, S), -epsilon)                                          
np.fill_diagonal(pesos2, 1)                                                
bias = np.zeros((S, 1))                                                     

for epocas in range(100):
    for i in range(patrones.shape[0]):                                      
        a1 = pesos1.dot(patrones[i,:].T) + bias.flatten()                   
        a2 = np.where(pesos2.dot(a1) <= 0, 0, pesos2.dot(a1))               
        ganador = np.argmax(a1)                                             

        for j in range(pesos1.shape[0]):                                    
            if j == ganador:
                pesos1[j,:] += alpha*(patrones[i,:] - pesos1[j,:])          
        bias[ganador] = bias[ganador] - 0.2*(1 + bias[ganador])             





# Testing
renglones = np.array(fil_seg(img_bin))                                      
patrones_test = []                                                          
letras = list("AáaBbCcDdEéeFfGgHhIíiJjKkLlMmNnOóoPpQqRrSsTtUúuVvWwXxYyZzÑñ")
texto_completo = []                                                         
palabra_actual = []                                                         
for k in range(0, 68):
    patron = np.array(col_seg(np.array(renglones[k])))                      
    patrones_test.append(patron)
    patrones_test = np.vstack(patrones_test)                                
    patrones_test = patrones_test.astype(np.float64)
    
    for i in range(patrones_test.shape[0]):                                 
        a1 = pesos1.dot(patrones_test[i, :].T) + bias.flatten()             
        ganador = np.argmax(a1)
        letra_predicha = letras[ganador]

        if np.all(patrones_test[i, :] == 0):                                
            if palabra_actual:
                palabra_completa = ''.join(palabra_actual)
                texto_completo.append(palabra_completa)
                palabra_actual = []
        else:                                                               
            palabra_actual.append(letra_predicha)

    if palabra_actual:                                                      
        palabra_completa = ''.join(palabra_actual)
        texto_completo.append(palabra_completa)
        palabra_actual = []
        
    frase=' '.join(texto_completo)
    print(frase)
    print("\n")
    
    engine=hablar()
    
    engine.say(frase)
    engine.runAndWait()

    # Limpiar para el siguiente renglón
    patron = []
    patrones_test = []
    texto_completo = []