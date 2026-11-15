import os
import numpy as np
from skimage import io, color
from PIL import Image
import matplotlib.pyplot as plt


plt.close('all')
imagen = io.imread(r"abecedario_fuente.png")
imagen = imagen[0:380,:]                                           



if imagen.shape[2] == 4:
    imagen = imagen[:, :, :3]               

                       

img_gris = color.rgb2gray(imagen)                                  
fil, col = img_gris.shape                                          
img_bin = np.uint8(np.copy(img_gris))                              




for i in range(fil):
    for j in range(col):
        if img_gris[i,j] >= 0.8: 
            img_bin[i,j] = 0                                       
        else:
            img_bin[i,j] = 1          


                             

def centrar_filas(img_org):
    proyeccion = np.sum(img_org, axis = 1)                         
    fil, col = img_org.shape                                       
    letra_indices = []                                             
    dentro_letra = False                                           

  
    for i, valor in enumerate(proyeccion):
        if valor > 0 and not dentro_letra:
            inicio_ind = i                                         
            dentro_letra = True                

        elif valor == 0 and dentro_letra:    
            if (fil - i) > 6:                                      
                if  proyeccion[i + 6] != 0:                        
                    pass
                elif proyeccion[i + 6] == 0:                       
                    final_ind = i
                    letra_indices.append((inicio_ind,final_ind))
                    dentro_letra = False                           
            
            elif (fil - i) < 5:                                    
                final_ind = i
                letra_indices.append((inicio_ind,final_ind))
                dentro_letra = False
    return inicio_ind,final_ind




def segmentar_filas(original_img):
    proyeccion = np.sum(original_img, axis = 1)
    fil, col = original_img.shape
    letra_indices = []
    dentro_letra = False

    for i, valor in enumerate(proyeccion):
        if valor > 0 and not dentro_letra:
            inicio_ind = i            
            dentro_letra = True

        elif valor == 0 and dentro_letra:
            if (fil - i) > 6: 
                if  proyeccion[i + 6] != 0:
                    pass
                elif proyeccion[i + 6] == 0:
                    final_ind = i
                    letra_indices.append((inicio_ind,final_ind))
                    dentro_letra = False
            elif (fil - i) < 5:
                final_ind = i
                letra_indices.append((inicio_ind,final_ind))
                dentro_letra = False
    
    renglones = []                                                 
    for inicio_ind, final_ind in letra_indices:
        renglon = original_img[inicio_ind - 3 : final_ind + 3]    
        renglones.append(renglon)
    return renglones





def segmentar_columnas(renglon, original_img, letras, prefijo):
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
  
    letra_ind = 0                                                  

    
    for inicio_ind, final_ind in letra_indices: 
       
        if letra_ind < len(letras): 
            
            yini, yfin = centrar_filas(renglon[:, inicio_ind:final_ind])     
                
            letra_img = original_img[yini:yfin, inicio_ind:final_ind]*255  
                   
            letra_img_pre = Image.fromarray(letra_img) 
            
            letra_img_pre.save(os.path.join(output_dir, f"{letras[letra_ind]}{prefijo}.jpg")) 
            print(f"Letra {letras[letra_ind]} guardada como {letras[letra_ind]}{prefijo}.jpg") 
           
            letra_ind += 1
    


letras_mayus = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")
letras_minus = list("abcdefghijklmnñopqrstuvwxyz")
letras_acen = list("aeiou")



output_dir = "abecedario1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  

renglones = segmentar_filas(img_bin)
renglon1 = np.array(renglones[0])
renglon2 = np.array(renglones[1])
renglon3 = np.array(renglones[2])



plt.figure(figsize=(10,5))
plt.subplot(3,1,1)
plt.imshow(renglon1, cmap='gray')
plt.title('Primer Renglón Segmentado')
plt.axis('off')


plt.subplot(3,1,2)
plt.imshow(renglon2, cmap='gray')
plt.title('Segundo Renglón Segmentado')
plt.axis('off')


plt.subplot(3,1,3)
plt.imshow(renglon3, cmap='gray')
plt.title('Tercer Renglón Segmentado')
plt.axis('off')



segmentar_columnas(renglon1, renglon1, letras_mayus, "")    # Mayúsculas
segmentar_columnas(renglon2, renglon2, letras_minus, "m")   # Minúsculas
segmentar_columnas(renglon3, renglon3, letras_acen, "c")    # Vocales acentuadas

plt.show()