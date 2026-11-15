Text Recognition Using Competitive Neural Networks 

This project reads a poem and recites it using a competitive neural network.

First, you must run the segmentacion_abecedario code to obtain the segmented letters. The code will generate a folder containing the segmented letters of the alphabet, as shown in the abecedario1 folder.

Then, run the LEER poema code. This code executes a competitive neural network and predicts the letters in the poem image. The result you can show it in Poema_detectado.txt

Libraries to install: os, numpy, skimage, PIL, matplotlib, pyttsx3