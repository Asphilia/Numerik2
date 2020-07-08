import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fftpack as sfft


    
def quant1(V8,p):
# Berechnet zu gegebenem 8x8 Block und gegebenem Verlustparameter p den 
# entsprechenden Block mit niedrigem Speicherbedarf und transformiert 
# ihn wieder zurueck
    return Alow


    
lena = Image.open('lena.jpg')
plt.figure(1)
plt.imshow(lena)
x = np.array(lena)
# Rufen Sie von hier aus die Funktion quant1 auf (entweder fuer einen einzelnen
# Block oder bei groesseren Bildern fuer das gesamte Bild, zerlegt in Blöcke)
reconstructed = Image.fromarray(x,'L')
plt.figure(2)
plt.imshow(reconstructed)
plt.show()