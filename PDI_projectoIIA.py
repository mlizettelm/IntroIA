import cv2 as cv
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np
from tkinter import *


#Opc1 - Convoluciones
def convoluciones():
    path1 = r"C:\MCC\IIAproject\images\books.jpg"

    image01 = cv.imread(path1) # tomar la imagen a color
    image02 = cv.imread(path1,0) # tomar canales grises 

    # Convertir BGR a RGB
    img1 = cv.cvtColor(image01, cv.COLOR_BGR2RGB)
    img2 = cv.cvtColor(image02,cv.COLOR_BGR2RGB)

    fig, axs = plt.subplots(2,2, figsize = (15,15)) #para crear la matriz de imagenes.

    kernel3 = np.ones((3,3), np.float32)/9
    kernel5 = np.ones((20,20), np.float32)/400

    imk3 = cv.filter2D(img2,-1,kernel3)
    imk5 = cv.filter2D(img1,-1,kernel5)

    axs[0,0].imshow(img1)
    axs[0,0].set_title("Imagen a color")
    axs[0,1].imshow(img2)
    axs[0,1].set_title("Escala de grises")
    axs[1,0].imshow(imk3)
    axs[1,0].set_title("Kernel 3")
    axs[1,1].imshow(imk5)
    axs[1,1].set_title("Kernel 5")

    mng = plt.get_current_fig_manager()    
    mng.window.state('zoomed')  

    plt.show()

#Opc2 - Separación de canales
def canales():
    path2 = r"C:\MCC\IIAproject\images\RGB.jpg"
    cmyk = cv.imread(path2)

    cmykRGB = cv.cvtColor(cmyk,cv.COLOR_BGR2RGB)
    
    B, G, R = cv.split(cmykRGB)  #ojo: revisar como hace el split de colores creo que es RGB

    fig, axs = plt.subplots(1,4, figsize = (15,15))

    axs[0].imshow(R)
    axs[0].set_title("Blue")
    axs[1].imshow(G)
    axs[1].set_title("Green")
    axs[2].imshow(B)
    axs[2].set_title("Red")
    axs[3].imshow(cmykRGB)

    mng = plt.get_current_fig_manager()    
    mng.window.state('zoomed')  
    
    plt.show()

#Opc3 - Binarización
def binarizacion():
    path3 = r"C:\MCC\IIAproject\images\12.jpg"
    letras = cv.imread(path3,0)

    imgGray = cv.cvtColor(letras,cv.COLOR_BGR2RGB)

    # pixel gris 
    imgGray[50,100,1]

    # Threshold (Umbralizar/Binarizar la imagen)
    ret, thres1 = cv.threshold(imgGray,220,255,cv.THRESH_BINARY)
    ret, thres2 = cv.threshold(imgGray,220,255,cv.THRESH_BINARY_INV)

    fig, axs = plt.subplots(1,3,figsize = (10,10))
    axs[0].imshow(imgGray)
    axs[0].set_title("Escala de grises")
    axs[1].imshow(thres1)
    axs[1].set_title("Binaria Est.")
    axs[2].imshow(thres2)
    axs[2].set_title("Binaria Inv. Est.")
    
    mng = plt.get_current_fig_manager()    
    mng.window.state('zoomed')  

    plt.show()

#Opc4 - Operaciones Lógicas
def operaciones():
    
    path4 = r"C:\MCC\IIAproject\images\04.jpg"
    path5 = r"C:\MCC\IIAproject\images\57.jpg"

    img1 = cv.imread(path4)
    img2 = cv.imread(path5)

    ##### Resta
    resta = cv.subtract(img1,img2) #operacion.....substraccion
    opAnd = cv.bitwise_and(img1,img2,mask=None) #operacion..... and
    opOr = cv.bitwise_or(img1,img2, mask = None) #operacion.....or
    op1Negada = cv.bitwise_not(img1,mask=None) #operacion.....
    
    fig, axs = plt.subplots(1,6, figsize = (10,10)) 

    axs[0].imshow(img1)
    axs[0].set_title("Imagen original")
    axs[1].imshow(img2)
    axs[1].set_title("Imagen original")
    axs[2].imshow(resta)
    axs[2].set_title("Resta")
    axs[3].imshow(opAnd)
    axs[3].set_title("Y")
    axs[4].imshow(opOr)
    axs[4].set_title("O")
    axs[5].imshow(op1Negada)
    axs[5].set_title("Negación")

    mng = plt.get_current_fig_manager()    
    mng.window.state('zoomed')

    plt.show()

#Opc5 - Filtros
def filtros():
    path8 = r"C:\MCC\IIAproject\images\05.jpg"

    image = cv.imread(path8)

    # Convertir BGR a RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    fig, axs = plt.subplots(4,3, figsize = (30,30))

    axs[0,0].imshow(image)
    axs[0,0].set_title("Imagen Original")

    # Aplicar filtro gaussiano
    Gaussiano = cv.GaussianBlur(image,(5,5),0)
    axs[0,1].imshow(Gaussiano)
    axs[0,1].set_title("Gaussiano 3x3")

    Gaussiano = cv.GaussianBlur(image,(9,9),0)
    axs[0,2].imshow(Gaussiano)
    axs[0,2].set_title("Gaussiano nxn")

    # Aplicar filtro desenfoque medio
    median = cv.medianBlur(image,3)

    axs[1,0].imshow(image)
    axs[1,0].set_title("Imagen Original")

    axs[1,1].imshow(median)
    axs[1,1].set_title("Mediana 3")

    median = cv.medianBlur(image,11)
    axs[1,2].imshow(median)
    axs[1,2].set_title("Mediana n")

    # Filtro bilateral
    bilateral = cv.bilateralFilter(image,9,200,200)

    axs[2,0].imshow(image)
    axs[2,0].set_title("Imagen Original")

    axs[2,1].imshow(bilateral)
    axs[2,1].set_title(" B 9/75/75")

    bilateral = cv.bilateralFilter(image, 9,300,300)
    axs[2,2].imshow(bilateral)
    axs[2,2].set_title("B 9/200/200")


    # Filtro afilado (Sharpening)
    sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

    sharp_img = cv.filter2D(image,-1, sharp_kernel)

    axs[3,0].imshow(image)
    axs[3,0].set_title("Imagen Original")

    axs[3,1].imshow(sharp_img)

    sharp_kernel = np.array([[1, 0, -1], [0, 0 ,0] ,[-1, 0, 1]])
    sharp_img = cv.filter2D(image,-1, sharp_kernel)
    axs[3,2].imshow(sharp_img)

    mng = plt.get_current_fig_manager()    
    mng.window.state('zoomed')

    plt.show()

#Opc6 - operaciones morfologicas
def morfologicas():
    path9 = r"C:\MCC\IIAproject\images\44.jpg"
    image = cv.imread(path9)

    # Convertir BGR a RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Aplicar filtro desenfoque medio
    median = cv.medianBlur(image,3)

    # Histograma
    #plt.hist(median.ravel(),10,[0,256])

    # Generar imagen binaria mediante umbral estático
    _, thresh1 = cv.threshold(image, 128,255,cv.THRESH_BINARY_INV)

    # Erosión de la imagen
    kernel = np.ones((7,7), np.uint8)
    img_erosion = cv.erode(thresh1, kernel, iterations = 1)

    # Dilatación de la imagen
    img_dilatacion = cv.dilate(thresh1, kernel, iterations = 1)

    fig, axs = plt.subplots(1,5, figsize = (30,30))

    axs[0].imshow(image)
    axs[0].set_title("Imagen Original")
    axs[1].imshow(median)
    axs[1].set_title("Filtro de medianas")
    axs[2].imshow(thresh1)
    axs[2].set_title("Binaria")
    axs[3].imshow(img_erosion)
    axs[3].set_title("Erosion")
    axs[4].imshow(img_dilatacion)
    axs[4].set_title("Dilatación")

    mng = plt.get_current_fig_manager()    
    mng.window.state('zoomed')

    plt.show() 

#Opc 7 - Segmentacion k-means
def segmentacion():
    path10 = r"C:\MCC\IIAproject\images\37.jpg"
    num_clusters = 3

    
    image = cv.imread(path10)

    image_copy = np.copy(image)
    

    cv.imshow("Imagen inicial",image)
   
    if (len(image_copy.shape)< 3):
        pixel_values = image_copy.reshape((-1,1))
    elif (len(image_copy.shape)==3):
        pixel_values = image_copy.reshape((-1,3))

    pixel_values = np.float32(pixel_values)

    
    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,10,1.0)

    number_of_attemps = 10

    centroid_initialization_strategy = cv.KMEANS_RANDOM_CENTERS

    _, labels, centers = cv.kmeans(pixel_values,
                                num_clusters,
                                None,
                                stop_criteria,
                                number_of_attemps,
                                centroid_initialization_strategy)

    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    segmented_image = segmented_data.reshape((image_copy.shape))

    cv.imshow("Imagen Segmentada",segmented_image)
    cv.waitKey(0)




#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#ventana principal
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

root = Tk()
root.title("Procesamiento digital de Imágenes")
root.state("zoomed")

frame01= Frame() #define la instancia del frame01
frame01.pack() #empaqueta el frame para que este contenido en la raiz
frame01.config(width="1050",height="650") #da un tamaño al frame
frame01.config(bg="LightSkyBlue1") #pone un color de back ground
frame01.config(bd=15) #da un ancho de borde
frame01.config(relief="ridge") #pone un relieve
frame01.place(x=20,y=50)
frame01.grid_propagate(False) #evita que el frame sea esclavo del tamñao de sus componenetes cuando usamos un grid


label01= Label(frame01,text="Opciones de Procesamiento")
label01.config(font=("Arial",28))
label01.config(bg="LightSkyBlue1") 
label01.config(fg="white")
label01.grid(row=0,column=0,sticky="w",pady=10)

buttonOpc1 = Button(frame01,text="Convoluciones", command=convoluciones)
buttonOpc1.grid(row=1,column=0,pady=8,sticky="e")
buttonOpc1.config(cursor="hand2")
buttonOpc1.config(bg="bisque")
buttonOpc1.config(font=("Arial",12))
buttonOpc1.config(cursor="hand2")

buttonOpc2 = Button (frame01,text="Separación de Canales", command=canales)
buttonOpc2.grid(row=2,column=0,pady=8,sticky="e")
buttonOpc2.config(cursor="hand2")
buttonOpc2.config(bg="bisque")
buttonOpc2.config(font=("Arial",12))
buttonOpc2.config(cursor="hand2")

buttonOpc3 = Button (frame01,text="Binarización", command=binarizacion)
buttonOpc3.grid(row=3,column=0,pady=8,sticky="e")
buttonOpc3.config(cursor="hand2")
buttonOpc3.config(bg="bisque")
buttonOpc3.config(font=("Arial",12))
buttonOpc3.config(cursor="hand2")

buttonOpc4 = Button (frame01,text="Operaciones lógicas", command=operaciones)
buttonOpc4.grid(row=4,column=0,pady=8,sticky="e")
buttonOpc4.config(cursor="hand2")
buttonOpc4.config(bg="bisque")
buttonOpc4.config(font=("Arial",12))
buttonOpc4.config(cursor="hand2")

buttonOpc5 = Button (frame01,text="Filtros", command=filtros)
buttonOpc5.grid(row=5,column=0,pady=8,sticky="e")
buttonOpc5.config(cursor="hand2")
buttonOpc5.config(bg="bisque")
buttonOpc5.config(font=("Arial",12))
buttonOpc5.config(cursor="hand2")

buttonOpc6 = Button (frame01,text="Operaciones Morfologicas", command=morfologicas)
buttonOpc6.grid(row=6,column=0,pady=8,sticky="e")
buttonOpc6.config(cursor="hand2")
buttonOpc6.config(bg="bisque")
buttonOpc6.config(font=("Arial",12))
buttonOpc6.config(cursor="hand2")


buttonOpc7 = Button (frame01,text="Segmentación K-means", command=segmentacion)
buttonOpc7.grid(row=7,column=0,pady=8,sticky="e")
buttonOpc7.config(cursor="hand2")
buttonOpc7.config(bg="bisque")
buttonOpc7.config(font=("Arial",12))
buttonOpc7.config(cursor="hand2")

root.mainloop() #ciclo principal para que aparezca el root o pantalla principal
