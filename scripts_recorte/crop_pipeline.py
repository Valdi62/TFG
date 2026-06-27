from corners_dataset import CustomImageDataset
from corners_model import TransferLearning
import torch
import pandas as pd
import os
import cv2
import numpy as np
import sys

# Se inicializan las variables globales
puntos = []
img_muestra = None

def click_event(event,x,y,flags,params):
    """
    Función con el callback que se usará para marcar opcionalmente las esquinas que se estén revisando.

    Parámetros que se usan (el resto se ignoran):
        'event' - tipo de evento que se detecta
        'x'     - la coordenada x en la que se ha hecho click
        'y'     - la coordenada y en la que se ha hecho click 
    """
    # Se trabaja con variables globales
    global puntos,img_muestra
    # Se comprueba si se hace click izquierdo para guardar la esquina marcada
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos)<8:
        puntos.extend((x, y))
        # Se dibuja un punto para feedback visual
        cv2.circle(img_muestra,(x,y),5,(255,200,0),-1)
        cv2.imshow("Imagen", img_muestra)

def ordenar_puntos(puntos):
    """
    Función para ordenar los cuatro puntos de un cuadrado que podría quedar deforme. Se ordenan desde la esquina superior
    izquierda hasta la esquina inferior izquierda rodenado en sentido horario.

    Parámetros:
        'puntos' - lista de coordenadas en formato ["x1","y1","x2","y2","x3","y3","x4","y4"]
    """
    # Se juntan las coordenadas de ambas dimensiones y se calcula el centro de los puntos
    puntos = np.array(puntos,dtype=np.float32).reshape(4,2)
    centro = np.mean(puntos,axis=0)

    # Se ordenan los puntos alrededor de su centro para evitar deformaciones
    puntos_ordenados = np.array(sorted(puntos,key=lambda p:np.arctan2(p[1]-centro[1],p[0]-centro[0])),dtype=np.float32)

    # Se mueve el orden de los puntos manteniendo la ordenación calculada para empezar por la esquinas superior izquierda
    idx = np.argmin(puntos_ordenados[:,0]+puntos_ordenados[:,1])
    puntos_ordenados = np.roll(puntos_ordenados,-idx,axis=0)
    return puntos_ordenados

def recorte_img(img,puntos,ancho=1000,alto=1000):
    """
    Función para recortar la imagen a partir de cuatro puntos, para extraer con un cuadrado o rectángulo.

    Parámetros
        'img'    - imagen cargada con cv2 y dispuesta en formato de numpy np.uint8
        'puntos' - lista de formato ["x1","y1","x2","y2","x3","y3","x4","y4"] con las coordenadas a recortar
        'ancho'  - ancho para nuestra nueva imagen
        'alto'   - alto para nuestra nueva imagen
    """
    
    # Debemos reorganizar los puntos obtenidos en el orden adecuado empezando por la esquina superior
    # izquierda y avanzando en sentido horario esquina inferior izquierda. Al poder haber deformaciones
    # o poder estar los puntos desordenados, debemos calcular a qué esquina se corresponde cada coordenada
    puntos_partida = ordenar_puntos(puntos)

    # Vamos a redimensionar al ancho y alto deseado
    puntos_llegada = np.float32([
        [0,0],
        [ancho,0],
        [ancho,alto],
        [0,alto]
    ])

    # Calculamos la matriz de transformación para que las imágenes recortadas no queden deformes
    matrix = cv2.getPerspectiveTransform(puntos_partida,puntos_llegada)
    resultado = cv2.warpPerspective(img,matrix,(ancho,alto))
    return resultado


def crop_pipeline(img_dir,ruta_modelo="Corners_EfficientNet_B3_model.pth",dir_salida="fotos_recortadas",revision=False):
    """
    Función completa que utiliza el modelo entrenado y recortar las imágenes para quedarnos con el interior de los cuadrados.
    La función construirá el modelo elegido y cargará sus parámetros de un archivo en la variable 'ruta_modelo' que tendremos 
    previamente después de entrenar el modelo. Con este modelo a partir de un directorio con imágenes que pueden tener 
    el formato .jpg, .jpeg o .png se preprocesarán las imágenes y se realizará inferencia con el modelo, para las imágenes que 
    se predigan con esquinas se almacenarán ya recortadas por las coordenadas predichas de esas esquinas y procesadas en 
    un directorio llamado como se indique en la variable 'dir_salida'.
    Se puede seleccionar la opción 'revision' a través de la variable con ese nombre para poder comprobar las coordenadas de
    las esquinas predichas y corregir aquellas que estén demasiado alejadas, se podrán marcar esquinas en la foto y sustituirá
    cada a la esquina más cercana predicha por el modelo, de esta forma se pueden marcar solo las peores esquinas predichas y
    quedarte con las predicciones buenas dentro de cada imagen.
    
    Parámetros:
        'img_dir'          - ruta del directorio que contiene únicamente las imágenes a recortar
        'ruta_modelo'      - ruta completa de directorios y nombre del archivo con los parámetros del modelo
        'dir_salida'       - directorio que contendrá las imagenes recortadas
        'revision'         - True si se quiere realizar la revisión de las etiquetas del modelo

    Controles de la revisión:
        'click izquierdo'  - marcar en la imagen una esquinas del cuadrado hasta 4 veces máximo
        'tecla n'          - pasar a la siguiente foto y rellenar con las predicciones del modelo las esquinas no marcadas
        'tecla retroceso'  - descartar la imagen por estar mal etiquetada y no tener cuadrado
        'tecla escape'     - cerrar el programa por completo
    """
    # Variables globales para nuestro callback
    global img_muestra,puntos
    # Vamos a crear y cargar la configuración del modelo entrenado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransferLearning("EfficientNet_B3",0.4,896,384).to(device)
    model.load_state_dict(torch.load(ruta_modelo,map_location=device))

    # Creamos el dataset con las imágenes del directorio
    names = [n for n in os.listdir(img_dir) if n.endswith((".jpg",".jpeg",".png"))]
    images_names = pd.DataFrame(names,columns=["image"])
    dataset = CustomImageDataset(img_dir,images_names,False)

    # Se etiqueta cada imagen del directorio
    os.makedirs(dir_salida,exist_ok=True)
    with torch.no_grad():
        # Con el modelo en evaluación se recorren las imágenes
        model.eval()
        for image,image_name in dataset:
            pred = model(image.unsqueeze(0).to(device))
            # Se comprueba lo primero que haya figura delimitadora
            if pred["classification"].item()>0:
                # Como para red neuronal tuvimos que redimensionar las fotos y bajar su calidad pero las queremos
                # almacenar con la mayor calidad posible, vamos a usar las originales con tamaño (1920,1080)
                ruta = os.path.join(img_dir,image_name)
                imagen_actual = cv2.imread(ruta)

                # Ponemos las imagenes en horizontal y las redimensionamos al mismo tamaño por simplicidad
                h,w=imagen_actual.shape[:2]
                if w<h:
                    imagen_actual=cv2.rotate(imagen_actual,cv2.ROTATE_90_COUNTERCLOCKWISE)
                imagen_actual=cv2.resize(imagen_actual,(1920,1080))  

                coords_normalized = pred["coordinates"].cpu().numpy().flatten()
                # Multiplicar las coordenadas normalizadas por el vector de escalado para obtener las coordenadas reales
                scaling_vector = np.array([1920,1080]*4, dtype=np.float32)
                coords_pixel = (coords_normalized*scaling_vector).tolist()

                # Se reinician estas variables para cada imágen de entrada
                puntos = []
                descartar = False
                # Si queremos revisar los resultados antes de guardarlo, se marcan las nuevas esquinas o se descarta la imágen 
                if revision:
                    img_muestra=imagen_actual.copy()
                    # Se extraen las coordenadas de las esquinas y se marca un punto en cada una
                    for i in range(0,len(coords_pixel),2):
                        x, y = int(coords_pixel[i]),int(coords_pixel[i+1])
                        cv2.circle(img_muestra,(x,y),radius=5,color=(0,0,255),thickness=-1)
                    cv2.imshow("Imagen", img_muestra)
                    cv2.setMouseCallback("Imagen",click_event)
                    # solo se para el bucle si pulsamos escape para salir o n para pasar foto
                    while True:
                        key = cv2.waitKey(1) & 0xFF  # Aseguramos la compatibilidad pasando de 32 bits a 8 bits
                        # Si pulsas la tecla "n" se pasa la foto sin etiquetar coordenadas
                        if key == ord("n") or key == ord("N"):
                            break
                        # Si se pulta la techa "DEL" se descarta la foto
                        if key == 8:
                            descartar=True
                            break
                        # Si se pulsa la tecla la techa "ESC" se cierra el programa
                        if key == 27:
                            cv2.destroyAllWindows()
                            sys.exit()
                    #cv2.destroyAllWindows() # Descomentar esta línea en caso de problemas de RAM
                if not descartar:
                    if len(puntos) != 8:
                        # Si no se ha reescrito algún punto manualmente, se utilizan los que predijo el modelo
                        for i in range(0,len(puntos),2):
                            # Calculamos la distancia entre cada punto elegido y los puntos predichos para ver el más cercano
                            dist=[round(np.sqrt((puntos[i]-coords_pixel[j])**2+(puntos[i+1]-coords_pixel[j+1])**2)) for j in range(0,len(coords_pixel),2)]
                            cercano=dist.index(min(dist))
                            # El punto más cercano de los predichos se elimina porque ya hay uno marcado manualmente
                            del coords_pixel[cercano*2:cercano*2+2]
                        # Se añaden los puntos predichos por el modelo que falten
                        puntos.extend(coords_pixel)
                        # Se recorta y almacena la imágen recortada y se pasa a la siguiente
                    recortado = recorte_img(imagen_actual,puntos,ancho=1000,alto=1000)
                    cv2.imwrite(f"{dir_salida}/{image_name}",recortado)
        cv2.destroyAllWindows() #Destruimos cualquier posible ventana residual que pudiera quedar


