import cv2
import os
import csv
import sys

# Código para etiquetar imágenes marcando las coordenadas de las esquinas en las figuras cuadradas delimitadoras

# Se inicializan las variables globales
corners = []
img = None
img_name = None
# Archivo de salida
csv_salida = "esquinas.csv"

def click_event(event,x,y,flags,param):
    """
    Función con el callback que se usará para marcar las esquinas de las imágenes.

    Parámetros que se usan (el resto se ignoran):
        'event' - tipo de evento que se detecta
        'x'     - la coordenada x en la que se ha hecho click
        'y'     - la coordenada y en la que se ha hecho click 
    """    
    # Se trabaja con variables globales
    global corners, img, img_name
    # Se comprueba si se hace click izquierdo para guardar la esquina marcada
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append((x,y))
        # Se dibuja un punto para feedback visual
        cv2.circle(img,(x,y),5,(0,0,255),-1)
        cv2.imshow("Imagen",img)
        # Si ya hay 4 esquinas: guardar y continuar a la siguiente imágen
        if len(corners) == 4:
            with open(csv_salida,mode="a",newline="") as f:
                writer = csv.writer(f)
                registro=[img_name,1]

                for i in range(len(corners)):
                    registro.extend([corners[i][0],corners[i][1]])
                registro
                writer.writerow(registro)
            # Se reinician las coordenadas de las esquinas cuando se termina de tratar la imágen
            corners = []
            cv2.destroyAllWindows()

    # Si se hace click derecho entonces se guarda la imágen indicando que no tiene figura delimitadora y se pasa a la siguiente
    elif event==cv2.EVENT_RBUTTONDOWN:
        with open(csv_salida,mode="a",newline="") as f:
            writer = csv.writer(f)
            registro=[img_name,0]
            for i in range(4):
                registro.extend([0,0])

            writer.writerow(registro)
        corners = []
        cv2.destroyAllWindows()

def main():
    # Se trabaja con variables globales
    global corners,img,img_name
    # Carpeta con las imágenes a etiquetar
    dir = "1_photos"
    imagenes = [f for f in os.listdir(dir) if f.endswith((".jpg",".jpeg",".png"))]
    
    # Crear CSV con encabezados si no existía
    if not os.path.exists(csv_salida):
        with open(csv_salida, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image","has_corners","x1","y1","x2","y2","x3","y3","x4","y4"])

    # Se recorren todas las imágenes del directorio
    for img_name in imagenes:
        img_path = os.path.join(dir,img_name)
        img = cv2.imread(img_path)

        # Ponemos las imagenes enn horizontal y les damos a todas las mismas dimensiones
        h,w=img.shape[:2]
        if w<h:
            img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.resize(img,(1920,1080))    
        
        # Mostrar la imágen para seleccionar las esquinas de la figura
        cv2.imshow("Imagen",img)
        cv2.setMouseCallback("Imagen",click_event)

        # Si se presiona "ESC" se acaba la ejecución del programa
        key = cv2.waitKey(0) & 0xFF # Aseguramos la compatibilidad pasando de 32 bits a 8 bits
        if key == 27:  # 27 = tecla "ESC"
            cv2.destroyAllWindows()
            sys.exit()  # Termina el programa completamente 
    
    
if __name__ == "__main__":
    main()


