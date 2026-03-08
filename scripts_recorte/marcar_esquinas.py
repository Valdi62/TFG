import cv2
import os
import csv
import sys

# Archivo de salida
salida_csv = "esquinas.csv"
puntos = []
img = None
nombre_imagen = None

def click_event(event, x, y, flags, param):
    global puntos, img, nombre_imagen
    if event == cv2.EVENT_LBUTTONDOWN:
        puntos.append((x, y))
        # Dibujar el punto
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Imagen", img)

        # Si ya hay 4 puntos: guardar y continuar
        if len(puntos) == 4:
            with open(salida_csv, mode="a", newline="") as f:
                writer = csv.writer(f)
                registro=[nombre_imagen,1]

                for i in range(len(puntos)):
                    registro.extend([puntos[i][0],puntos[i][1]])
                registro
                writer.writerow(registro)
            puntos = []
            cv2.destroyAllWindows()

    elif event== cv2.EVENT_RBUTTONDOWN:
        with open(salida_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            registro=[nombre_imagen,0]
            for i in range(4):
                registro.extend([0,0])

            writer.writerow(registro)
        puntos = []
        cv2.destroyAllWindows()

def main():
    global puntos, img, nombre_imagen
    # Carpeta con las imágenes a etiquetar
    dir = "1_photos"
    imagenes = [f for f in os.listdir(dir) if f.endswith((".jpg",".jpeg",".png"))]
    
    # Crear CSV con encabezados si no existía
    if not os.path.exists(salida_csv):
        with open(salida_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image","has_corners","x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])

    # Recorremos todas las imágenes
    for nombre_imagen in imagenes:
        img_path = os.path.join(dir, nombre_imagen)
        img = cv2.imread(img_path)

        # Ponemos las imagenes enn horizontal y les damos a todas las mismas dimensiones
        h,w=img.shape[:2]
        if w<h:
            img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.resize(img,(1920,1080))    
        
        cv2.imshow("Imagen", img)
        cv2.setMouseCallback("Imagen", click_event)

        # Si presionamos escape se acaba la ejecución del programa
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # 27 = tecla ESC
            cv2.destroyAllWindows()
            sys.exit()  # Termina el programa completamente 
    
    
if __name__ == "__main__":
    main()


