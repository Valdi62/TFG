import pandas as pd
import os
import cv2
import random
# Data augmentation para las imágenes recortadas de las cuales vamos a calcular su proporción de clases.

# --- Hacemos data augmentation para tener más fotos variadas haciendo que nuestro modelo sea más robusto frente a factores
#     como la iluminación o la orientación de las fotografías, para las pruebas de hiperparámetros usaremos el csv con las
#     etiquetas de las fotos originales, pero para el entrenamiento final de los modelos que usen ese tipo de etiquetas vamos
#     a usar un csv que incluye tanto las fotos originales como las fotos creadas artificialmente.
#     En general data augmentation ayuda al modelo a generalizar mejor al ver más datos y más variados

def random_mod(img):
    """
    Función que elige aleatoriamente transformaciones y se la aplica a la imagen de entrada.

    Parámetros:
        'img'             - imagen de entrada para modificar

    Transformaciones:
        'Flip horizontal' - reflejo horizontal de la imagen
        'Flip Vertical'   - reflejo vertical de la imagen
        'Rotación'        - rotación de la imagen que puede ser de 90, 180 o 270 grados para mantener la forma
        'Brillo'          - modificamos ligeramente el brillo de la foto multiplicandolo por un factor
        'Desenfoque'      - aplicamos un filtro gaussiano con lo que aumentamos ligeramente el desenfoque de la foto
    """
    # Flip horizontal
    if random.random() < 0.5:
        img = cv2.flip(img,1)
    # Flip vertical
    if random.random() < 0.5:
        img = cv2.flip(img,0)
    # Rotación que puede no darse o ser de 90, 180 o 270 grados
    if random.random() < 0.5:
        angle = random.choice([cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_COUNTERCLOCKWISE])
        img = cv2.rotate(img,angle)
    # Modificación aleatoria del brillo
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.7, 1.3)
        img = cv2.convertScaleAbs(img,alpha=brightness_factor)
    # Desenfoque aleatorio
    if random.random() < 0.2:
        kernel_size = random.choice([3,5,7])
        img = cv2.GaussianBlur(img,(kernel_size,kernel_size),0)

    return img


def random_data_augmentation(original_df,in_dir,out_dir):
    """
    Función para generar nuevas fotos modificadas a partir de las fotos originales.
    Se toma un data frame de entrada del cual se cargarán sus imágenes del campo 'foto' y se aplicarán aleatoriamente transformaciones 
    a cada una, posteriormente se almacenarán esas imágenes modificadas en la carpeta elegida y se devolverá un data frame con el mismo
    formato que el de entrada pero con las nuevas fotos y sus etiquetas correspondientes, que serán las mismas que en el original.

    Parámetros:
        'original_df' - data frame de entrada con las etiquetas que se esperan y un campo 'foto' obligatorio
        'in_dir'      - ruta al directorio en el que se tienen almacenadas las fotos del data frame
        'out_dir'     - ruta al directorio en el que se van a guardar las nuevas imágenes modificadas
    """
    # Lo primero que hacemos es crear el directorio de salida si no existiera
    os.makedirs(out_dir,exist_ok=True)

    new_rows = []
    # Vamos a iterar en el data frame para ir modificando las imágenes una por una
    for _,i in original_df.iterrows():
        img_name = i["foto"]
        img_path = os.path.join(in_dir,img_name)
        img = cv2.imread(img_path)

        # Aplicamos transformaciones aleatorias que mantengan las proporciones para poder utilizar las etiquetas originales
        mod_img = random_mod(img)

        # Generamos el nombre de la nueva foto modificada que será muy parecido al anterior
        new_name = f"RE_{img_name}"
        out_path = os.path.join(out_dir,new_name)
        
        # Guardamos la imagen modificada
        cv2.imwrite(out_path,mod_img)
        
        # Creamos una fila para cada imagen modificada con su nombre y las mismas etiquetas que la foto original
        new_row = i.copy()
        new_row["foto"] = new_name
        new_rows.append(new_row)
    
    # Creamos el nuevo data frame que devolveremos
    mod_df = pd.DataFrame(new_rows)   
    return mod_df


def main():
    df = pd.read_csv("etiquetas_fotos.csv")
    # En este caso el directorio del que se toman las fotos y el directorio en el que se guardan es el mismo
    dir = "./fotos_recortadas"
    mod_df = random_data_augmentation(df,dir,dir)

    # Creamos un csv con las etiquetas de estas nuevas imágenes modificadas
    mod_df.to_csv("etiquetas_augmented.csv",index=False)
    

if __name__ == "__main__":
    main()