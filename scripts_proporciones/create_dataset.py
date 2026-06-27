import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from skmultilearn.model_selection import iterative_train_test_split
from torchvision import transforms
import pandas as pd
import os
import random

## Código con la clase del dataset y la función para dividir el conjunto de datos en train/val/test

class RandomRotation(torch.nn.Module):
    """
    Esta clase se utiliza para realizar rotaciones aleatorias de múltiplos de 90 grados al aplicar data augmentation.
    """
    def forward(self,x):
        angle = random.choice([0,90,180,270]) # Puede darse el caso de que la imagen no rote
        return transforms.functional.rotate(x,angle)

class CustomImageDataset(Dataset):
    """ 
    Esta clase construye el dataset preprocesado a partir de un directorio con imágenes y un data frame con los nombres de las mismas, que opcionalmente 
    tendrá también etiquetas para cada imagen con las proporciones de las clases para cada una de estas, en caso de querer usarse en el entrenamiento o validación.
    En caso de que el dataset tenga los campos de etiquetas y coordenadas deberá especificarse el campo train de la clase.
    
    Parámetros:
        'images_dir'        - ruta del directorio que contiene las fotos recortadas
        'df'                - data frame que contiene los nombres
        'train'             - valor booleano que indica si el dataset se usará para el entrenamiento o no (puede ser para validación o test también)
        'img_size'          - valor numérico que indica el tamaño final que tendrán las imágenes del dataset
        'augmentation'      - valor booleano que indica si queremos aplicar data augmentation al conjunto de datos 
        'hist'              - valor booleano que indica si se va a trabajar con la capa de histograma o no

    El formato del data frame es el siguiente.
    Columna obligatoria:
        'foto'               - nombre de la foto del registro
    Columnas opcionales solo si train es igual a True:
        'n. noltei'          - porcentaje de presencia que tiene este elemento en la foto     
        'z. marina'          - porcentaje de presencia que tiene este elemento en la foto     
        'g. vermiculophylla' - porcentaje de presencia que tiene este elemento en la foto             
        'sedimento'          - porcentaje de presencia que tiene este elemento en la foto     
        'arena'              - porcentaje de presencia que tiene este elemento en la foto 
        'roca'               - porcentaje de presencia que tiene este elemento en la foto 
        'algas verdes'       - porcentaje de presencia que tiene este elemento en la foto         
        'algas pardas'       - porcentaje de presencia que tiene este elemento en la foto         
        'algas rojas'        - porcentaje de presencia que tiene este elemento en la foto     
        'microfitobentos'    - porcentaje de presencia que tiene este elemento en la foto          
    """
    def __init__(self,images_dir,df,train=False,img_size=384,augmentation=False,hist=False):
        self.images_dir = images_dir
        self.images = df["foto"].values
        self.augmentation = augmentation
        self.hist = hist

        self.transform = transforms.Compose([
                                            transforms.Resize((img_size, img_size)), # Transforma el tamaño de las imagenes
                                            transforms.ConvertImageDtype(torch.float32), # Convierte las imagenes a flotantes en el rango 0-1
                                            ])
        self.imagenet_norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Realiza la misma normalización que se hace con las imagenes de imagenet

        # Transformaciones que se van a aplicar para para el data augmentation
        self.augment_transform = transforms.Compose([
                                # Volteos como si se usara un espejo
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                # Rotaciones de múltiplos de 90 grados
                                RandomRotation(),
                                # Ajustes de color en las imágenes
                                transforms.RandomApply([
                                transforms.ColorJitter(
                                    brightness=0.25,  # Variación de brillo
                                    contrast=0.25,    # Variación de contraste
                                    saturation=0.2,   # Variación de saturación
                                    hue=0.03          # Variación de tono, se elige un valor bajo para no cambiar mucho los colores
                                )],p=0.3),
                                transforms.RandomApply([
                                transforms.GaussianBlur(kernel_size=5,sigma=(0.1,1.5) # Simulamos ruido gaussiano con efecto borroso
                                )],p=0.2)])

        # Obtenemos las etiquetas si se quiere usar el dataset para el proceso de entrenamiento
        self.train = train
        self.target_cols = ['n. noltei','z. marina','g. vermiculophylla','sedimento','arena','roca','algas verdes','algas pardas','algas rojas','microfitobentos']

        if self.train:
            try:
                self.labels = df[self.target_cols].values.astype(np.float32) 
            except KeyError:
                # En caso de marcar el dataset para entrenar pero no proporcionar las etiquetas correspondientes
                raise ValueError(f"Para crear un dataset de entrenamiento es obligatorio tener las siguientes columnas en el data frame: {self.target_cols}")
            
            # Comprobamos si todas las filas suman 100 o 1
            tot = self.labels.sum(axis=1)
            # Si todas las filas suman 100 dividimos entre dicho valor para que pasen a sumar 1
            # Se usa "np.allclose" para permitir una pequeña tolerancia y que puedan sumar 99.9 o 100.1
            if np.allclose(tot,100,atol=0.1):
                self.labels = self.labels/100
            # Si las filas no sumaban 100 y comprobamos que tampoco suman 1 entonces hay algo incorrecto
            # Aqui reducimos la tolerancia por trabajar en el rango 0-1 siendo menor que el 0-100
            elif not np.allclose(tot,1,atol=0.001):
                raise ValueError("Todas las filas de las etiquetas tienen que sumar 1 o 100 en total")
        else:
            self.labels = None
            
    def __len__(self):
        # Se devuelve la longitud del dataset
        return len(self.images)

    def __getitem__(self, idx):
        # Se extrae un elemento concreto del dataset
        image_name = self.images[idx]
        img_path = f"{self.images_dir}/{image_name}"
        image = read_image(img_path) # De esta forma se pasan directamente a tensor

        # Si se utiliza data augmentation se aplican transformaciones aleatorias
        if self.augmentation: 
            image = self.augment_transform(image)

        # Se aplican las transformaciones clásicas para tratar con las imágenes
        image = self.transform(image)
        image_norm = self.imagenet_norm(image)

        # Si el dataset se va a usar en entrenamiento devolvemos las etiquetas, si es para inferencia devolvemos el nombre de la imagen
        if self.train:
            labels = torch.tensor(self.labels[idx])
        else:
            labels = image_name
        
        # Si queremos usar la capa de histograma en el modelo, se necesita por un lado las imagenes normalizadas con los valores de imagenet,
        # para la pasada por la base del modelo, y luego las imagenes clásicas simplemente normalizadas entre 0 y 1 para pasar por la capa de histogramas
        if self.hist:
            return image_norm,labels,image
        else:
            return image_norm,labels,[]

    

def create_dataset(df,target_cols,out_dir,train_size=0.7):
    """
    Función para dividir un conjunto de datos en formato data frame de pandas en conjuntos de train, val y test
    Se almacenarán en formato csv con esos mismos nombres, en un directorio de salida elegido.

    Parámetros:
        'df'           - data frame con los registros a separar por conjuntos
        'target_cols'  - lista con las columnas que conforman las etiquetas
        'out_dir'      - nombre del directorio de salida en el que almacenar los datasets divididos
        'train_size'   - tamaño del conjunto de entrenamiento (el de validación y el de test siempre será cada uno la mitad del tamaño restante)
    """
    # Se crea el directorio de salida en caso de no existir
    os.makedirs(out_dir,exist_ok=True)

    # Creamos los data frames con train, val y test en un 70/15/15
    class_present = (df[target_cols]>0).astype(int).values
    # Como existe desequilibrio entre clases y algunas aparecen en muy pocas fotos, se utiliza estratificación iterativa.
    # De forma que se van repartiendo las clases entre conjuntos de forma equitativa, empezando por las más raras
    train,_,trial,_ = iterative_train_test_split(df.values,class_present,test_size=1-train_size)
    train = pd.DataFrame(train,columns=df.columns)
    trial = pd.DataFrame(trial,columns=df.columns)

    class_present_trial = (trial[target_cols]>0).astype(int).values
    val,_,test,_ = iterative_train_test_split(trial.values,class_present_trial,test_size=0.5)
    val = pd.DataFrame(val,columns=df.columns)
    test = pd.DataFrame(test,columns=df.columns)

    # Almacenamos los dataframes en formato csv
    train.to_csv(f"{out_dir}/train.csv",index=False)
    val.to_csv(f"{out_dir}/val.csv",index=False)
    test.to_csv(f"{out_dir}/test.csv",index=False)


def main():
    """
    Desde el main de este script se crea un conjunto de entrenamiento, validación y test utilizando estratificación iterativa para solventar 
    posibles problemas de desequilibrio de clases, para ello se pasa una lista con las columnas que nos interesan como etiqueta.
    """
    # Fijamos una semilla por replicabilidad
    torch.manual_seed(67)
    np.random.seed(67)

    # Cargamos el csv con las etiquetas
    labels = pd.read_csv("etiquetas_fotos.csv")
    target_cols = ['n. noltei','z. marina','g. vermiculophylla','sedimento','arena','roca','algas verdes','algas pardas','algas rojas','microfitobentos']
    out_dir = "./dataset_dividido"

    # Creamos y almacenamos los datasets divididos
    create_dataset(labels,target_cols,out_dir,train_size=0.7)
    
    
if __name__ == "__main__":
    main()