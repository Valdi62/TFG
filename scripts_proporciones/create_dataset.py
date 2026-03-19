import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from skmultilearn.model_selection import iterative_train_test_split
from torchvision import transforms
import pandas as pd
import os


class CustomImageDataset(Dataset):
    """ 
    Clase que construye el dataset preprocesado a partir de un directorio con imágenes y un data frame con los nombres de las imagenes que opcionalmente 
    tendrá etiquetas para cada imagen junto con las coordenadas de las esquinas en dichas imágenes en caso de querer usarse para el entrenamiento del modelo.
    El preprocesado se realiza de la misma forma que los ejemplos usados en el entrenamiento para poder pasar el dataset directamente por la red y obtener 
    resultados coheretes. En caso de que el dataset tenga los campos de etiquetas y coordenadas deberá especificarse el campo train de la clase.
    Las transformaciones que se le apliquen a las imágenes al crear el dataset no se deben modificar puesto que tienen que coincidir con las que se aplicaron
    a las imágenes que se usaron en el entrenamiento de los modelos, por tanto no hay un parámetro de entrada para modificarlas.
    
    Parámetros:
        'images_dir'        - ruta del directorio que contiene las fotos recortadas
        'df'                - data frame que contiene los nombres
        'train'             - valor booleano que indica si el dataset se usará para el entrenamiento o no  

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
    def __init__(self, images_dir,df,train=False):
        self.images_dir = images_dir
        self.images = df["foto"].values
        self.transform = transforms.Compose([
                         transforms.Resize((400,400)), # Transforma el tamaño de las imagenes
                         transforms.ConvertImageDtype(torch.float32), # Convertimos las imagenes a flotantes en el rango 0-1 para la normalización posterior
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Realizamos la misma normalización que se hace con las imagenes de imagenet
                         ])
        # Obtenemos las etiquetas solo si queremos usar el dataset para el entrenamiento
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
            # usamos "np.allclose" para permitir una pequeña tolerancia y que puedan sumar 99.9 o 100.1
            if np.allclose(tot,100,atol=0.1):
                self.labels = self.labels/100
            # Si las filas no sumaban 100 y comprobamos que tampoco suman 1 entonces hay algo incorrecto
            # Aqui reducimos la tolerancia por trabajar con valores más pequeños
            elif not np.allclose(tot,1,atol=0.001):
                raise ValueError("Todas las filas de las etiquetas tienen que sumar 1 o 100 en total")
            # Si vemos que suman 1 entonces no tenemos que hacer nada

        else:
            self.labels = None
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_path = f"{self.images_dir}/{image_name}"
        image = read_image(img_path) # De esta forma se pasan directamente a tensor

        if self.transform: # Aplicamos las transformaciones de tamaño
            image = self.transform(image)

        # Comprobamos si tenemos las etiquetas para devolver el conjunto completo para entreamiento o solo las imágenes
        if self.train:
            labels = torch.tensor(self.labels[idx])
            return image,labels
        else:
            return image,image_name
    

def create_dataset(df,target_cols,out_dir,train_size=0.7,augmentation=False):
    """
    Función para dividir un conjunto de datos en formato data frame en conjuntos de train, val y test que se almacenarán como 
    documentos csv en un directorio de salida elegido con esos mismos nombres. En caso de querer añadir imagenes creadas mediante
    data augmentation a los datasets se deberá indicar y se añadirán tres archivos csv nuevos que incluirán los registros originales
    asi como los nuevos registros aumentados que tendrán las mismas etiquetas que los originales pero los caracteres 'RE_' delante
    de el nombre de la foto, esto es asi porque se espera que el nombre de las nuevas imágenes aumentadas se haya construido de 
    dicha forma.

    Parámetros:
        df           - data frame con los registros a separar por conjuntos
        target_cols  - lista con las columnas que conforman las etiquetas
        out_dir      - nombre del directorio de salida en el que almacenar los datasets divididos
        train_size   - tamaño del conjunto de entrenamiento (el de validación y el de test siempre será cada uno la mitad del tamaño restante)
        augmentation - parámetro para indicar si se incluirán imágenes aumentadas o no

    """
    # Fijamos una semilla para replicabilidad para este trabajo pero en la práctica se eliminaria para fomental la aleatoriedad
    torch.manual_seed(67)
    np.random.seed(67)

    # Creamos el directorio de salida si no existiera
    os.makedirs(out_dir,exist_ok=True)

    # Creamos los data frames con train, val y test en un 70/15/15, usaremos los mismos data frames para entrenar y probar todos los modelos que usen este tipo de etiquetas
    class_present = (df[target_cols]>0).astype(int).values
    train,_,trial,_ = iterative_train_test_split(df.values,class_present,test_size=1-train_size)
    train = pd.DataFrame(train,columns=df.columns)
    trial = pd.DataFrame(trial,columns=df.columns)

    class_present_trial = (trial[target_cols]>0).astype(int).values
    val,_,test,_ = iterative_train_test_split(trial.values,class_present_trial,test_size=0.5)
    val = pd.DataFrame(val,columns=df.columns)
    test = pd.DataFrame(test,columns=df.columns)

    # Si queremos comprobamos que todas las clases están presentes en los tres conjuntos de datos
    # (train[target_cols]>0).sum()
    # (val[target_cols]>0).sum()
    # (test[target_cols]>0).sum()

    # Almacenamos los dataframes en formato csv
    train.to_csv(f"{out_dir}/train.csv",index=False)
    val.to_csv(f"{out_dir}/val.csv",index=False)
    test.to_csv(f"{out_dir}/test.csv",index=False)

    if augmentation==True:
        train_aug = train.copy()
        train_aug["foto"] = "RE_" + train_aug["foto"]

        val_aug = val.copy()
        val_aug["foto"] = "RE_" + val_aug["foto"]

        test_aug = test.copy()
        test_aug["foto"] = "RE_" + test_aug["foto"]

        # Concatenamos los diferentes data frames
        train_def = pd.concat([train,train_aug],ignore_index=True)
        val_def = pd.concat([val,val_aug],ignore_index=True)
        test_def = pd.concat([test,test_aug],ignore_index=True)

        train_def.to_csv(f"{out_dir}/train_def.csv",index=False)
        val_def.to_csv(f"{out_dir}/val_def.csv",index=False)
        test_def.to_csv(f"{out_dir}/test_def.csv",index=False)


def main():
    """
    Desde el main de este script en caso de ejecutarlo directamente lo que hacemos es crear un conjunto de entrenamiento, validación y test utilizando estratificación
    iterativa para solventar posibles problemas de desequilibrio de clases, para ello le pasamos una lista con las columnas que nos interesan como etiqueta.
    """
    # Cargamos el csv con las etiquetas
    labels = pd.read_csv("etiquetas_fotos.csv")
    ## Como tenemos desequilibrio de clases y algunas aparecen en muy pocas fotos, vamos a utilizar estratificación iterativa de forma que se vayan repartiendo las clases
    ##  de forma equitativa empezando por las más raras
    target_cols = ['n. noltei','z. marina','g. vermiculophylla','sedimento','arena','roca','algas verdes','algas pardas','algas rojas','microfitobentos']
    out_dir = "./dataset_dividido"

    create_dataset(labels,target_cols,out_dir,train_size=0.7,augmentation=True)
    
    
if __name__ == "__main__":
    main()