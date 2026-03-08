import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import rotate
from torchvision import transforms

class CustomImageDataset(Dataset):
    """ 
    Clase que construye el dataset preprocesado a partir de un directorio con imágenes y un data frame con los nombres de las imagenes que opcionalmente 
    tendrá etiquetas para cada imagen junto con las coordenadas de las esquinas en dichas imágenes en caso de querer usarse para el entrenamiento del modelo.
    El preprocesado se realiza de la misma forma que los ejemplos usados en el entrenamiento para poder pasar el dataset directamente por la red y obtener 
    resultados coheretes. En caso de que el dataset tenga los campos de etiquetas y coordenadas deberá especificarse el campo train de la clase

    Parámetros:
        'images_dir'  - ruta del directorio que contiene las fotos recortadas
        'df'          - data frame que contiene los nombres
        'train'       - valor booleano que indica si el dataset se usará para el entrenamiento o no
    
    El formato del data frame es el siguiente.
    Columna obligatoria:
        'image'       - nombres de las imágenes en orden
    Columnas opcionales solo si train es igual a True:
        'has_corners' - 0 si no hay esquinas en la imagen y 1 si las hay
        'x1'          - posición de x de la primera esquina
        'y1'          - posición de y de la primera esquina
        'x2'          - posición de x de la segunda esquina
        ...
    Se continuan los campos de coordenadas hasta y4 que tiene la posición y de la cuarta esquina
    """
    def __init__(self, images_dir, df, train=False):
        self.images_dir = images_dir
        self.images = df["image"].values
        self.transform = transforms.Compose([
                         transforms.Resize((270, 480)), # Transforma el tamaño de las imagenes
                         transforms.ConvertImageDtype(torch.float32), # Convertimos las imagenes a flotantes en el rango 0-1 para la normalización posterior
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Realizamos la misma normalización que se hace con las imagenes de imagenet
                         ])
        # Obtenemos las etiquetas solo si queremos usar el dataset para el entrenamiento
        self.train = train
        self.target_cols = ['has_corners','x1','y1','x2','y2','x3','y3','x4','y4']
        self.labels = df[self.target_cols].values.astype(np.float32) if self.train else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name=self.images[idx]
        img_path = f"{self.images_dir}/{image_name}"
        image = read_image(img_path) # De esta forma se pasan directamente a tensor

        # Image.shape[0] son el numero de canales
        if image.shape[2] < image.shape[1]: # Rotamos si está en vertical
            image = rotate(image, 90, expand=True)
        if self.transform: # Aplicamos las transformaciones de tamaño
            image = self.transform(image)

        # Comprobamos si tenemos las etiquetas para devolver el conjunto completo para entreamiento o solo las imágenes
        if self.train:
            has_corners = self.labels[idx][0]
            coords = self.labels[idx][1:]
            has_corners = torch.tensor(has_corners)

            ###   ¡¡¡IMPORTANTE!!!   ###
            # Normalizamos las coordenadas en función de estas dimensiones fijas porque son las que se usaro durante su etiquetado
            # es decir todas las fotos se rotaron y ajustaron en tamaño antes de etiquetarlas
            coords = torch.tensor(coords)
            coords = coords/np.array([1920,1080]*4,dtype=np.float32)
            return image,{"classification": has_corners, "coordinates": coords}
        else:
            return image,image_name
    
