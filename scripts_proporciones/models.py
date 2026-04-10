import torch
import torch.nn as nn
from torchvision import models
from . HardHistogramBatched import HardHistogramBatched
from . HardHistogram import HardHistogram

# --- MR = MultivariateRegression ---

# 1. Red Neuronal clásica de multiregresión que usa una red convolucional como base
class MRConvolutionalModel(nn.Module):
    def __init__(self,base_model,dropout=0.2,size1=1024,size2=512,use_histogram=False,num_bins=32):
        super().__init__()
        self.base_model = base_model
        self.use_histogram = use_histogram
        self.num_bins = num_bins

        if self.use_histogram:
            # Instanciamos la capa de histograma
            self.histogram = HardHistogramBatched(n_features=3,num_bins=self.num_bins)
            # Instanciamos un conjunto de capas para preparar la salida del histograma y poder concatenarla con la salida normal de la red
            self.out_channels = 3*num_bins
            self.hist_proj = nn.Sequential(nn.Linear(self.out_channels,size2),
                                           nn.ReLU())

        # Cargar el modelo deseado
        if self.base_model == "RegNet_Y_3_2GF":
            self.model = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Identity()
            self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(1000,size1))
        elif self.base_model == "ResNet50":
           self.model  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
           self.model.fc = nn.Identity()
           self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(2048,size1))
        elif self.base_model == "EfficientNetV2_small":
            self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(1280,size1))
            
        elif self.base_model == "ConvNeXt_tiny":
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Flatten(),nn.LayerNorm(768),nn.Dropout(dropout),nn.Linear(768,size1))
        elif self.base_model == "ConvNeXt_small":
            self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Flatten(),nn.LayerNorm(768),nn.Dropout(dropout),nn.Linear(768,size1))
        else:
            raise ValueError(f"El modelo base {self.base_model} no está permitido, elija entre ['RegNet_Y_3_2GF','ResNet50','EfficientNetV2_small','ConvNeXt_tiny','ConvNeXt_small']")

        # Congelar todos los pesos del modelo base pre-entrenado inicialmente
        for param in self.model.parameters():
            param.requires_grad = False              

        self.layers = nn.Sequential(#nn.ReLU(),
                                    nn.GELU(),  # Como estamos utilizando ConvNext usamos Gelu en vez de Relu porque su arquitectura emplea esta función
                                    nn.Dropout(dropout),
                                    nn.Linear(size1,size2),
                                    #nn.ReLU(),
                                    nn.GELU()
                                    )
        if self.use_histogram:
            final_size = size2+size2
        else:
            final_size = size2

        self.output = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(final_size,10),
                                    nn.LogSoftmax(dim=1)) # Capa de salida logsoftmax para usar la divergencia kl con logaritmos como funcion de perdida

    def forward(self,x):
        ## Pasada por la red convolucional ##
        x_conv = self.model(x)
        x_conv = self.head(x_conv)
        x_conv = self.layers(x_conv)

        if self.use_histogram: 
            ## Preprocesado de la entrada para poder pasar por la capa de histograma
            # Para aplicar la capa de histograma podemos reducir el tamaño general de las imágenes para que no se agote la memoria por las operaciones
            # Al usar el modo bilinear cada pixel resultante se calcula a partir de la media de pixeles cercanos en la imagen original
            x_reduced = nn.functional.interpolate(x,size=(128,128),mode='bilinear',align_corners=False)
            # Combinamos las dimensiones de alto y ancho al final para aplanar la imagen
            x_reduced = x_reduced.view(x_reduced.shape[0],x_reduced.shape[1],-1)
            # Calculamos el menor y mayor valor de cada canal de la imagen
            x_min = x_reduced.min(dim=2,keepdim=True).values
            x_max = x_reduced.max(dim=2,keepdim=True).values
            # Normalizamos las entradas entre 0 y 1, añadimos 1e-8 para evitar la división por 0
            x_norm = (x_reduced-x_min)/(x_max-x_min+1e-8)
            # Intercambiamos las dos últimas dimensiones porque luego la capa de histograma las vuelve a intercambiar y las dejará correctamente
            x_norm = x_norm.mT

            ## Pasada por la capa de Histograma ##
            x_hist = self.histogram(x_norm)
            x_hist = self.hist_proj(x_hist)

            # Concatenamos la salida de la red con la salida del histograma (Revisar)
            x_def = torch.cat([x_conv,x_hist],dim=1)
        else:
            x_def = x_conv
            
        return self.output(x_def)

    @property
    def name(self):
        if self.use_histogram:
            return f"MRConvolutional_Histogram_{self.base_model}"
        else:
            return f"MRConvolutional_{self.base_model}"
    


# 2. Red Neuronal que usa un Vision Transformer como base
class MRVisionTransformer(nn.Module):
    def __init__(self,base_model,dropout=0.2,size1=512,size2=128,use_histogram=False,num_bins=32):
        super().__init__()
        self.base_model = base_model
        self.use_histogram = use_histogram
        self.num_bins = num_bins

        if self.use_histogram:
            # Instanciamos la capa de histograma
            self.histogram = HardHistogramBatched(n_features=3,num_bins=self.num_bins)
            # Instanciamos un conjunto de capas para preparar la salida del histograma y poder concatenarla con la salida normal de la red
            self.out_channels = 3*num_bins
            self.hist_proj = nn.Sequential(nn.Linear(self.out_channels,size2),
                                           nn.ReLU())

        # Cargar el modelo deseado
        if self.base_model == "ViT_B_16":
           self.model  = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
           self.model.heads = nn.Identity()
           self.head = nn.Sequential(nn.LayerNorm(768),nn.Dropout(dropout),nn.Linear(768,size1))
        elif self.base_model == "Swin_V2_S":
           self.model  = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
           self.model.head = nn.Identity()
           self.head = nn.Sequential(nn.LayerNorm(768),nn.Dropout(dropout),nn.Linear(768,size1))
        # El tercer Vision Transformer que vamos a probar se carga desde el repositorio de Meta
        elif self.base_model == "DINOv2_ViT_B":
            self.model = torch.hub.load("facebookresearch/dinov2","dinov2_vitb14")
            # DINOv2_ViT_B no tiene cabeza de clasificación
            self.head = nn.Sequential(nn.LayerNorm(768),nn.Dropout(dropout), nn.Linear(768, size1))

        else:
            raise ValueError(f"El modelo base {self.base_model} no está permitido, elija entre ['ViT_B_16','Swin_V2_S','DINOv2_ViT_B']")

        # Congelar todos los pesos del modelo pre-entrenado inicialmente
        for param in self.model.parameters():
            param.requires_grad = False              
                                
        self.layers = nn.Sequential(nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(size1,size2),
                                    nn.GELU()
                                    )
        if self.use_histogram:
            final_size = size2+size2
        else:
            final_size = size2

        self.output = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(final_size,10),
                                    nn.LogSoftmax(dim=1)) # Capa de salida logsoftmax para usar la divergencia kl con logaritmos como funcion de perdida
                                   

    def forward(self,x):
        ## Pasada por la red convolucional ##
        x_conv = self.model(x)
        x_conv = self.head(x_conv)
        x_conv = self.layers(x_conv)

        if self.use_histogram: 
            ## Preprocesado de la entrada para poder pasar por la capa de histograma
            # Para aplicar la capa de histograma podemos reducir el tamaño general de las imágenes para que no se agote la memoria por las operaciones
            # Al usar el modo bilinear cada pixel resultante se calcula a partir de la media de pixeles cercanos en la imagen original
            x_reduced = nn.functional.interpolate(x,size=(128,128),mode='bilinear',align_corners=False)
            # Combinamos las dimensiones de alto y ancho al final para aplanar la imagen
            x_reduced = x_reduced.view(x_reduced.shape[0],x_reduced.shape[1],-1)
            # Calculamos el menor y mayor valor de cada canal de la imagen
            x_min = x_reduced.min(dim=2,keepdim=True).values
            x_max = x_reduced.max(dim=2,keepdim=True).values
            # Normalizamos las entradas entre 0 y 1, añadimos 1e-8 para evitar la división por 0
            x_norm = (x_reduced-x_min)/(x_max-x_min+1e-8)
            # Intercambiamos las dos últimas dimensiones porque luego la capa de histograma las vuelve a intercambiar y las dejará correctamente
            x_norm = x_norm.mT

            ## Pasada por la capa de Histograma ##
            x_hist = self.histogram(x_norm)
            x_hist = self.hist_proj(x_hist)

            # Concatenamos la salida de la red con la salida del histograma (Revisar)
            x_def = torch.cat([x_conv,x_hist],dim=1)
        else:
            x_def = x_conv
            
        return self.output(x_def)

    @property
    def name(self):

        if self.use_histogram:
            return f"MRVisionT_Histogram_{self.base_model}"
        else:
            return f"MRVisionT_{self.base_model}"
        
        