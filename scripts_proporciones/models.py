import torch.nn as nn
from torchvision import models
from . HardHistogramBatched import HardHistogramBatched

# --- MR = MultivariateRegression ---

# 1. Red Neuronal clásica de multiregresión que usa una red convolucional como base
class MRConvolutionalModel(nn.Module):
    def __init__(self,base_model,dropout=0.2,size1=512,size2=128):
        super().__init__()
        self.base_model = base_model
        # Cargar el modelo deseado
        if self.base_model == "RegNetY_3_2GF":
            self.model = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
            self.model.fc = nn.Identity()   # Eliminamos directamente la cabeza original para poder congelar los pesos de todo el modelo base en un solo paso
            self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(1000,size1))

        elif self.base_model == "ResNet50":
           self.model  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
           self.model.fc = nn.Identity()
           self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(2048,size1))

        elif self.base_model == "EfficientNetV2_small":
            self.model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(1280,size1))

        # ConvNeXt es la familia ganadora y por tanto probamos en profundidad sus modelos
        elif self.base_model == "ConvNeXt_tiny":
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(768,size1))
        elif self.base_model == "ConvNeXt_small":
            self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(768,size1))
        
        else:
            raise ValueError(f"El modelo base {self.base_model} no está permitido, elija entre ['RegNetY_3_2GF','ResNet50','EfficientNetV2_small','ConvNeXt_tiny','ConvNext_small']")

        # Congelar todos los pesos del modelo pre-entrenado inicialmente
        for param in self.model.parameters():
            param.requires_grad = False              
                                
        self.layers = nn.Sequential(nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(size1,size2),
                                    # nn.BatchNorm1d(size2),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(size2,10),
                                    nn.LogSoftmax(dim=1) # Capa de salida logsoftmax para usar la divergencia kl con logaritmos como funcion de perdida
                                   )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.layers(x)
        return x

    @property
    def name(self):
        return f"MRConvolutional_{self.base_model}"



# 2. Red Neuronal clásica de multiregresión que usa una red convolucional como base y una capa con histogramas a la salida de la base
class MRConvolutionalModelHistogram(nn.Module):
    def __init__(self,base_model,num_bins=32,dropout=0.2,size1=1024,size2=512,size3=128):
        super().__init__()
        self.base_model = base_model
        self.num_bins = num_bins

        # Cargar el modelo deseado
        # El modelo usa la misma base que el anterior pero usa una capa de histograma adicional después de la cabeza del modelo base
        if self.base_model == "ConvNeXt_tiny":
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.model.avgpool = nn.Identity() # Eliminamos también la capa de pooling global antes de la capa de histograma
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Flatten(2),                                    
                                      HardHistogramBatched(768,self.num_bins),
                                      nn.Dropout(dropout),
                                      nn.Linear(768*self.num_bins,size1),
                                      nn.ReLU())
            
        elif self.base_model == "ConvNeXt_small":
            self.model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            self.model.avgpool = nn.Identity()
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Flatten(2),
                                      HardHistogramBatched(768,self.num_bins),
                                      nn.Dropout(dropout),
                                      nn.Linear(768*self.num_bins,size1),
                                      nn.ReLU())
        else:
            raise ValueError(f"El modelo base {self.base_model} no está permitido, elija entre ['RegNetY_3_2GF','ResNet50','EfficientNetV2_small','ConvNeXt_tiny','ConvNext_small']")

        # Congelar todos los pesos del modelo pre-entrenado inicialmente
        for param in self.model.parameters():
            param.requires_grad = False              

        self.layers = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(size1,size2),
                                    # nn.BatchNorm1d(size2),
                                    nn.ReLU(),

                                    nn.Dropout(dropout),
                                    nn.Linear(size2,size3),
                                    # nn.BatchNorm1d(size3),
                                    nn.ReLU(),
                                    
                                    nn.Dropout(dropout),
                                    nn.Linear(size3,10),
                                    nn.LogSoftmax(dim=1) # Capa de salida logsoftmax para usar la divergencia kl con logaritmos como funcion de perdida
                                   )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.layers(x)
        return x

    @property
    def name(self):
        return f"MRConvolutional_Histogram_{self.base_model}"
    


# 3. Red Neuronal que usa un Vision Transformer como base
class MRVisionTransformer(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.base_model = base_model