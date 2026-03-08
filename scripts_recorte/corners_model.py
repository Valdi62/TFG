import torch.nn as nn
from torchvision import models

class TransferLearning(nn.Module):
    def __init__(self,base_model,dropout=0.4,size1=512,size2=128):
        super().__init__()

        # Cargar el modelo deseado
        if base_model == "MobileNetV2":
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)   
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(1280,size1))

        elif base_model == "ResNet34":
           self.model  = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
           self.model.fc = nn.Identity()
           self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(512,size1))

        elif base_model == "EfficientNetB3":
            self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(1536,size1))

        else:
            raise ValueError(f"El modelo base {base_model} no está permitido")


        # Congelar todos los pesos del modelo pre-entrenado inicialmente
        for param in self.model.parameters():
            param.requires_grad = False      

        # -- Añadido después de la búsqueda de hiperparámetros al haber elegido este modelo como el mejor --
        # Descongelamos los pesos de los últimos tres bloques del modelo pre-entrenado de la base que hemos elegido usar
        if base_model == "EfficientNetB3":
            for param in self.model.features[-3:].parameters():
                param.requires_grad = True             
                                
        self.layers = nn.Sequential(nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(size1,size2),
                                    nn.ReLU()
                                    )

        self.clas= nn.Sequential(
                            nn.Linear(size2,1) # No necesitamos usar una activacion sigmoide porque usamos BCE con logits
                                )
        self.reg = nn.Sequential(
                            nn.Linear(size2,8),
                            nn.Sigmoid()    # Usamos sigmoide porque las coordenadas estan normalizadas entre 0 y 1
                                )

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        x = self.layers(x)
        has_corners=self.clas(x)
        coordinates=self.reg(x)

        return {"classification": has_corners, "coordinates": coordinates}

