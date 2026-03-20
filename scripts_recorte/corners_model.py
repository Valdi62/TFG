import torch.nn as nn
from torchvision import models

class TransferLearning(nn.Module):
    def __init__(self,base_model,dropout=0.4,size1=512,size2=128):
        super().__init__()
        self.base_model=base_model

        # Cargar el modelo deseado
        if self.base_model == "ResNet50":
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
            self.head = nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(768,size1))
        # Si queremos que los modelos se puedan ejecutar de forma agil en dispositivos móviles para estudios de campo, este modelo ligero puede ser una buena alternativa
        elif self.base_model == "MobileNet_V3_Large":
            self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            self.model.classifier = nn.Identity()
            self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(960,size1))

        else:
            raise ValueError(f"El modelo base {self.base_model} no está permitido, elija entre ['ResNet50','EfficientNetV2_small','ConvNeXt_tiny','MobileNet_V3_Large']")

        # Congelar todos los pesos del modelo pre-entrenado inicialmente
        for param in self.model.parameters():
            param.requires_grad = False      
                                
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

