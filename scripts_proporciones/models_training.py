import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
import plotly.express as px
from . models import MRConvolutionalModel,MRConvolutionalModelHistogram,MRVisionTransformer
from . create_dataset import CustomImageDataset


# Divergencia KL ponderada
class WeightedKLDivLoss(nn.Module):
    """
    Clase con la divergencia KL que permite utilizarla añadiendo un vector de pesos con el que dar diferente importancia a cada clase,
    se puede utilizar para solventar el desequilibrio de clases dándole más peso a las clases menos frecuentes.

    Parámetros al crear el objeto de la clase:
        'class_weights' - vector con el peso de cada clase
        'reduction'     - reducción que se quiere aplicar con la divergencia kl

    Parámetros al aplicar el objeto de la clase:
        'log_preds'     - logaritmos con la predicción del modelo
        'targets'       - etiquetas reales a predecir
    """
    def __init__(self,class_weights,reduction="batchmean"):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self,log_preds,targets):
        # Calculamos la divergencia kl por cada clase
        ## Debido a errores de aproximaciones numéricas del método de kl_div existe la posibilidad de obtener un valor de la convergencia kl negativo, pero
        ##  esto no debería ser posible, es por esto que utilizamos .clamp(0) para forzar a que ese valor no pueda ser negativo
        kl_divergence = F.kl_div(log_preds,targets,reduction="none").clamp(0)
        # Aplicamos los pesos de las clases en función del batch
        weighted_kl_divergence = kl_divergence*self.class_weights.unsqueeze(0) # Añadimos la dimension de batches a class_weights para la multiplicacion
        
        # Retornamos la divergencia KL con los pesos modificados aplicando la reducción deseada
        if self.reduction == "batchmean":
            return weighted_kl_divergence.sum()/log_preds.size(0)
        elif self.reduction == "sum":
            return weighted_kl_divergence.sum()
        elif self.reduction == "mean":
            return weighted_kl_divergence.mean()


# Función para calcular pesos de las clases
def calculate_class_weights(dataloader,n_classes=10,smoothing=0.05,device="cpu"):
    """
    Función que calcula el vector de pesos para las distintas clases de forma que demos más peso a las clases menos frecuentes de nuestro dataset.

    Parámetros:
        'dataloader' - dataloader con los ejemplos de donde sacamos las frecuencias
        'n_classes'  - número total de clases diferentes
        'smoothing'  - valor que controla la relación de pesos:
                          * smoothing = 0 -> se equilibran los pesos para que al considerar todos los ejemplos las clases tengan importancia equitativa
                          * smoothing > 0 -> controla la relación de pesos entre las clases, a menor valor más importancia tendrán las clases infrecuentes
    """
    class_sums = torch.zeros(n_classes).to(device)
    total_samples = 0
    
    # Contamos uno por cada imagen en la que aparezca
    for _,data_labels in dataloader:
        data_labels = data_labels.to(device)
        class_sums += (data_labels>0).sum(dim=0)
        total_samples += data_labels.size(0)
    
    # Proporción de la frecuencia de cada clase
    class_proportions = class_sums/total_samples
    # Pesos de cada clase en función de su frecuencia con smoothing controlando el peso máximo posible
    weights = 1/(class_proportions+smoothing)
    # Por último dividimos los pesos entre la media para que esta sea 1 y se pueda comparar mejor la kl entre datasets diferentes
    weights = weights/weights.mean()
    return weights


# Función para generar las gráficas del entrenamiento
def save_graph(history,module,model,fine_tuning=0):
    # Convertimos a data frame para utilizar plotly a la hora de crear gráficas
    df = pd.DataFrame(history)
    # Agrupamos tanto el error de entrenamiento como el de validación en la misma columna
    df = df.melt(id_vars="epoch",var_name="type", value_name=module)

    # Construimos y configuramos la gráfica
    fig = px.line(df,x="epoch",y=module,color="type",
            title=f"Evolución del {module.upper()} en el entrenamiento de {model}",
            labels={"epoch":"Época",module:f"{module.upper()} loss"})
    #fig.update_layout(template="white")

    # Añadimos una línea vertical para indicar cuando empieza el fine tuning
    if fine_tuning > 1:
        fig.add_vline(x=fine_tuning,line_dash="dash",line_color="red",annotation_text="Fine tuning")

    # Guardamos la gráfica en formato .html para que sea interactiva
    fig.write_html(f"evolucion_{module}_{model}.html")
    

# Bucles de validación
def validation(model,val_dataloader,label_smoothing=0.01,device="cpu"):
    # - Función con el bucle de validación -
    mae_loss_module = nn.L1Loss()
    kl_divergence_module = nn.KLDivLoss(reduction="batchmean")

    val_mae_loss = 0
    val_kl_divergence = 0
    class_mae_loss = 0

    with torch.no_grad():
        model.eval()
        for data_inputs,data_labels in val_dataloader:
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)

            # Vamos a evitar que las etiquetas tengan el valor exacto 0 para evitar que la divergencia kl colapse
            smooth_labels = data_labels*(1-label_smoothing)+label_smoothing/10

            log_preds = model(data_inputs)

            # Calcular el valor de la función de pérdida mae y la divergencia kl
            y_pred = torch.exp(log_preds)

            val_mae_loss += mae_loss_module(y_pred,data_labels).item()
            val_kl_divergence += kl_divergence_module(log_preds,smooth_labels).item()
            # Calculamos ademas el MAE por cada clase individual
            class_mae_loss += torch.mean(torch.abs(y_pred-data_labels),dim=0)

        return val_mae_loss/len(val_dataloader),val_kl_divergence/len(val_dataloader),class_mae_loss/len(val_dataloader)


def train_model(model,opt,train_dataloader,val_dataloader,patience=5,max_epochs=30,learning_rate=0.001,label_smoothing=0.01,
                device="cpu",fine_tuning=False,callback=None,start_epoch=0):
    # - Función con el bucle de entrenamiento de los modelos -
    min_kl_divergence = float('inf')
    no_improvement = 0
    mejor_log = None
    last_epoch = 0

    # Queremos que los objetos history se vayan actualizando según continuamos con el entrenamiento y solo se creen una vez
    global history_mae,history_kl,history_class
    if start_epoch==0:
        history_mae = {"train_mae":[],"val_mae":[]}
        history_kl = {"train_kl":[],"val_kl":[]}
        history_class = {"n. noltei":[],"z. marina":[],"g. vermiculophylla":[],"sedimento":[],"arena":[],"roca":[],"algas verdes":[],
                        "algas pardas":[],"algas rojas":[],"microfitobentos":[]}

    # Como función de pérdida vamos a usar la divergencia kl que nos aproximará la distribución de las probabilidades que predice el modelo con la distribución real
    #  mide cuanta información se pierde usando nuestra distribución para aproximar la distribución real
    # Adicionalmente se calcula el MAE para mostrar por pantalla puesto que es más fácil de interpretar
    # Utilizaremos una clase personalizada de la función divergencia KL con pesos para las clases
    mae_loss_module = nn.L1Loss()
    # Los pesos calculados se usaran tanto en entrenamiento como en validación para asemejar el comportamiento a la inferencia real y usamos un smoothing de 0.05
    #  porque esta a medio camino de 0.1 (que no daba suficiente peso a las clases raras) y 0.01 (que quitaba demasiado peso a las clases comunes)
    class_weights = calculate_class_weights(train_dataloader,n_classes=10,smoothing=0.05,device=device)
    kl_divergence_module = WeightedKLDivLoss(class_weights,reduction="batchmean")
    
    # Vamos a filtrar los parámetros que no están congelados para solo pasarle los que sean estrictamente necesarios al optimizador
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Elegimos el optimizador deseado con su learning rate
    if opt == "SGD":
        optimizer = torch.optim.SGD(trainable_params,lr=learning_rate,momentum=0.9)
    elif opt == "AdamW":
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    elif opt == "RMSprop":
        optimizer = torch.optim.RMSprop(trainable_params, lr=learning_rate)
    elif opt == "Adam":
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    else:
        raise ValueError(f"El optimizado {opt} no está soportado, elija uno entre ['SGD','AdamW','Adam','RMSprop']")
    
    # Vamos a implementar una técnica de reducción del learning rate a medida que el modelo va estancandose en el entrenamiento
    # De esta forma cuando se este acercando al mínimo, se podrán ajustar los pesos de forma más precisa para optimizarlos
    scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=round(2*patience/3),min_lr=1e-7)

    # Bucle de entrenamiento
    pbar = tqdm(range(max_epochs))
    for epoch in pbar:
        last_epoch = epoch

        # Poner lo que sea necesario del modelo en modo entrenamiento
        model.train()
        # Pasamos el modelo base a evaluacion para que la batch_normalization de la base sea fija
        if fine_tuning:
            # Comprobamos cada modulo de la base para dejar aquellos que esten descongelados en train, pero los que
            #  sigan congelados se pongan en eval
            for module in model.model.modules():
                if not any(i.requires_grad for i in module.parameters()):
                    module.eval()
        else:
            model.model.eval()  

        epoch_mae_loss = 0
        epoch_kl_divergence = 0

        for data_inputs,data_labels in train_dataloader:
            # Hacer una pasada hacia delante
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            # Vamos a evitar que las etiquetas tengan el valor exacto 0 para evitar que la divergencia kl colapse
            smooth_labels = data_labels*(1-label_smoothing)+label_smoothing/10

            # Reiniciar los gradientes
            optimizer.zero_grad()
            # Predicción del modelo que saldrá en forma de logaritmos al usar una LogSoftmax
            log_preds = model(data_inputs)

            # Calcular el valor de la función de pérdida mae y la divergencia kl
            mae_loss = mae_loss_module(torch.exp(log_preds),data_labels)
            kl_divergence = kl_divergence_module(log_preds,smooth_labels)
            # Acumular errores
            epoch_mae_loss += mae_loss.item()
            epoch_kl_divergence += kl_divergence.item()

            # Pasada hacia atrás
            kl_divergence.backward()
            # Actualizar los parámetros
            optimizer.step()

        # Calcular la pérdida de validación que nos servirá para evaluar como de bueno es el modelo actual
        val_mae_loss,val_kl_divergence,class_mae_loss = validation(model,val_dataloader,label_smoothing,device)
        class_mae = " | ".join([str(round(i,4)) for i in class_mae_loss.tolist()])

        # Caculamos las pérdidas de entrenamiento promedio para esta etapa
        mean_epoch_kl = epoch_kl_divergence/len(train_dataloader)
        mean_epoch_mae = epoch_mae_loss/len(train_dataloader)

        # Almacenamos los valores de las pérdidas para visualizarlos posteriormente
        history_mae["train_mae"].append(mean_epoch_mae)
        history_mae["val_mae"].append(val_mae_loss)

        history_kl["train_kl"].append(mean_epoch_kl)
        history_kl["val_kl"].append(val_kl_divergence)
        
        for clave,valor in zip(history_class.keys(),class_mae_loss.tolist()):
            history_class[clave].append(valor)

        # Actualizamos con la divergencia kl nuestro scheduler que controla cuando se reduce el learning rate
        scheduler.step(val_kl_divergence)
        
        # Mostramos los valores de las métricas durante el entrenamiento incluyendo el MAE que solo se utiliza para visualizar el error
        current_log=("Training WeightedKL Divergence: %.4f, Validation KL Divergence: %.4f, Training MAE Loss: %.4f, Validation MAE Loss: %.4f, Patience: %d/%d\n"
                     "Validation classes MAE: {%s}"% 
                    (mean_epoch_kl,val_kl_divergence,mean_epoch_mae,val_mae_loss,no_improvement,patience,class_mae))
        
        # Intentamos llamar al callback pero si no lo tenemos definido simplemente lo ignoramos
        if callback is not None:
            callback(val_kl_divergence,start_epoch+last_epoch)
        
        if val_kl_divergence < min_kl_divergence:
            min_kl_divergence = val_kl_divergence
            no_improvement = 0
            mejor_log = current_log
            torch.save(model.state_dict(),f"./{model.name}_best_model.pth")
        else:
            no_improvement += 1
        print(current_log)
        if no_improvement >= patience:
            print("No hay mejora por %d épocas. Parada Temprana!!" % patience)
            break
    print("\nMejor época:")
    print(mejor_log,"\n")

    # Añadimos las épocas que se hayan ejecutado
    x = range(1,len(history_mae[f"train_mae"])+1)
    history_mae["epoch"] = x
    history_kl["epoch"] = x
    history_class["epoch"] = x

    save_graph(history_mae,"mae",model.name,start_epoch+0.5)
    save_graph(history_kl,"kl",model.name,start_epoch+0.5)
    save_graph(history_class,"class_mae",model.name,start_epoch+0.5)
    return min_kl_divergence,start_epoch+last_epoch+1


# Funcion del entrenamiento completo para realizar las dos etapas diferentes
def complete_training(model_type,model_name,opt_name,train_dataloader,val_dataloader,lr1=0.001,lr2=0.00001,dropout=0.2,fine_tuning=False,
                      size1=1024,size2=512,size3=128,patience1=5,patience2=10,max_epochs1=25,max_epochs2=40,label_smoothing=0.01,device="cpu",callback=None):
    torch.cuda.empty_cache()
    # Llamamos a la función que crea el modelo
    if model_type == "MRConvolutional":
        model = MRConvolutionalModel(model_name,dropout,size1,size2).to(device)
    elif model_type == "MRConvolutional_Hist":
        model = MRConvolutionalModelHistogram(model_name,22,dropout,size1,size2,size3).to(device)
    elif model_type == "MRVisionTransformer":
        model = MRVisionTransformer(model_name,dropout,size1,size2).to(device)
    else:
        raise ValueError(f"El tipo de modelo {model_type} no está soportado, elija uno entre ['MRConvolutional','MRConvolutional_Hist,'MRVisionTransformer]")
    
    # Primero entrenamos solo la cabeza del modelo
    obj_loss,next_epoch = train_model(model,opt_name,train_dataloader,val_dataloader,patience1,max_epochs1,lr1,label_smoothing,
                           device,fine_tuning=False,callback=callback,start_epoch=0)

    # --- No hacer fine tuning hasta hasta haber decidido el mejor modelo con la búsqueda de hiperparámetros ---
    # Descongelamos los últimos bloques de los modelos base si queremos hacer un fine tuning adicional
    if fine_tuning:
        # Cargamos el mejor modelo obtenido de la primera parte del entrenamiento
        model.load_state_dict(torch.load(f"./{model.name}_best_model.pth",map_location=device))

        # Para las redes convolucionales
        if model_name == "ResNet50":
            layers = list(model.model.layer4.parameters())
        elif model_name in ["EfficientNetV2_small","RegNet_Y_3_2GF","ConvNeXt_tiny","ConvNeXt_small"]:
            # En estas redes hay un bloques de normalización que deberían ir descongelados junto con el último
            layers = list(model.model.features[-2:].parameters())

        # Para los vision transformers
        elif model_name == "ViT_B_16":
            # Descongelamos el último bloque de transformer y la capa de normalización final
            layers = list(model.model.encoder.layers[-1].parameters()) + list(model.model.encoder.ln.parameters())
        elif model_name == "Swin_V2_S":
            layers = list(model.model.features[-2:].parameters())
        elif model_name == "DINOv2_ViT_B":
            # Descongelamos el último bloque de transformer y la capa de normalización final
            layers = list(model.model.blocks[-1].parameters()) + list(model.model.norm.parameters())

        for param in layers:
            param.requires_grad=True
        obj_loss,_ = train_model(model,opt_name,train_dataloader,val_dataloader,patience2,max_epochs2,lr2,label_smoothing,
                               device,fine_tuning=True,callback=callback,start_epoch=next_epoch)

    del model
    torch.cuda.empty_cache()
    return obj_loss


def main():
    """
    Desde el main creamos los dataloaders y el modelo correspondiente con los mejores hiperparámetros enconctrados y realizamos el 
    entrenamiento del mismo guardando nuestro mejor modelo que será el que posteriormente exportemos para comparar su eficacia.
    """
    # Fijamos una semilla considerando que las ejecuciones han sido todas realizadas con GPU y no con CPU
    torch.manual_seed(67)
    torch.cuda.manual_seed_all(67)
    # Elegimos la gpu si está disponible y si no la cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargamos en data frames los conjuntos de datos ya divididos
    train_def = pd.read_csv("./dataset_dividido/train_def.csv")
    val = pd.read_csv("./dataset_dividido/val.csv")
    # El de test lo usaremos más tarde para comparar y no ahora para entrenar

    # Creamos los datasets y dataloaders definitivos que vamos a usar para entrenar y validar
    train_dataset_def = CustomImageDataset("./fotos_recortadas",train_def,True)
    val_dataset = CustomImageDataset("./fotos_recortadas",val,True)
    train_dataloader_def = DataLoader(train_dataset_def,batch_size=64,shuffle=True,pin_memory=True)
    val_dataloader = DataLoader(val_dataset,batch_size=128,shuffle=False,pin_memory=True)

    complete_training("MRVisionTransformer","Swin_V2_S","AdamW",train_dataloader_def,val_dataloader,lr1=5e-4,lr2=4.5e-5,dropout=0.15,fine_tuning=True,
                      size1=1024,size2=512,size3=None,patience1=10,patience2=15,max_epochs1=50,max_epochs2=60,label_smoothing=0.01,device=device)

if __name__ == "__main__":
    main()