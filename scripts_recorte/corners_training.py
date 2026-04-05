import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import plotly.express as px
from . corners_model import TransferLearning
from . corners_dataset import CustomImageDataset


# Función para generar las gráficas del entrenamiento
def save_graph(history,module,model,fine_tuning=0):
    # Convertimos a data frame para utilizar seaborn a la hora de crear gráficas
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
def validation(model,val_dataloader,device="cpu"):
    # - Función con el bucle de validación -

    # Como función de pérdida para representar  las coordenadas usamos la MSE loss para penalizar errores grandes
    loss_module_coords = nn.MSELoss()
    # Siempre vamos a comprobar el MAE de las coordenadas en training
    mae_loss = nn.L1Loss()
    # Usaremos el BCE con logit loss para comprobar si tiene o no esquinas
    loss_module_corners = nn.BCEWithLogitsLoss()
    
    val_coords_mae=0
    val_corners_loss=0
    val_coords_loss=0
    correct_predictions=0
    total_predictions=0
    total_positives=0
    corner_trues=[]
    corner_preds=[]

    with torch.no_grad():
        model.eval()
        for data_inputs,data_labels in val_dataloader:
            corner_trues.extend(data_labels["classification"].numpy())
            data_inputs = data_inputs.to(device)
            data_corners_labels = data_labels["classification"].to(device)
            data_coords_labels = data_labels["coordinates"].to(device)

            logits = model(data_inputs)
            corners_logits = logits["classification"].squeeze(1)
            coords_logits = logits["coordinates"]

            # Con esto calcularemos la accuracy para el conjunto de validacion
            predicted_classes = (corners_logits > 0).int()
            corner_preds.extend(predicted_classes.detach().cpu().numpy())
            correct_predictions += (predicted_classes == data_corners_labels).sum().item()
            total_predictions += data_corners_labels.size(0)

            # Solo nos interesan las coordenadas si tiene esquinas
            has_corners_mask = data_corners_labels.bool()
            positives_in_batch = has_corners_mask.sum().item()
            total_positives += positives_in_batch

            # Si hay alguna imagen con esquinas
            if positives_in_batch>0:
                # Filtrar predicciones y targets
                reg_pred_corners = coords_logits[has_corners_mask]
                reg_true_corners = data_coords_labels[has_corners_mask]

                # Calcular la pérdida y MAE SOLO en los ejemplos positivos
                # Primero multiplicamos las losses por el número de datos con esquinas para luego que sea proporcional al número de imagenes positivas y no al número de batches
                val_coords_loss += loss_module_coords(reg_pred_corners, reg_true_corners).item()*positives_in_batch
                val_coords_mae += mae_loss(reg_pred_corners, reg_true_corners).item()*positives_in_batch
            val_corners_loss += loss_module_corners(corners_logits, data_corners_labels).item()

        # La clase positiva va a ser "Tiene esquinas"
        precision = precision_score(corner_trues,corner_preds,pos_label=1,zero_division=0)
        recall = recall_score(corner_trues,corner_preds,pos_label=1,zero_division=0)
        accuracy = correct_predictions / total_predictions
        return (accuracy,precision,recall,val_corners_loss/len(val_dataloader),val_coords_loss/max(1,total_positives),val_coords_mae/max(1,total_positives))


def train_model(model,opt,train_dataloader,val_dataloader,patience=5,max_epochs=30,coords_weights=1,corners_weights=1,learning_rate=0.001,
                device="cpu",fine_tuning=False,callback=None,start_epoch=0,warmup=5):
    # - Función con el bucle de entrenamiento de clasificación y regresión -

    min_objective_loss = float('inf')
    no_improvement = 0
    mejor_log=None
    last_epoch=0

    global history_mae,history_mse,history_bce,history_corners
    if start_epoch==0:
        history_mae = {"train_mae":[],"val_mae":[]}
        history_mse = {"train_mse":[],"val_mse":[]}
        history_bce = {"train_bce":[],"val_bce":[]}
        history_corners = {"accuracy":[],"precision":[],"recall":[]}

    # Como función de pérdida para representar  las coordenadas usamos la MSE loss para penalizar errores grandes y porque tras el etiquetado manual de los datos
    #  no existen muchos outliers
    loss_module_coords = nn.MSELoss()
    # Siempre vamos a comprobar el MAE de las coordenadas en training
    # Como la mae solo la vamos a mostrar podemos calcular la suma de todos los ejemplos solo de la clase positiva y luego dividir directamente entre el total de estos
    mae_loss = nn.L1Loss()
    # Usaremos el BCE con logit loss para comprobar si tiene o no esquinas
    loss_module_corners = nn.BCEWithLogitsLoss()

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
    scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=round(2*patience/3),min_lr=1e-6)

    # Bucle de entrenamiento
    pbar = tqdm(range(max_epochs))
    for epoch in pbar:
        last_epoch = epoch

        # Poner lo que sea necesario del modelo en modo entrenamiento
        model.train()
        # Pasamos el modelo base a evaluacion para que la batch_normalization de la base sea fija
        if fine_tuning:
            # Comprobamos cada modulo de la base para dejar aquellos que esten descongelados en train, pero los que sigan congelados se pongan en eval
            # Adicionalmente el batch normalization de la base lo vamos a mantener siempre congelado para que no se modifique y se estropee
            for module in model.model.modules():
                if not any(i.requires_grad for i in module.parameters()):
                    module.eval()
        else:
            model.model.eval()

        total_positives = 0
        epoch_corners_loss = 0
        epoch_coords_loss = 0
        epoch_mae_loss = 0
        epoch_total_loss = 0

        for data_inputs,data_labels in train_dataloader:
            # Hacer una pasada hacia delante
            data_inputs = data_inputs.to(device)
            data_corners_labels = data_labels["classification"].to(device)
            data_coords_labels = data_labels["coordinates"].to(device)

            # Reiniciar los gradientes
            optimizer.zero_grad()
            # Realizamos la pasada hacia delante
            preds = model(data_inputs)

            # Calcular el valor de la función de pérdida para ver si tiene esquinas
            corners_logits = preds["classification"].squeeze(1)
            loss_corners = loss_module_corners(corners_logits, data_corners_labels)
            # Acumular el error
            epoch_corners_loss += loss_corners.item()

            # Solo me interesa el valor de las coordenadas si tiene esquinas
            has_corners_mask = data_corners_labels.bool()
            positives_in_batch = has_corners_mask.sum().item()
            total_positives +=positives_in_batch
            if positives_in_batch>0:
                reg_pred_corners = preds["coordinates"][has_corners_mask]
                reg_true_corners = data_coords_labels[has_corners_mask]
                
                loss_coords = loss_module_coords(reg_pred_corners, reg_true_corners)
                # Multiplicamos por el número de fotos con esquinas para luego hacer que sea proporcional al número de ejemplos positivos
                epoch_coords_loss += loss_coords.item()*positives_in_batch
                epoch_mae_loss += mae_loss(reg_pred_corners, reg_true_corners).item()*positives_in_batch
                total_loss = corners_weights * loss_corners + coords_weights * loss_coords
            else:
                total_loss = loss_corners
            epoch_total_loss += total_loss.item()
            
            # Pasada hacia atrás
            total_loss.backward()

            # Actualizar los parámetros
            optimizer.step()

        # Calcular las métricas de validación
        accuracy,precision,recall,val_corners_loss,val_coords_loss,val_coords_mae = validation(model,val_dataloader,device)

        # Comparamos el MAE de cada combinación porque es lo más intuitivo y fácil de interpretar
        # Le damos más peso a la regresión porque es la tarea más complicada y precisa
        objective_loss = 3*val_coords_mae + val_corners_loss

        # Caculamos las pérdidas de entrenamiento promedio para esta etapa
        mean_epoch_coords_loss = epoch_coords_loss/max(1,total_positives)
        mean_epoch_mae = epoch_mae_loss/max(1,total_positives)
        mean_epoch_corner_loss = epoch_corners_loss/len(train_dataloader)

        # Almacenamos los valores de las pérdidas para visualizarlos posteriormente
        history_mae["train_mae"].append(mean_epoch_mae)
        history_mae["val_mae"].append(val_coords_mae)

        history_mse["train_mse"].append(mean_epoch_coords_loss)
        history_mse["val_mse"].append(val_coords_loss)

        history_bce["train_bce"].append(mean_epoch_corner_loss)
        history_bce["val_bce"].append(val_corners_loss)

        history_corners["accuracy"].append(accuracy)
        history_corners["precision"].append(precision)
        history_corners["recall"].append(recall)

        # Actualizamos con la objective loss nuestro scheduler que controla cuando se reduce el learning rate
        scheduler.step(objective_loss)

        current_log=("\nTraining corners Loss %0.4f, Validation corners Loss %0.4f, has_corners accuracy %0.2f%%, has_corners precision %0.2f%%, has_corners recall %0.2f%% \n"
                    "Patience: %d/%d, Training coords Loss %0.4f, Training MAE Loss %0.4f, Validation coords Loss %0.4f, Validation MAE loss %0.4f, Total Train Loss %0.4f, Objective loss %0.4f" % 
                    (mean_epoch_corner_loss, val_corners_loss, accuracy*100, precision*100, recall*100,no_improvement, patience,
                    mean_epoch_coords_loss, mean_epoch_mae, val_coords_loss,val_coords_mae,epoch_total_loss/len(train_dataloader),objective_loss))

        # Intentamos llamar al callback pero si no lo tenemos definido simplemente lo ignoramos
        if callback is not None:
            callback(objective_loss,start_epoch+last_epoch)
        
        if objective_loss < min_objective_loss:
            min_objective_loss=objective_loss
            no_improvement = 0
            mejor_log=current_log
            torch.save(model.state_dict(),f"./{model.name}_model.pth")
        else:
            if epoch >= warmup:
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
    history_mse["epoch"] = x
    history_bce["epoch"] = x
    history_corners["epoch"] = x

    save_graph(history_mae,"mae",model.name,start_epoch+0.5)
    save_graph(history_mse,"mse",model.name,start_epoch+0.5)
    save_graph(history_bce,"bce",model.name,start_epoch+0.5)
    save_graph(history_corners,"has_corners",model.name,start_epoch+0.5)
    return min_objective_loss,start_epoch+last_epoch+1


# Funcion del entrenamiento completo para realizar las dos etapas diferentes
def complete_training_crop(model_name,opt_name,train_dataloader,val_dataloader,coords_weights=1,corners_weights=1,lr1=1e-3,lr2=1e-5,dropout=0.2,
                    size1=512,size2=128,patience1=10,patience2=10,max_epochs1=20,max_epochs2=40,fine_tuning=False,device="cpu",callback=None):
    torch.cuda.empty_cache()
    
    # Llamamos a la función que crea el modelo
    model = TransferLearning(model_name,dropout,size1,size2).to(device)

    # Primero entrenamos solo la cabeza del modelo
    obj_loss,next_epoch = train_model(model,opt_name,train_dataloader,val_dataloader,patience1,max_epochs1,coords_weights,corners_weights,
                                      lr1,device,fine_tuning=False,callback=callback,start_epoch=0)

    # Descongelamos los últimos bloques de los modelos base si queremos hacer un fine tuning adicional
    if fine_tuning:
        # Cargamos el mejor modelo de la primera parte del entrenamiento
        model.load_state_dict(torch.load(f"./{model.name}_model.pth",map_location=device))

        # Para los modelos no elegidos solo ofrecemos un fine tuning descongelando el último bloque
        if model_name == "ResNet50":
            layers = list(model.model.layer4.parameters())
        elif model_name in ["ConvNeXt_tiny","MobileNet_V3_Large"]:
            # En estas redes hay un bloques de normalización que deberían ir descongelados junto con el último convolucional
            layers = list(model.model.features[-2:].parameters())
        # Como este el modelo ganador de la busqueda de hiperparametros vamos a descongelar alguna capa adicional para comprobar si se obtiene mejor rendimiento  
        elif model_name == "EfficientNet_B3":
            layers = list(model.model.features[-3:].parameters())

        for param in layers:
            param.requires_grad=True
        obj_loss,_ = train_model(model,opt_name,train_dataloader,val_dataloader,patience2,max_epochs2,coords_weights,corners_weights,
                                      lr2,device,fine_tuning=True,callback=callback,start_epoch=next_epoch)

    del model
    torch.cuda.empty_cache()
    return obj_loss


"""
Después de realizar multiples pruebas relacionadas con los últimos bloques del modelo base extraemos conlcuiones.
En el conjunto de Test
Primera prueba - modelo base congelado y solo entrenamos la cabeza
Test Corners BCE 0.1753, Test MSE 0.0031, Test MAE 0.0424
Accuracy 98.88%, Precision 98.82%, Recall 100.00%

Segunda prueba - primera fase con el modelo base congelado y una segunda fase donde se descongela las últimas capas del modelo base
Test Corners BCE 0.1311, Test MSE 0.0032, Test MAE 0.0431
Accuracy 98.88%, Precision 98.82%, Recall 100.00%

Tercera prueba - directamente entrenar una fase con la cabeza y las últimas capas del modelo base descongeladas y el resto congelado
Test Corners BCE 0.1605, Test MSE 0.0008, Test MAE 0.0175
Accuracy 98.88%, Precision 98.82%, Recall 100.00%

Parece que empezar con esas capas ya descongeladas y entrenar directamente el modelo completo es mejor que dividirlo en dos fases,
aunque la bce sea un poco peor de esta forma, no parece impactar tanto a las medidas de precisión y la perdida de regresión es mucho mejor.
"""

def main():
    """
    Desde el main creamos los dataloaders y el modelo correspondiente con los mejores hiperparámetros enconctrados y realizamos el 
    entrenamiento del mismo guardando nuestro mejor modelo que será el que posteriormente exportemos y utilicemos para realizar la 
    clasificación y el recorte de todas las imágenes.
    """
    # Fijamos una semilla considerando que las ejecuciones han sido todas realizadas con GPU y no con CPU
    torch.manual_seed(67)
    torch.cuda.manual_seed_all(67)

    # Elegimos la gpu si está disponible y si no la cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargamos en data frames los conjuntos de datos ya divididos
    train = pd.read_csv("./dataset_dividido/train_crop.csv")
    val = pd.read_csv("./dataset_dividido/val_crop.csv")

    # Creamos los datasets y dataloaders que vamos a usar para entrenar y validar
    train_dataset = CustomImageDataset("./1_photos",train,True)
    val_dataset = CustomImageDataset("./1_photos",val,True)
    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True,pin_memory=True,num_workers=2)
    val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=False,pin_memory=True,num_workers=2)

    # Definimos el modelo ganador con los valores que mejor funcionan
    model = TransferLearning("EfficientNet_B3",0.4,896,384).to(device)
    # Descongelamos sus últimas capas
    layers = list(model.model.features[-3:].parameters())
    for param in layers:
        param.requires_grad=True
    
    # Realizamos el entrenamiento del modelo primero con el conjunto de train y val para comprobar su rendimiento en test
    train_model(model,"AdamW",train_dataloader,val_dataloader,patience=30,max_epochs=120,coords_weights=10,corners_weights=1,
                learning_rate=1e-3,device=device,fine_tuning=True)
    

if __name__ == "__main__":
    main()

"""
Evaluación final

Test Corners BCE 0.1605, Test MSE 0.0008, Test MAE 0.0175
Accuracy 98.88%, Precision 98.82%, Recall 100.00%
"""