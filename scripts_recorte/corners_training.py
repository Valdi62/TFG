import torch
import torch.nn as nn
from corners_model import TransferLearning
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from corners_dataset import CustomImageDataset
import pandas as pd
import sklearn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        for data_inputs, data_labels in val_dataloader:
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


def train_model(model,opt,train_dataloader,val_dataloader,global_min_obj=float('inf'),patience=5,max_epochs=30,coords_weights=1,corners_weights=1,learning_rate=0.001,
                device="cpu",fine_tuning=False,callback=None,start_epoch=0):
    # - Función con el bucle de entrenamiento de clasificación y regresión -

    min_objective_loss = float('inf')
    no_improvement = 0
    mejor_log=None
    last_epoch=0

    # Como función de pérdida para representar  las coordenadas usamos la MSE loss para penalizar errores grandes
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
    scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=round(2*patience/3))

    # Bucle de entrenamiento
    pbar = tqdm(range(max_epochs))
    for epoch in pbar:
        last_epoch = epoch
        # Poner lo que sea necesario del modelo en modo entrenamiento
        model.train()
        if not fine_tuning:
            model.model.eval()  
        # Pasamos el modelo base a evaluacion para que la batch_normalization de la base sea fija y el dropout se desactive si no estamos haciendo fine_tuning

        total_positives = 0
        epoch_corners_loss = 0
        epoch_coords_loss = 0
        epoch_mae_loss = 0

        for data_inputs, data_labels in train_dataloader:
            # Hacer una pasada hacia delante
            data_inputs = data_inputs.to(device)
            data_corners_labels = data_labels["classification"].to(device)
            data_coords_labels = data_labels["coordinates"].to(device)
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

            # Reiniciar los gradientes
            optimizer.zero_grad()
            # Pasada hacia atrás
            total_loss.backward()
            # Actualizar los parámetros
            optimizer.step()

        # Calcular las métricas de validación
        val_accuracy,precision,recall,val_corners_loss,val_coords_loss,val_coords_mae = validation(model,val_dataloader,device)

        # Comparamos el MAE de cada combinación porque es lo más intuitivo y fácil de interpretar
        objective_loss = val_coords_mae + (1-precision)
        # Las losses no son igual de importantes y la MAE en este caso es más importante porque es más dificil predecir coordenadas que si tiene o no esquinas
        
        # Actualizamos con la objective loss nuestro scheduler que controla cuando se reduce el learning rate
        scheduler.step(objective_loss)

        current_log=("Training corners Loss %0.4f, Validation corners Loss %0.4f, has_corners accuracy %0.2f%%, has_corners precision %0.2f%%, has_corners recall %0.2f%% \n"
                    "Patience: %d/%d, Training coords Loss %0.4f, Training MAE Loss %0.4f, Validation coords Loss %0.4f, Validation MAE loss %0.4f, Objective loss %0.4f " % 
                    (epoch_corners_loss/len(train_dataloader), val_corners_loss, val_accuracy*100, precision*100, recall*100,no_improvement, patience,
                    epoch_coords_loss/max(1,total_positives), epoch_mae_loss/max(1,total_positives), val_coords_loss, val_coords_mae, objective_loss))

        # Intentamos llamar al callback pero si no lo tenemos definido simplemente lo ignoramos
        if callback is not None:
            callback(objective_loss,start_epoch+last_epoch)
        
        if objective_loss < min_objective_loss:
            min_objective_loss=objective_loss
            no_improvement = 0
            mejor_log=current_log

            if objective_loss < global_min_obj:
                torch.save(model.state_dict(), "./Corners_model.pth")
                global_min_obj = objective_loss
        else:
            no_improvement += 1

        print(current_log)

        if no_improvement >= patience:
            print("No hay mejora por %d épocas. Parada Temprana!!" % patience)
            break
        
    print(mejor_log)
    return min_objective_loss,start_epoch+last_epoch+1


# Funcion del entrenamiento completo para realizar las dos etapas diferentes
def complete_training_crop(model_name,opt_name,train_dataloader,val_dataloader,coords_weights=1,corners_weights=1,lr1=0.001,lr2=0.00001,dropout=0.2,
                    size1=512,size2=128,patience1=10,patience2=10,max_epochs1=20,max_epochs2=40,fine_tuning=False,device="cpu",callback=None):
    torch.cuda.empty_cache()
    # Llamamos a la función que crea el modelo
    model = TransferLearning(model_name,dropout,size1,size2).to(device)
    
    # Primero entrenamos solo la cabeza del modelo
    obj_loss,next_epoch = train_model(model,opt_name,train_dataloader,val_dataloader,float('inf'),patience1,max_epochs1,coords_weights,corners_weights,
                                      lr1,device,fine_tuning=False,callback=callback,start_epoch=0)

    # Descongelamos los últimos bloques de los modelos base si queremos hacer un fine tuning adicional
    if fine_tuning:
        # Borramos el modelo de memoria para después cargar el mejor modelo hasta el momento
        del model
        torch.cuda.empty_cache()
        model = TransferLearning(model_name,dropout=0.4,size1=512,size2=128).to(device)
        model.load_state_dict(torch.load(f"./Corners_model.pth",map_location=device))

        # Para los modelos no elegidos solo ofrecemos un fine tuning descongelando el último bloque
        if model_name == "ResNet50":
            layers = list(model.model.layer4.parameters())
        elif model_name in ["ConvNeXt_tiny","EfficientNetV2_small"]:
            # En estas redes hay un bloques de normalización que deberían ir descongelados junto con el último
            layers = list(model.model.features[-2:].parameters())
        # De mobilenet descongelmos algún bloque más por ser mucho más ligera y por eso podemos permitirnoslo
        elif model_name == "MobileNet_V3_Large":
            layers = list(model.model.features[-4:].parameters())

        for param in layers:
            param.requires_grad=True
        obj_loss,_ = train_model(model,opt_name,train_dataloader,val_dataloader,obj_loss,patience2,max_epochs2,coords_weights,corners_weights,
                                      lr2,device,fine_tuning=True,callback=callback,start_epoch=next_epoch)

    del model
    torch.cuda.empty_cache()
    return obj_loss


def main():
    """
    Desde el main creamos los dataloaders y el modelo correspondiente con los mejores hiperparámetros enconctrados y realizamos el 
    entrenamiento del mismo guardando nuestro mejor modelo que será el que posteriormente exportemos y utilicemos para realizar la 
    clasificación y el recorte de todas las imágenes.
    """
    pass
    
if __name__ == "__main__":
    main()