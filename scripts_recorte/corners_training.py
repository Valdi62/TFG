import torch
import torch.nn as nn
from corners_model import TransferLearning
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from corners_dataset import CustomImageDataset
import pandas as pd
import sklearn
from torch.utils.data import DataLoader


# Bucles de validación
def validation(model, loss_module_corners,loss_module_coords, val_dataloader,device="cpu"):
    # - Función con el bucle de validación -

    # Siempre vamos a comprobar el MAE de las coordenadas en validación
    mae_loss=nn.L1Loss()
    
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


def train_model(model,opt,train_dataloader,val_dataloader,loss_module_coords,patience=5,max_epochs=30,coords_weights=1,corners_weights=1,learning_rate=0.001,
                device="cpu",fine_tuning=True, callback=None,start_epoch=0):
    # - Función con el bucle de entrenamiento de clasificación y regresión -

    min_objective_loss = float('inf')
    no_improvement = 0
    mejor_log=None
    last_epoch=0

    # Siempre vamos a comprobar el MAE de las coordenadas en training
    # Como la mae solo la vamos a mostrar podemos calcular la suma de todos los ejemplos solo de la clase positiva y luego dividir directamente entre el total de estos
    mae_loss=nn.L1Loss()

    # Siempre usaremos el BCE con logit loss para comprobar si tiene o no esquinas
    loss_module_corners = nn.BCEWithLogitsLoss()

    # Vamos a filtrar los parámetros que no están congelados para solo pasarle los que sean estrictamente necesarios al optimizador
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Elegimos el optimizador deseado con su learning rate
    if opt == "SGD":
        optimizer = torch.optim.SGD(trainable_params, lr=learning_rate)
    elif opt == "AdamW":
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    elif opt == "RMSprop":
        optimizer = torch.optim.RMSprop(trainable_params, lr=learning_rate)
    elif opt == "Adam":
        # Por defecto usaremos el optimizador Adam
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    else:
        raise ValueError(f"El optimizado {opt} no está soportado, elija uno entre ['SGD','AdamW','Adam','RMSprop']")

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
        val_accuracy,precision,recall,val_corners_loss,val_coords_loss,val_coords_mae = validation(model, loss_module_corners, loss_module_coords, val_dataloader,device)

        # Comparamos el MAE de cada combinación porque es lo más intuitivo y fácil de interpretar
        objective_loss = val_coords_mae + (1-precision)
        # Las losses no son igual de importantes y la MAE en este caso es más importante porque es más dificil predecir coordenadas que si tiene o no esquinas
        
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
            torch.save(model.state_dict(), "./Corners_model.pth")
            mejor_log=current_log
        else:
            no_improvement += 1

        print(current_log)

        if no_improvement >= patience:
            print("No hay mejora por %d épocas. Parada Temprana!!" % patience)
            break
        
    print(mejor_log)
    return min_objective_loss,start_epoch+last_epoch+1


def classification_train_model(model,opt,train_dataloader,val_dataloader,patience=5,max_epochs=30,learning_rate=0.001,
                                device="cpu"):
    # - Función con el bucle de entrenamiento de clasificación inicial - (en el entrenamiento del modelo definitivo no se usa)

    min_objective_loss = float('inf')
    no_improvement = 0
    mejor_log = None

    # Siempre usaremos el BCE con logit loss para comprobar si tiene o no esquinas
    loss_module_corners = nn.BCEWithLogitsLoss()

    # Vamos a filtrar los parámetros que no están congelados para solo pasarle los que sean estrictamente necesarios al optimizador
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Elegimos el optimizador deseado con su learning rate
    if opt == "SGD":
        optimizer = torch.optim.SGD(trainable_params, lr=learning_rate)
    elif opt == "AdamW":
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    elif opt == "RMSprop":
        optimizer = torch.optim.RMSprop(trainable_params, lr=learning_rate)
    elif opt == "Adam":
        # Por defecto usaremos el optimizador Adam
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    else:
        raise ValueError(f"El optimizado {opt} no está soportado, elija uno entre ['SGD','AdamW','Adam','RMSprop']")

    # Bucle de entrenamiento
    pbar = tqdm(range(max_epochs))
    for epoch in pbar:
        # Poner lo que sea necesario del modelo en modo entrenamiento
        model.train()
        model.model.eval() 
        # Pasamos el modelo base a modo evaluacion para que la batch_normalization de la base sea fija y el dropout se desactive

        epoch_corners_loss = 0

        for data_inputs, data_labels in train_dataloader:
            # Hacer una pasada hacia delante
            data_inputs = data_inputs.to(device)
            data_corners_labels = data_labels["classification"].to(device)
            preds = model(data_inputs)

            # Calcular el valor de la función de pérdida para ver si tiene esquinas
            corners_logits = preds["classification"].squeeze(1)
            loss_corners = loss_module_corners(corners_logits, data_corners_labels)
            # Acumular el error
            epoch_corners_loss += loss_corners.item()

            # Reiniciar los gradientes
            optimizer.zero_grad()
            # Pasada hacia atrás
            loss_corners.backward()
            # Actualizar los parámetros
            optimizer.step()

        # Como queremos reutilizar el codigo de validación le pasaremos una función de pérdida de coordenadas aunque no nos fijaremos en ella
        val_accuracy,precision,recall,val_corners_loss,_,_ = validation(model, loss_module_corners, nn.L1Loss(), val_dataloader,device)

        # Para decidir cuando parar
        objective_loss = val_corners_loss
        # Las losses no son igual de importantes y la MAE en este caso es más importante porque es más dificil predecir coordenadas que si tiene o no esquinas
        
        current_log=("Training corners Loss %0.4f, Validation corners Loss %0.4f, has_corners accuracy %0.2f%%, has_corners precision %0.2f%%, has_corners recall %0.2f%% \n"
                    "Patience: %d/%d" % (epoch_corners_loss/len(train_dataloader), val_corners_loss, val_accuracy*100, precision*100, recall*100,no_improvement, patience))
        
        if objective_loss < min_objective_loss:
            min_objective_loss=objective_loss

            no_improvement = 0
            torch.save(model.state_dict(), "./Corners_model.pth")
            mejor_log=current_log
        else:
            no_improvement += 1
        print(current_log)

        if no_improvement >= patience:
            print("No hay mejora por %d épocas. Parada Temprana!!" % patience)
            break
    print(mejor_log)


# Funcion del entrenamiento completo para realizar las dos etapas diferentes
def complete_training(model_name,opt_name,train_dataloader,val_dataloader,loss_module_coords,coords_weights=1,corners_weights=1,lr1=0.001,lr2=0.001,lr3=0.00001,dropout=0.4,bloques=1,
                    size1=512,size2=128,patience1=5,patience2=10,patience3=10,max_epochs1=10,max_epochs2=20,max_epochs3=40,device="cpu",callback=None,multietapa=True):
    # Llamamos a la función que crea el modelo
    model = TransferLearning(model_name,dropout,size1,size2).to(device)
     
    # Primero vamos a entrenar solo la clasificación de los estados por lo que congelamos la cabeza de regresión
    if multietapa:
        for param in model.reg.parameters():
            param.requires_grad = False 
        classification_train_model(model,opt_name,train_dataloader,val_dataloader,patience1,max_epochs1,lr1,device)
        # Cargamos el mejor modelo de los obtenidos después de la primera etapa
        model.load_state_dict(torch.load("./Corners_model.pth", map_location=device))
        # Cuando acabamos el entrenamiento de clasificación desongelamos de nuevo los pesos de la regresión
        for param in model.reg.parameters():
            param.requires_grad = True 

    # Después entrenamos regresión con cierto ajuste de clasificación solo la cabeza del modelo
    obj_loss,next_epoch=train_model(model,opt_name,train_dataloader,val_dataloader,loss_module_coords,patience2,max_epochs2,coords_weights,corners_weights,lr2,device,
                           fine_tuning=False,callback=callback,start_epoch=0)

    # Descongelamos los últimos bloques de los modelos base para el fine tuning final de nuestro entrenamiento
    if bloques!=0:
        # Cargamos el mejor modelo de los obtenidos
        model.load_state_dict(torch.load("./Corners_model.pth", map_location=device))

        if model_name in ["EfficientNetB3","MobileNetV2"]:
            layers = list(model.model.features[-bloques:].parameters())
        elif model_name=="ResNet34":
            layers = list(model.model.layer4[-bloques:].parameters())

        for param in layers:
            param.requires_grad=True
        obj_loss,_=train_model(model,opt_name,train_dataloader,val_dataloader,loss_module_coords,patience3,max_epochs3,coords_weights,corners_weights,lr3,device,
                            fine_tuning=True,callback=callback,start_epoch=next_epoch)

    del model
    torch.cuda.empty_cache()
    
    return obj_loss


def main():
    """
    Desde el main creamos los dataloaders y el modelo correspondiente con los mejores hiperparámetros enconctrados y realizamos el 
    entrenamiento del mismo guardando nuestro mejor modelo que será el que posteriormente exportemos y utilicemos para realizar la 
    clasificación y el recorte de todas las imágenes.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(67)

    res = pd.read_csv("esquinas_def.csv")

    train,val=sklearn.model_selection.train_test_split(res, test_size=0.15,stratify=res["has_corners"],random_state = 67)

    train_dataset=CustomImageDataset("1_photos",train,True)
    val_dataset=CustomImageDataset("1_photos",val,True)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

    model = TransferLearning("EfficientNetB3",0.01,550,150).to(device)
    loss_module_coords = nn.SmoothL1Loss(beta=0.1)
    obj_loss,_ = train_model(model,"Adam",train_dataloader,val_dataloader,loss_module_coords,50,200,0.9,0.1,9e-4,device)
    print(obj_loss)
    
if __name__ == "__main__":
    main()