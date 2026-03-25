# TFG
Repositorio con todo el código del TFG - Sistema de monitorización de praderas marinas mediante análisis de imágenes georreferenciadas capturadas con dispositivos móviles

## Carpetas
**- Hiperparametros_MRConvolutionalModel:** Resultados de la búsqueda de hiperparámetros utilizando optuna, para ajustar el modelo de multiregresión con base convolucional

**- Prueba_hiperparametros_recorte:** 
Resultados de la búsqueda de hiperparámetros utilizando optuna, para ajustar el modelo de multiregresión con base convolucional

**- scripts_proporciones:** 
Código relacionado con el problema de predicción de proporciones de clases

**- scripts_recortes:** 
Código relacionado con el problema de recorte de cuadrados en las imágenes

## Archivos
**- TFG.ipynb:** Notebook de Jupyter que contiene las primeras pruebas junto con el código para realizar las búsquedas de hiperparámetros, el preprocesado y análisis exploratorio de las etiquetas, así como las pruebas de los modelos sobre el conjunto de test

**- esquinas_def.csv:** Etiquetas con las coordenadas de las esquinas de los cuadrados en las imágenes para la primera parte de recorte

**- etiquetas_fotos.csv:** Etiquetas con las proporciones de las clases presentes en las imágenes

**- etiquetas_augmented.csv:**Etiquetas con las proporciones de las clases presentes en las imágenes obtenidas al aplicar data augmentation
