# TFG
Repositorio con todo el código del TFG - Sistema de monitorización de praderas marinas mediante análisis de imágenes georreferenciadas capturadas con dispositivos móviles

## Carpetas
**- Hiperparametros_MR:** Resultados de la búsqueda de hiperparámetros utilizando optuna, para ajustar el modelo de multiregresión composicional con base convolucional

**- Hiperparametros_recorte:** 
Resultados de la búsqueda de hiperparámetros utilizando optuna, para ajustar el modelo multicabeza de recorte de imágenes con base convolucional

**- dataset_dividido:** 
Carpeta con cada uno de los datasets ya divididos para utilizar durante el entrenamiento y la validación de los modelos

**- scripts_proporciones:** 
Código relacionado con el problema de estimación de proporciones por clase

**- scripts_recortes:** 
Código relacionado con el problema de recorte de los cuadrados en las imágenes

## Archivos
**- Resultados.xlsx:** Archivo excel que contiene el resumen de los resultados obtenidos al aplicar el segundo modelo, de estimación de proporciones, a los conjuntos de Test

**- TFG.ipynb:** Notebook de Jupyter que contiene el código para realizar las búsquedas de hiperparámetros, el preprocesado y análisis exploratorio de las etiquetas, así como las pruebas de los modelos sobre el conjunto de test y ejemplos individuales

**- esquinas.csv:** Etiquetas con las coordenadas de las esquinas de los cuadrados en las imágenes para la primera tarea de recorte

**- etiquetas_fotos.csv:** Etiquetas con las proporciones de las clases en las imágenes para la segunda tarea de estimación de proporciones

**- etiquetas_nuevas_fotos.csv:** Etiquetas con las proporciones de las clases en las imágenes del segundo conjunto de Test para la segunda tarea de estimación de proporciones
