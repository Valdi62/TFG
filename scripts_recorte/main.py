from crop_pipeline import crop_pipeline

def main():
    # Se realiza la clasificación y recorte de las imágenes contenidas en el directorio de entrada usando el modelo entrenado
    # Posteriormente se almacenan las imágenes recortadas en un directorio de salida
    crop_pipeline("4_new_dataset",ruta_modelo="./Modelos/Corners_EfficientNet_B3_model.pth",dir_salida="recortes",revision=True)
    
if __name__ == "__main__":
    main()