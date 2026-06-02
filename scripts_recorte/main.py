from crop_pipeline import crop_pipeline

def main():
    crop_pipeline("4_new_dataset",ruta_modelo="./Modelos/Corners_EfficientNet_B3_model.pth",dir_salida="nuevo_dataset_recortado",revision=True)
    
if __name__ == "__main__":
    main()