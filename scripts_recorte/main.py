from crop_pipeline import crop_pipeline

def main():
    crop_pipeline("1_photos",ruta_modelo="./Modelos/Corners_EfficientNet_B3_model.pth",dir_salida="fotos_recortadas",revision=True)
    
if __name__ == "__main__":
    main()