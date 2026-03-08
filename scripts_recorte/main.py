from crop_pipeline import crop_pipeline

def main():
    crop_pipeline("1_photos",ruta_modelo="Corners_model.pth",dir_salida="fotos_rec",revision=True)
    
if __name__ == "__main__":
    main()