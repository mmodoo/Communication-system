from noise_removal import EnhancedImageProcessor

def main():
    processor = EnhancedImageProcessor()
    success, message = processor.process_image(
        input_path="Lena_restored_kmeans_a.png",
        output_path="upscaled_Lena.png",
        scale=4
    )
    print(message)

if __name__ == "__main__":
    main()
