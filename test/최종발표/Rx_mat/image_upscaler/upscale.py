from cpu_upscaler import CPUImageUpscaler

upscaler = CPUImageUpscaler()
success, message = upscaler.upscale_image(
    input_path="Lena_restored_kmeans_a.png",
    output_path="upscaled_Lena.png",
    scale=4
)