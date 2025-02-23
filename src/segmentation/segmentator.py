import subprocess

def segmentate_image(image_path, heatmap_path, output_path, k, use8Way, euclidif, adj, minComponentSize, buildingBlockTreshold):
    # 1. Crear el archivo de configuración.
    config_path = "segmentation/config.txt"
    with open(config_path, "w") as config_file:
        # Puedes ajustar el formato de salida según lo que espere tu función readConfig.
        config_file.write(f"{k}\n")
        config_file.write(f"{use8Way}\n")
        config_file.write(f"{euclidif}\n")
        config_file.write(f"{adj}\n")
        config_file.write(f"{minComponentSize}\n")
        config_file.write(f"{buildingBlockTreshold}\n")
    
    # 2. Compilar el archivo C++.
    compile_cmd = ["g++", "segmentation/main.cpp", "-o", "segmentation/main.exe"]
    subprocess.run(compile_cmd, check=True)
    
    # 3. Ejecutar el .exe y capturar la salida.
    # Nota: en Windows podrías necesitar llamar a "segmentation/main.exe" sin "./".
    run_cmd = ["./segmentation/main.exe"]
    process = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Mostrar la salida estándar en tiempo real.
    for line in process.stdout:
        print(line, end="")  # 'end=""' evita agregar saltos de línea extra.
    
    # También puedes mostrar la salida de error, si la hubiera.
    for error_line in process.stderr:
        print("Error:", error_line, end="")
    
    # Esperar a que el proceso finalice.
    process.wait()
