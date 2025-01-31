import subprocess

def segmentate_image(image_path, heatmap_path, output_path, k, use8Way, euclidif, adj, minComponentSize, buildingBlockTreshold):
    # Compile the C++ file
    compile_cmd = ["g++", "main.cpp", "-o", "main.exe"]
    subprocess.run(compile_cmd, check=True)

    # Run the executable and stream output
    run_cmd = ["./main.exe"]  # Use "main.exe" on Windows
    process = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read and print output line by line
    for line in process.stdout:
        print(line, end="")  # end="" prevents double newlines