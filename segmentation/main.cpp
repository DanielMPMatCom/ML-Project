#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <stack>
#include <algorithm>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Color structure
struct Color {
    unsigned char r, g, b;
};

// Global configuration
struct Config {
    double k;
    bool use8Way;
    bool euclidif;
    bool adj;
    int minComponentSize;
    double buildingBlockTreshold;
};

// Function to calculate color difference
double colorDifference(const Color& c1, const Color& c2, bool euclidif) {
    if (euclidif) {
        return std::sqrt(std::pow(c1.r - c2.r, 2) +
                         std::pow(c1.g - c2.g, 2) +
                         std::pow(c1.b - c2.b, 2));
    } else {
        return std::abs(c1.r - c2.r) +
               std::abs(c1.g - c2.g) +
               std::abs(c1.b - c2.b);
    }
}

int currentComponentXmin, currentComponentXmax, currentComponentYmin, currentComponentYmax, currentComponentSize;

// Modified Flood Fill Algorithm
void floodFillIterative(std::vector<Color>& image, int startX, int startY, int width, int height,
                        std::vector<bool>& visited, const Color& startColor, double k,
                        bool use8Way, bool adj, bool euclidif, const Color& newColor, std::vector<bool>& bigMask) {
    std::stack<std::tuple<int, int, Color>> stack;
    stack.push(std::make_tuple(startX, startY, startColor));

    while (!stack.empty()) {
        int x = std::get<0>(stack.top());
        int y = std::get<1>(stack.top());
        Color neighborColor = std::get<2>(stack.top());
        stack.pop();

        if (x < 0 || x >= width || y < 0 || y >= height || visited[y * width + x]) {
            continue;
        }

        currentComponentXmin = std::min(currentComponentXmin, x);
        currentComponentXmax = std::max(currentComponentXmax, x);
        currentComponentYmin = std::min(currentComponentYmin, y);
        currentComponentYmax = std::max(currentComponentYmax, y);
        currentComponentSize++;

        visited[y * width + x] = true;
        bigMask[y * width + x] = true;

        Color currentColor = image[y * width + x];
        Color compareColor = adj ? neighborColor : startColor; // Use neighbor's color if adj is true

        if (colorDifference(currentColor, compareColor, euclidif) <= k) {
            image[y * width + x] = newColor;

            stack.push(std::make_tuple(x + 1, y, currentColor));
            stack.push(std::make_tuple(x - 1, y, currentColor));
            stack.push(std::make_tuple(x, y + 1, currentColor));
            stack.push(std::make_tuple(x, y - 1, currentColor));

            if (use8Way) {
                stack.push(std::make_tuple(x + 1, y + 1, currentColor));
                stack.push(std::make_tuple(x + 1, y - 1, currentColor));
                stack.push(std::make_tuple(x - 1, y + 1, currentColor));
                stack.push(std::make_tuple(x - 1, y - 1, currentColor));
            }
        }
    }
}

// Function to read configuration
Config readConfig(const std::string& configFile) {
    Config config;
    std::ifstream file(configFile);
    if (!file) {
        throw std::runtime_error("Could not open config file.");
    }

    file >> config.k >> config.use8Way >> config.euclidif >> config.adj >> config.minComponentSize >> config.buildingBlockTreshold;
    file.close();
    return config;
}

// Function to get files in a directory
std::vector<std::string> getFiles(const std::string& directory) {
    std::vector<std::string> files;
    DIR* dir = opendir(directory.c_str());
    if (!dir) {
        throw std::runtime_error("Could not open directory: " + directory);
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name != "." && name != "..") {
            files.push_back(directory + "/" + name);
        }
    }
    closedir(dir);
    return files;
}

// Function to create a directory
bool createDirectory(const std::string& dir) {
    #ifdef _WIN32
    return mkdir(dir.c_str()) == 0 || errno == EEXIST; // Windows
    #else
    return mkdir(dir.c_str(), 0777) == 0 || errno == EEXIST; // POSIX
    #endif
}

// Function to save the segmentation image
void saveSegmentation(const std::vector<Color>& image, int width, int height, const std::string& outputPath) {
    unsigned char* outputData = new unsigned char[width * height * 3];
    for (int j = 0; j < width * height; ++j) {
        outputData[j * 3] = image[j].r;
        outputData[j * 3 + 1] = image[j].g;
        outputData[j * 3 + 2] = image[j].b;
    }
    stbi_write_jpg(outputPath.c_str(), width, height, 3, outputData, 100);
    delete[] outputData;
}

// Function to save a mask for a connected component
void saveMask(const std::vector<bool>& mask, int width, int height, const std::string& filePath) {
    std::vector<unsigned char> maskImage(width * height * 3, 255);
    for (int i = 0; i < width * height; ++i) {
        if (mask[i]) {
            maskImage[i * 3] = 0;     // Black pixel for component
            maskImage[i * 3 + 1] = 0;
            maskImage[i * 3 + 2] = 0;
        }
    }
    stbi_write_jpg(filePath.c_str(), width, height, 3, maskImage.data(), 100);
}

void processImages(const std::string& inputDir, const std::string& outputDir, const Config& config) {
    std::vector<std::string> files = getFiles(inputDir);
    std::cout << "Number of files in the directory: " << files.size() << "\n";

    int i = -1;
    for (const auto& filePath : files) {
        std::cout << filePath <<"\n";
        // Check if the file is an image based on its extension
        if (filePath.size() >= 4 && filePath.substr(filePath.size() - 4) == ".hmp") continue;
        i++;

        // Load image data
        int width, height, channels;
        unsigned char* imgData = stbi_load(filePath.c_str(), &width, &height, &channels, 3);
        if (!imgData) {
            std::cerr << "Failed to load image: " << filePath << "\n";
            continue;
        }

        std::cout << "Processing image " << i + 1 << ": " << filePath
                  << " (Width: " << width << ", Height: " << height << ", Channels: " << channels << ")\n";

        // Derive the heatmap file path
        std::string heatmapPath = filePath.substr(0, filePath.size() - 4) + ".hmp";

        // Open and read the heatmap file
        std::ifstream heatmapFile(heatmapPath, std::ios::binary);
        if (!heatmapFile) {
            std::cerr << "Failed to load heatmap file: " << heatmapPath << "\n";
            stbi_image_free(imgData);
            continue;
        }

        std::vector<float> heatmap(width * height);
        heatmapFile.read(reinterpret_cast<char*>(heatmap.data()), width * height * sizeof(float));
        if (!heatmapFile) {
            std::cerr << "Error reading heatmap data from file: " << heatmapPath << "\n";
            stbi_image_free(imgData);
            continue;
        }
        heatmapFile.close();

        std::vector<Color> image(width * height);
        std::vector<Color> buildingBlocksImage(width * height);
        for (int j = 0; j < width * height; ++j) {
            image[j] = {imgData[j * 3], imgData[j * 3 + 1], imgData[j * 3 + 2]};
            buildingBlocksImage[j] = {255, 255, 255};
        }
        stbi_image_free(imgData);

        std::ostringstream folderPath;
        folderPath << outputDir << "/" << std::setw(3) << std::setfill('0') << (i + 1);
        if (!createDirectory(folderPath.str())) {
            std::cerr << "Failed to create directory: " << folderPath.str() << "\n";
            continue;
        }
        std::ostringstream buildingBlocksFolderPath;
        buildingBlocksFolderPath << folderPath.str() << "/building_blocks";
        if (!createDirectory(buildingBlocksFolderPath.str())) {
            std::cerr << "Failed to create directory: " << buildingBlocksFolderPath.str() << "\n";
            continue;
        }
        std::ostringstream nonBuildingBlocksFolderPath;
        nonBuildingBlocksFolderPath << folderPath.str() << "/non_building_blocks";
        if (!createDirectory(nonBuildingBlocksFolderPath.str())) {
            std::cerr << "Failed to create directory: " << nonBuildingBlocksFolderPath.str() << "\n";
            continue;
        }

        // Open a file to store component information
        std::ofstream componentInfoFile(folderPath.str() + "/components_info.txt");
        if (!componentInfoFile.is_open()) {
            std::cerr << "Failed to create components_info.txt\n";
            continue;
        }

        std::vector<bool> visited(width * height, false);
        std::vector<bool> bigMask(width * height, false);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, 255);

        auto start = std::chrono::high_resolution_clock::now();

        int componentCount = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (!visited[y * width + x]) {
                    Color newColor = {static_cast<unsigned char>(distrib(gen)),
                                      static_cast<unsigned char>(distrib(gen)),
                                      static_cast<unsigned char>(distrib(gen))};

                    currentComponentXmin = 1000000000;
                    currentComponentXmax = -1000000000;
                    currentComponentYmin = 1000000000;
                    currentComponentYmax = -1000000000;
                    currentComponentSize = 0;

                    floodFillIterative(image, x, y, width, height, visited, image[y * width + x],
                                       config.k, config.use8Way, config.adj, config.euclidif, newColor, bigMask);

                    int componentWidth = currentComponentXmax - currentComponentXmin + 1;
                    int componentHeight = currentComponentYmax - currentComponentYmin + 1;

                    // Filter too-small components (try to keep only building blocks, heuristic)
                    if (currentComponentSize < config.minComponentSize || currentComponentSize < (componentWidth * componentHeight) / 3) {
                        for (int cx = currentComponentXmin; cx <= currentComponentXmax; cx++) {
                            for (int cy = currentComponentYmin; cy <= currentComponentYmax; cy++) {
                                bigMask[cy * width + cx] = false;
                            }
                        }
                        continue;
                    }

                    float heatmapThreshold = config.buildingBlockTreshold;

                    // Inside the component processing loop
                    float totalProbability = 0.0f;
                    int pixelCount = 0;

                    // Save mask for the component
                    std::vector<bool> mask(componentWidth * componentHeight, false);
                    for (int cx = currentComponentXmin; cx <= currentComponentXmax; cx++) {
                        for (int cy = currentComponentYmin; cy <= currentComponentYmax; cy++) {
                            if (bigMask[cy * width + cx]) {
                                mask[(cy - currentComponentYmin) * componentWidth + cx - currentComponentXmin] = true;
                                totalProbability += heatmap[cy * width + cx];
                                ++pixelCount;
                            }
                        }
                    }


                    float avgProbability = totalProbability / pixelCount;

                    for (int cx = currentComponentXmin; cx <= currentComponentXmax; cx++) {
                        for (int cy = currentComponentYmin; cy <= currentComponentYmax; cy++) {
                            if (bigMask[cy * width + cx]) {
                                bigMask[cy * width + cx] = false;
                                if(avgProbability >= heatmapThreshold && currentComponentSize < width*height/4)
                                    buildingBlocksImage[cy * width + cx] = {0,0,0};
                            }
                        }
                    }

                    // Save the component in the appropriate folder
                    std::ostringstream targetPath;
                    if (avgProbability >= heatmapThreshold) {
                        targetPath << buildingBlocksFolderPath.str() << "/component_" << std::setw(5) << std::setfill('0') << (componentCount + 1) << ".jpg";
                    } else {
                        targetPath << nonBuildingBlocksFolderPath.str() << "/component_" << std::setw(5) << std::setfill('0') << (componentCount + 1) << ".jpg";
                    }

                    saveMask(mask, componentWidth, componentHeight, targetPath.str());

                    // Write component information to the file
                    componentInfoFile << "Component " << (componentCount + 1) << ":\n";
                    componentInfoFile << "  Top-left corner: (" << currentComponentXmin << ", " << currentComponentYmin << ")\n";
                    componentInfoFile << "  Width: " << componentWidth << "\n";
                    componentInfoFile << "  Height: " << componentHeight << "\n";
                    componentInfoFile << "  Building block probability: " << avgProbability << "\n\n";

                    ++componentCount;
                    if(componentCount % 100 == 0) std::cout << componentCount << " processed components.\n";
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Finished processing: " << filePath
                  << " (Components: " << componentCount << ", Time: " << elapsed.count() << "s)\n";

        // Save segmentation image
        std::ostringstream segPath;
        segPath << outputDir << "/output_" << std::setw(3) << std::setfill('0') << (i + 1) << ".jpg";
        saveSegmentation(image, width, height, segPath.str());
        std::ostringstream buildingBlocksImagePath;
        buildingBlocksImagePath << outputDir << "/building_blocks_" << std::setw(3) << std::setfill('0') << (i + 1) << ".jpg";
        saveSegmentation(buildingBlocksImage, width, height, buildingBlocksImagePath.str());

        // Save segmentation in the folder as well
        std::ostringstream segFolderPath;
        segFolderPath << folderPath.str() << "/output.jpg";
        saveSegmentation(image, width, height, segFolderPath.str());

        // Close the component info file
        componentInfoFile.close();
    }
}


int main() {
    try {
        Config config = readConfig("config.txt");
        std::cout << "Configuration: k=" << config.k
                  << ", use8Way=" << config.use8Way
                  << ", euclidif=" << config.euclidif
                  << ", adj=" << config.adj
                  << ", minComponentSize=" << config.minComponentSize
                  << ", buildingBlockTreshold=" << config.buildingBlockTreshold<< "\n";

        processImages("to_process", "processed", config);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
