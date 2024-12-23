//
// Created by nv on 24-12-21.
//

#include "fs.h"

#include <fstream>

#include "glog/logging.h"

std::string readFileToString(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::in | std::ios::binary);
    CHECK(file) << "Error opening file: " << filePath;
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

void writeStringToFile(const std::string& filePath, const std::string& content) {
    std::ofstream file(filePath, std::ios::out | std::ios::binary);
    CHECK(file) << "Error opening file for writing: " << filePath;
    file.write(content.c_str(), content.size());
    file.close();
}
