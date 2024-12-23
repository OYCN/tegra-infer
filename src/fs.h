//
// Created by nv on 24-12-21.
//

#ifndef FS_H
#define FS_H

#include <string>

std::string readFileToString(const std::string& filePath);
void writeStringToFile(const std::string& filePath, const std::string& content);

#endif //FS_H
