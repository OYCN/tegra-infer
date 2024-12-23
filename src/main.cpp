//
// Created by nv on 24-12-21.
//

#include "pipline.h"
#include "tensorrt.h"
#include "fs.h"

int main() {
    auto txt = readFileToString("./config.json");
    auto j = nlohmann::json::parse(txt);
    Pipline pipline(j);
    // for (int i = 0; i < 30; i++) {
    while (true) {
        pipline.process();
    }
    TrtModels::destroy();
    return 0;
}
