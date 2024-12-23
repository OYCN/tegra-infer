//
// Created by nv on 24-12-22.
//

#ifndef KERNEL_H
#define KERNEL_H

#include "typedef.h"

void decoder(const DecoderParam& param, cudaStream_t stream);

#endif //KERNEL_H
