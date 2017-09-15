#pragma once

#include "SfMdata.h"

int readCamsFile(const char *filename, SfMdata &sfmdata);

int readPtsFile(const char *filename, SfMdata &fi);

void quat2vec(double *inp, int nin, double *outp);