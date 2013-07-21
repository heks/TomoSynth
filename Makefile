# (c) 2010 The Board of Trustees of the University of Illinois.

LANGUAGE=cuda
SRCDIR_OBJS=TomoSynth.o DectorConfig.o DynRangeEqual.o trt_utils.o WeightTables.o Reconstruct.o  writeTif.o EdgeGainCorrect.o
APP_CUDALDFLAGS=-ltiff
APP_CUDACFLAGS=-arch sm_20