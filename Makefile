# Copyright (c) 2011, 2014, Gerhard Zumbusch.
# This software contains source code provided by NVIDIA Corporation.

CC = nvcc
CC += --use_fast_math
# CC += -G -g

SMS = 20 30 35 50
$(foreach sm,$(SMS),$(eval CC += -gencode arch=compute_$(sm),code=sm_$(sm)))

LIB = -lGL -lglut
BIN = ray

default: $(BIN)

ray: ray.cu cpu_anim.h gl_helper.h kern.h cholesky.h metric.h ppm.h
	$(CC) ray.cu -o ray $(LIB)

clean:
	rm -f $(BIN)


