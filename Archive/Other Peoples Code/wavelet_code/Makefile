SRCS=alloc.c wav_trf.c wav_basic.c test_denoise.c  test_transforms.c
OBJECTS=alloc.o wav_trf.o wav_basic.o test_denoise.o  test_transforms.o
CFLAGS = -O3 -Wall
LIBFLAGS = -lm 
.c.o:
	gcc -c $(CFLAGS) $<

all: test_denoise test_transforms
test_denoise: alloc.o wav_trf.o wav_basic.o test_denoise.o
	gcc -o test_denoise $(CFLAGS) alloc.o wav_trf.o wav_basic.o test_denoise.o $(LIBFLAGS)
	
test_transforms: alloc.o wav_trf.o wav_basic.o test_transforms.o
	gcc -o test_transforms $(CFLAGS) alloc.o wav_trf.o wav_basic.o test_transforms.o $(LIBFLAGS)
	
# DO NOT DELETE

alloc.o: macros.h 
wav_trf.o: wav_basic.h wav_filters_extern.h alloc.h wav_gen.h
wav_basic.o: wav_filters_extern.h alloc.h wav_gen.h
test_denoise.o: macros.h alloc.h wav_filters.h wav_trf.h wav_gen.h
test_transforms.o: macros.h alloc.h wav_filters.h wav_trf.h wav_gen.h
