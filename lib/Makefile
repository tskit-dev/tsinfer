
CC=gcc
CFLAGS=-std=c99 -g -O3 -march=native -funroll-loops -ffast-math \
       # -ftree-vectorize \
       # -ftree-vectorizer-verbose=6 \
       # -fopt-info-vec-missed

all: main

main: main.c ls.o 
	gcc ${CFLAGS} -o main main.c ls.o -lm

ctags:
	ctags *.c *.h

clean:
	rm -f main *.o tags
