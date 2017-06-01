
CC=gcc
CFLAGS=-std=c99 -pedantic -Werror -Wall -W \
  -Wno-unused-parameter\
  -Wmissing-prototypes -Wstrict-prototypes \
  -Wconversion -Wshadow -Wpointer-arith \
  -Wcast-align -Wcast-qual \
  -Wwrite-strings -Wnested-externs \
  -fshort-enums -fno-common -Dinline= 

all: main

main: main.c tsinfer.h ancestor_matcher.o
	gcc ${CFLAGS} -o main main.c ancestor_matcher.o -lm

ctags:
	ctags *.c *.h

clean:
	rm -f main *.o tags
