
CC=gcc
CFLAGS=-std=c99 -g -pedantic -Werror -Wall -W \
  -Wno-unused-parameter\
  -Wmissing-prototypes -Wstrict-prototypes \
  -Wconversion -Wshadow -Wpointer-arith \
  -Wcast-align -Wcast-qual \
  -Wwrite-strings -Wnested-externs \
  -fshort-enums -fno-common -Dinline= 
LDFLAGS=-lm

COMPILED=ancestor_matcher.o object_heap.o ancestor_builder.o
HEADERS=tsinfer.h err.h

all: main 

argtable3.o: argtable3.c
	${CC} -Wall -g -O2 -c argtable3.c

%.o : %.c ${HEADERS}
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

main: CFLAGS+=${EXTRA_CFLAGS}
main: main.c ${COMPILED} ${HEADERS} argtable3.o
	${CC} ${CFLAGS} ${EXTRA_CFLAGS} -o main main.c ${COMPILED} argtable3.o ${LDFLAGS} 

ctags:
	ctags *.c *.h

clean:
	rm -f main *.o tags
