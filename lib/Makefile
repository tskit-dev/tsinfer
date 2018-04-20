
CC=h5cc
CFLAGS=-std=c99 -g -pedantic -Werror -Wall -W \
  -Wno-unused-parameter\
  -Wmissing-prototypes -Wstrict-prototypes \
  -Wconversion -Wshadow -Wpointer-arith \
  -Wcast-align -Wcast-qual \
  -Wwrite-strings -Wnested-externs \
  -fshort-enums -fno-common -Dinline= \
  -Wno-conversion # UTHash does not pass strict type conversion checks.
LDFLAGS=

COMPILED=ancestor_matcher.o object_heap.o ancestor_builder.o \
	 tree_sequence_builder.c block_allocator.o avl.o 

HEADERS=tsinfer.h err.h block_allocator.h object_heap.h

all: main 


argtable3.o: argtable3.c
	${CC} -Wall -g -O2 -c argtable3.c


# We need a seperate rule for avl.c as it won't pass the strict checks.
avl.o: avl.c
	${CC} -Wall -g -O2 -c avl.c

%.o : %.c ${HEADERS}
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

main: CFLAGS+=${EXTRA_CFLAGS}
main: main.c ${COMPILED} ${HEADERS} argtable3.o
	${CC} ${CFLAGS} ${EXTRA_CFLAGS} -o main main.c ${COMPILED} argtable3.o ${LDFLAGS} 

ctags:
	ctags *.c *.h

clean:
	rm -f main *.o tags
