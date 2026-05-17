# Makefile - possibilistic-inversion C port.
# Dependency-free: C standard library only.

CC      = cc
CFLAGS  = -O2 -std=c11 -Wall -Wextra
LDFLAGS = -lm
SRC     = possibilistic_inversion.c decomposition.c straightray.c eikonal.c

pi: $(SRC) grid.h decomposition.h straightray.h eikonal.h
	$(CC) $(CFLAGS) $(SRC) -o pi $(LDFLAGS)

run: pi
	./pi

clean:
	rm -f pi

.PHONY: run clean
