CC = gcc
CC1=mpicc
CFLAGS = -Wall -lm -fopenmp -g -O2
EXEC = exec
INCLUDE = ./headers/
all : ./exec
par: ./exec_par
clean : cleanmaster
./lib/libOperations.a : lib/Lib_basic_fonction.c headers/Lib_basic_fonction.h
		$(CC) -o ./lib/Lib_basic_fonction.o -c $< -I$(INCLUDE) $(CFLAGS)
		ar rcs $@ ./lib/Lib_basic_fonction.o

./exec: main.c ./lib/libOperations.a
		$(CC) $< -I$(INCLUDE) -L./lib -lOperations -o $@ $(CFLAGS)

./lib/libOperation.a : lib/Lib_basic_fonction_paralel.c headers/Lib_basic_fonction_paralel.h
		$(CC1) -o ./lib/Lib_basic_fonction.o -c $< -I$(INCLUDE) $(CFLAGS)
		ar rcs $@ ./lib/Lib_basic_fonction.o

./exec_par: par.c ./lib/libOperation.a
		$(CC1) $< -I$(INCLUDE) -L./lib -lOperation -o $@ $(CFLAGS)



clean:
	rm -f *.o
	rm  ex*

#################
##### CLEAN #####
#################
cleanmaster :
	make cleanlib

cleanlib :
	rm -f ./lib/*.a
	rm -f ./lib/*.so
	rm -f ./lib/*.o
	rm -f 2*.txt


