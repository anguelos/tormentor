INC=inc/
SRC=src/
BIN=bin/
OBJ=obj/

OPTS=-lsfml-graphics -lsfml-window -lsfml-system -I $(INC)
CC=g++

$(BIN)DiamondSquare: $(OBJ)main.o $(OBJ)DiamondSquare.o
	$(CC) $(OBJ)main.o $(OBJ)DiamondSquare.o -o $@ $(OPTS)

$(OBJ)main.o: main.cpp
	$(CC) -c main.cpp -o $(OBJ)main.o $(OPTS)

$(OBJ)DiamondSquare.o: $(SRC)DiamondSquare.cpp $(INC)DiamondSquare.h
	$(CC) -c $(SRC)DiamondSquare.cpp -o $(OBJ)DiamondSquare.o $(OPTS)
clean:
	rm -rf *.o
