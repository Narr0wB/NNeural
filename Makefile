CC=g++
INCLUDE=-Iinclude/
SRC=src/main.cpp src/utils/Log.cpp src/tensor/Operations.cpp
TARGET=build/test.exe

OBJS=build/main.o build/Log.o build/Operations.o


$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)

build/main.o: src/main.cpp src/tensor/Tensor.h
	$(CC) -g -c $< -o $@
 
build/Log.o: src/utils/Log.cpp src/utils/Log.h
	$(CC) $(INCLUDE) -g -c $< -o $@

build/Operations.o: src/tensor/Operations.cpp src/tensor/Operations.h
	$(CC) -g -c $< -o $@




# clean:
# 	rm $(TARGET)