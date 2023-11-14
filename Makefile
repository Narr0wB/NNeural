CC=g++
INCLUDE=-Iinclude/
COPTIONS=-g
TARGET=build/test.exe

OBJS=build/main.o build/Log.o build/Tensor.o


$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)

build/main.o: src/main.cpp src/tensor/Tensor.h src/tensor/TensorOperations.h
	$(CC) $(INCLUDE) $(COPTIONS) -c $< -o $@
 
build/Log.o: src/utils/Log.cpp src/utils/Log.h
	$(CC) $(INCLUDE) $(COPTIONS) -c $< -o $@

build/Tensor.o: src/tensor/Tensor.cpp
	$(CC) $(COPTIONS) -c $< -o $@

clean:
	rm build/*.o
	rm build/*.exe

# clean:
# 	rm $(TARGET)