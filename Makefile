CC= g++
INCLUDE= -Iinclude/ -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/include"
LIBPATH= -Llib/
LIB= -lopencl
COPTIONS= -g -O0
TARGET= build/test.exe

OBJS=build/main.o build/Log.o build/Tensor.o build/Memory.o


$(TARGET): $(OBJS)
	$(CC) $(LIBPATH) $(COPTIONS) $(OBJS) -o $(TARGET) $(LIB)

build/main.o: src/main.cpp src/utils/Memory.h src/tensor/Tensor.h
	$(CC) $(INCLUDE) $(COPTIONS) -c $< -o $@
 
build/Log.o: src/utils/Log.cpp src/utils/Log.h
	$(CC) $(INCLUDE) $(COPTIONS) -c $< -o $@

build/Tensor.o: src/tensor/Tensor.cpp src/tensor/Tensor.h src/tensor/TensorOperations.h 
	$(CC) $(INCLUDE) $(COPTIONS) -c $< -o $@

build/Memory.o: src/utils/Hardware.cpp src/utils/Memory.h src/utils/Hardware.h
	$(CC) $(INCLUDE) $(COPTIONS) -c $< -o $@

clean:
	rm build/*.o
	powershell rm build/*.exe

# clean:
# 	rm $(TARGET)