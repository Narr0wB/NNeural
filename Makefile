CC=g++
INCLUDE=-Iinclude/
SRC= src/main.cpp src/utils/Log.cpp
TARGET=test

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET).exe $(INCLUDE)

clean:
	rm $(TARGET)