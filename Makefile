# Compiler and Flags
CC = g++
CFLAGS = -std=c++17 -g -fPIC -fopenmp

# Directories
INCLUDE_DIR = include
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
LIB_DIR = lib

# Target Executable and Library
EXECUTABLE = $(BIN_DIR)/main.exe
LIBRARY = $(LIB_DIR)/libdata.dll

# Source and Object Files
SOURCES = $(wildcard $(SRC_DIR)/*.cc)
OBJECTS = $(patsubst $(SRC_DIR)/%.cc, $(OBJ_DIR)/%.o, $(SOURCES))

# Default Make Target
.PHONY: all
all: $(LIBRARY) $(EXECUTABLE)

# Build the Shared Library
$(LIBRARY): $(LIB_DIR) $(OBJECTS)
	$(CC) -shared -o $(LIBRARY) $(OBJECTS) -fopenmp 

# Build the Executable
$(EXECUTABLE): $(BIN_DIR) $(OBJECTS)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $(EXECUTABLE) $(OBJECTS)

# Compile Source Files into Object Files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc | $(OBJ_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Create Necessary Directories (Windows)
$(OBJ_DIR):
	if not exist $(OBJ_DIR) mkdir $(OBJ_DIR)

$(BIN_DIR):
	if not exist $(BIN_DIR) mkdir $(BIN_DIR)

$(LIB_DIR):
	if not exist $(LIB_DIR) mkdir $(LIB_DIR)

# Clean Up Generated Files (Windows)
.PHONY: clean
clean:
	if exist $(OBJ_DIR) rmdir /S /Q $(OBJ_DIR)
	if exist $(BIN_DIR) rmdir /S /Q $(BIN_DIR)
	if exist $(LIB_DIR) rmdir /S /Q $(LIB_DIR)
