CC = gcc

TARGET = demo_app

SRCS = main.c
OBJS = $(SRCS:.c=.o)
INCLUDE_DIR = include

CFLAGS = -Wall -Wextra -std=gnu11 -I$(INCLUDE_DIR)

# Linker flags:
# -lbitsqz: Link with the bitsqz library (assumes libbitsqz.so or libbitsqz.a is installed)
LDFLAGS = -lbitsqz

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean