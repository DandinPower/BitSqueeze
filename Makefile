CC      = gcc
CFLAGS  = -Wall -Wextra -Wpedantic -O3 -D_POSIX_C_SOURCE=199309L -Wno-misleading-indentation -Wno-format
LDFLAGS = -lm

UNAME_S := $(shell uname -s)

# ifeq ($(UNAME_S),Linux)
#     CFLAGS  += -fopenmp
#     LDFLAGS += -fopenmp
# endif

INCLUDE_DIR = include
SRC_DIR     = src
BUILD_DIR   = build
TEST_DIR    = test

# -------------------------------------------------------------
# Sources
# -------------------------------------------------------------
LIB_SRCS  := $(shell find $(SRC_DIR) -name '*.c')
LIB_OBJS  := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))

TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%,$(TEST_SRCS))

# -------------------------------------------------------------
# Targets
# -------------------------------------------------------------
.PHONY: all clean

all: $(TEST_BINS)

$(BUILD_DIR)/%: $(BUILD_DIR)/%.o $(LIB_OBJS) | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
