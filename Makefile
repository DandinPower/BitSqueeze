CC      = gcc
# CFLAGS  = -Wall -Wextra -Wpedantic -O3 -std=c11 -fopenmp
CFLAGS  = -Wall -Wextra -Wpedantic -O3 -std=c11
LDFLAGS = -lm

INCLUDE_DIR = include
SRC_DIR     = src
BUILD_DIR   = build
TEST_DIR    = test

# -------------------------------------------------------------
# Sources
# -------------------------------------------------------------
LIB_SRCS  := $(wildcard $(SRC_DIR)/*.c)
LIB_OBJS  := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))

TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))

QUANT_TEST := $(BUILD_DIR)/test_quantization
SPARSE_TEST := $(BUILD_DIR)/test_sparsity
REAL_TEST := $(BUILD_DIR)/test_real_example
KQ_TEST := $(BUILD_DIR)/test_k_quantization
FP16_TEST := $(BUILD_DIR)/test_fp16

# -------------------------------------------------------------
# Targets
# -------------------------------------------------------------
.PHONY: all clean run-fp16

all: $(QUANT_TEST) $(SPARSE_TEST) $(REAL_TEST) $(KQ_TEST) $(FP16_TEST)

$(QUANT_TEST): $(LIB_OBJS) $(BUILD_DIR)/test_quantization.o
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $^ $(LDFLAGS)

$(SPARSE_TEST): $(LIB_OBJS) $(BUILD_DIR)/test_sparsity.o
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $^ $(LDFLAGS)

$(REAL_TEST): $(LIB_OBJS) $(BUILD_DIR)/test_real_example.o
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $^ $(LDFLAGS)

$(KQ_TEST): $(LIB_OBJS) $(BUILD_DIR)/test_k_quantization.o
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $^ $(LDFLAGS)

# NEW: fp16 round-trip tester
$(FP16_TEST): $(BUILD_DIR)/test_fp16.o
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run-fp16: $(FP16_TEST)
	$(FP16_TEST)

clean:
	rm -rf $(BUILD_DIR)
	@rm -f $(QUANT_TEST) $(SPARSE_TEST) $(REAL_TEST) $(KQ_TEST) $(FP16_TEST)
