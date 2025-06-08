
const int N_ELEMENTS = 64;
const int TILE_SIZE = 16;

void vec_add(int a[N_ELEMENTS], int b[N_ELEMENTS], int c[N_ELEMENTS]) {
#pragma HLS top name = "vec_add"

#pragma HLS array_partition variable = a type = cyclic factor =                \
    TILE_SIZE dim = 0
#pragma HLS array_partition variable = b type = cyclic factor =                \
    TILE_SIZE dim = 0
#pragma HLS array_partition variable = c type = cyclic factor =                \
    TILE_SIZE dim = 0

  for (int i = 0; i < N_ELEMENTS; i += TILE_SIZE) {
#pragma HLS unroll off = true
#pragma HLS pipeline II = 1
    for (int j = 0; j < TILE_SIZE; j++) {
#pragma HLS unroll
      // c[i + j] = a[i + j] + b[i + j];
      int a_val = a[i + j];
      int b_val = b[i + j];
      int c_val = a_val + b_val;
      c[i + j] = c_val;
    }
  }
}