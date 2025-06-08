#include <utility>

template <int ID> void level5(int &out, int in) {
#pragma HLS inline off
  out = (in ^ (123 + ID)) + (in << 2) - (in >> 1) + 77;
}

template <int ID> void level4(int &out, int in) {
#pragma HLS inline off
  int tmp1, tmp2;
  level5<ID * 2>(tmp1, in);
  level5<ID * 2 + 1>(tmp2, in + 1);
  out = tmp1 * 2 + tmp2;
}

template <int ID> void level3(int &out, int in) {
#pragma HLS inline off
  int tmp1, tmp2;
  level4<ID * 2>(tmp1, in);
  level4<ID * 2 + 1>(tmp2, in + 2);
  out = tmp1 + tmp2 + (in % 3);
}

template <int ID> void level2(int &out, int in) {
#pragma HLS inline off
  int tmp1, tmp2;
  level3<ID * 2>(tmp1, in);
  level3<ID * 2 + 1>(tmp2, in + 3);
  out = tmp1 ^ tmp2;
}

template <int ID> void level1(int &out, int in) {
#pragma HLS inline off
  int tmp1, tmp2;
  level2<ID * 2>(tmp1, in);
  level2<ID * 2 + 1>(tmp2, in + 4);
  out = tmp1 + tmp2 + 5;
}

template <std::size_t... Is>
void nesting_impl(int in[8], int out[8], std::index_sequence<Is...>) {
#pragma HLS inline off
  (void(std::initializer_list<int>{(level1<Is>(out[Is], in[Is]), 0)...}));
}

void nesting(int in[8], int out[8]) {
#pragma HLS top name = nesting
#pragma HLS inline off
  nesting_impl(in, out, std::make_index_sequence<8>{});
}