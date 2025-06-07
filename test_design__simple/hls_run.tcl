open_project -reset hls_proj

add_files ./vec_add.cpp

open_solution -reset -flow_target vivado hls_solution
set_part {xc7z020clg400-1}
create_clock -period 10 -name clk

set_top vec_add

csynth_design
