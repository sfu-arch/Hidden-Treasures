file(REMOVE_RECURSE
  "camera_pipe_auto_schedule.h"
  "camera_pipe_auto_schedule.o"
  "camera_pipe_auto_schedule.registration.cpp"
  "camera_pipe_auto_schedule.stmt"
  "libcamera_pipe_auto_schedule.a"
  "libcamera_pipe_auto_schedule.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/camera_pipe_auto_schedule.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
