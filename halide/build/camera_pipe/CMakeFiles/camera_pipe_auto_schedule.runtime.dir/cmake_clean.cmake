file(REMOVE_RECURSE
  "camera_pipe_auto_schedule.runtime.o"
  "libcamera_pipe_auto_schedule.runtime.a"
  "libcamera_pipe_auto_schedule.runtime.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/camera_pipe_auto_schedule.runtime.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
