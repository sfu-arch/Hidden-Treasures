file(REMOVE_RECURSE
  "camera_pipe.h"
  "camera_pipe.o"
  "camera_pipe.registration.cpp"
  "camera_pipe.stmt"
  "libcamera_pipe.a"
  "libcamera_pipe.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/camera_pipe.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
