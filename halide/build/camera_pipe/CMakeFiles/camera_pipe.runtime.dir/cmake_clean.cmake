file(REMOVE_RECURSE
  "camera_pipe.runtime.o"
  "libcamera_pipe.runtime.a"
  "libcamera_pipe.runtime.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/camera_pipe.runtime.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
