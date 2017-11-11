cmake_minimum_required( VERSION 2.8 )


set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

find_package( OpenCV 3.1 REQUIRED )



include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
   # ${G2O_INCLUDE_DIRS}
   # ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3/"
)

# pcl 
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

add_executable( 2d23d 2d23d.cpp  )
target_link_libraries( 2d23d ${OpenCV_LIBS}  ${PCL_LIBRARIES})

add_executable( feature_extraction feature_extration.cpp  )
target_link_libraries( feature_extraction ${OpenCV_LIBS} )

add_executable( sift_ransac sift_ransac.cpp  )
target_link_libraries( sift_ransac ${OpenCV_LIBS} )