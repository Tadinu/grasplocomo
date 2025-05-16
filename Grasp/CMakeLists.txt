IF (DX_BUILD_TEST)
    FILE(GLOB_RECURSE TEST_SOURCES_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp" )
    FILE(GLOB_RECURSE TEST_INCLUDE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/test/*.h")
    ADD_EXECUTABLE(dxTestGraspLoCoMo ${TEST_SOURCES_FILES} ${TEST_INCLUDE_FILES})
	TARGET_LINK_LIBRARIES(dxTestGraspLoCoMo libmjgrasplocomo)
    TARGET_INCLUDE_DIRECTORIES(dxTestGraspLoCoMo
            PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${PROJECT_SOURCE_DIR}/ThirdParty/nanoflann/include/
            ${PROJECT_SOURCE_DIR}/Core/Sys/include/
            ${PROJECT_SOURCE_DIR}/Core/Math/include/
            ${PROJECT_SOURCE_DIR}/Core/Search/include/
            ${PROJECT_SOURCE_DIR}/Grasp/include
    )

    SET_PROPERTY(TARGET dxTestGraspLoCoMo PROPERTY PROJECT_LABEL "TestdxGraspLoCoMo")
    SET_PROPERTY(TARGET dxTestGraspLoCoMo PROPERTY FOLDER "Modules/dxGraspLoCoMo/Test")

ENDIF (DX_BUILD_TEST)
