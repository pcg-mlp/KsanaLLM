include(FetchContent)

set(LOGURU_INCLUDE_DIR ${THIRD_PARTY_PATH}/install/loguru)

FetchContent_Populate(download_loguru
    GIT_REPOSITORY https://git.woa.com/thirdsrc/emilk/loguru.git
    GIT_TAG f63653183f69c5b8987a4415773ca64d9f3fc2f4
    SOURCE_DIR ${LOGURU_INCLUDE_DIR}
    SUBBUILD_DIR ${THIRD_PARTY_PATH}/tmp
    BINARY_DIR ${THIRD_PARTY_PATH}/tmp
)

include_directories(${LOGURU_INCLUDE_DIR})
add_definitions(-DLOGURU_USE_FMTLIB=1)
add_definitions(-DLOGURU_WITH_STREAMS=1)

add_library(loguru SHARED ${THIRD_PARTY_PATH}/install/loguru/loguru.cpp)
target_link_libraries(loguru fmt)
