if(CONFIG_MODEL_MNASNET)
    message(FATAL_ERROR "MNASNET CURRENTLY NOT SUPPORTED")
endif()
if(CONFIG_MODEL_DENSENET)
    message(WARNING "DENSENET MIGHT BE TO BIG TO FIT INTO MEMORY")
endif()
set(src_dirs        ./
                    ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(include_dirs    ./
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)

list(APPEND requires    esp32_s3_eye_noglib
                         esp_lcd
                         esp-dl)

file(GLOB embed_imgs "${CMAKE_CURRENT_SOURCE_DIR}/data/*")
file(GLOB embed_models "${CMAKE_CURRENT_SOURCE_DIR}/models/*")

set(embed_files ${embed_imgs} ${embed_models})

idf_component_register(SRC_DIRS ${src_dirs} INCLUDE_DIRS ${include_dirs} REQUIRES ${requires} PRIV_REQUIRES nvs_flash fatfs EMBED_FILES ${embed_files})
