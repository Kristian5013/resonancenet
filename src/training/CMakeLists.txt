add_library(rnet_training STATIC
    model_config.cpp
    checkpoint_io.cpp
    training_engine.cpp
    evaluator.cpp
    tokenizer.cpp
    data_loader.cpp
    lr_schedule.cpp
)

target_include_directories(rnet_training PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(rnet_training PUBLIC
    rnet_gpu
    rnet_consensus
    rnet_crypto
    rnet_util
)
