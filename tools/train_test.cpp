// Standalone GPU training test for ResonanceNet.
//
// Detects GPU (CUDA > CPU fallback), creates a small model, generates a
// synthetic dataset from embedded Shakespeare text (character-level), runs
// training for 100 steps, and prints loss at each step.
//
// Usage:
//   rnet-train-test                        (use embedded dataset, 100 steps)
//   rnet-train-test <dataset.bin>          (load dataset, 100 steps)
//   rnet-train-test <dataset.bin> <steps>  (load dataset, N steps)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>

#include "gpu/backend.h"
#include "gpu/context.h"
#include "training/training_engine.h"
#include "training/data_loader.h"
#include "training/model_config.h"

// Embedded sample text for testing (Shakespeare excerpt)
static const char* SAMPLE_TEXT =
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles,\n"
    "And by opposing end them. To die: to sleep;\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to, 'tis a consummation\n"
    "Devoutly to be wish'd. To die, to sleep;\n"
    "To sleep: perchance to dream: ay, there's the rub;\n"
    "For in that sleep of death what dreams may come\n"
    "When we have shuffled off this mortal coil,\n"
    "Must give us pause: there's the respect\n"
    "That makes calamity of so long life;\n"
    "For who would bear the whips and scorns of time,\n"
    "The oppressor's wrong, the proud man's contumely,\n"
    "The pangs of despised love, the law's delay,\n"
    "The insolence of office and the spurns\n"
    "That patient merit of the unworthy takes,\n"
    "When he himself might his quietus make\n"
    "With a bare bodkin? who would fardels bear,\n"
    "To grunt and sweat under a weary life,\n"
    "But that the dread of something after death,\n"
    "The undiscover'd country from whose bourn\n"
    "No traveller returns, puzzles the will\n"
    "And makes us rather bear those ills we have\n"
    "Than fly to others that we know not of?\n"
    "Thus conscience does make cowards of us all;\n"
    "And thus the native hue of resolution\n"
    "Is sicklied o'er with the pale cast of thought,\n"
    "And enterprises of great pith and moment\n"
    "With this regard their currents turn awry,\n"
    "And lose the name of action. Soft you now!\n"
    "The fair Ophelia! Nymph, in thy orisons\n"
    "Be all my sins remember'd.\n";

// Create a binary tokenized dataset from text (character-level, vocab=256)
static std::filesystem::path create_temp_dataset(const char* text) {
    auto path = std::filesystem::temp_directory_path() / "rnet_train_test.bin";
    size_t len = std::strlen(text);
    std::vector<int32_t> tokens(len);
    for (size_t i = 0; i < len; ++i) {
        tokens[i] = static_cast<int32_t>(static_cast<uint8_t>(text[i]));
    }

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(tokens.data()),
              static_cast<std::streamsize>(tokens.size() * sizeof(int32_t)));
    out.close();

    return path;
}

int main(int argc, char* argv[]) {
    printf("=== ResonanceNet GPU Training Test ===\n\n");

    // 1. Detect and create GPU backend
    auto backend_type = rnet::gpu::GpuBackend::auto_detect();
    printf("Backend: %s\n", rnet::gpu::backend_type_name(backend_type));

    auto backend = rnet::gpu::GpuBackend::create(backend_type);
    if (!backend) {
        fprintf(stderr, "Error: failed to create GPU backend\n");
        return 1;
    }
    printf("Device: %s\n", backend->device_name().c_str());
    printf("Memory: %.1f MB total, %.1f MB free\n\n",
           backend->total_memory() / (1024.0 * 1024.0),
           backend->free_memory() / (1024.0 * 1024.0));

    // 2. Create small model config for testing
    rnet::training::ModelConfig config;
    config.d_model = 128;
    config.n_layers = 2;
    config.n_slots = 16;
    config.d_ff = 256;
    config.vocab_size = 256;       // Character-level
    config.max_seq_len = 128;
    config.n_conv_branches = 3;
    config.kernel_sizes = {3, 7, 15, 0, 0, 0, 0, 0};

    printf("Model: d_model=%u, n_layers=%u, n_slots=%u, d_ff=%u, vocab=%u, seq_len=%u\n",
           config.d_model, config.n_layers, config.n_slots, config.d_ff,
           config.vocab_size, config.max_seq_len);
    printf("Parameters: %llu (%.2f MB at FP32)\n\n",
           static_cast<unsigned long long>(config.param_count()),
           config.param_count() * 4.0 / (1024.0 * 1024.0));

    // 3. Initialize training engine
    rnet::training::TrainingEngine engine(*backend);
    auto init_result = engine.init(config);
    if (init_result.is_err()) {
        fprintf(stderr, "Error: init failed: %s\n", init_result.error().c_str());
        return 1;
    }
    printf("Training engine initialized.\n");

    // 4. Load dataset
    std::filesystem::path dataset_path;
    if (argc >= 2) {
        dataset_path = argv[1];
        printf("Loading dataset: %s\n", dataset_path.string().c_str());
    } else {
        dataset_path = create_temp_dataset(SAMPLE_TEXT);
        printf("Using embedded Shakespeare dataset (%zu chars)\n", std::strlen(SAMPLE_TEXT));
    }

    rnet::training::DataLoader train_data;
    auto load_result = train_data.load_dataset(dataset_path);
    if (load_result.is_err()) {
        fprintf(stderr, "Error: failed to load dataset: %s\n", load_result.error().c_str());
        return 1;
    }
    printf("Dataset loaded: %zu tokens\n\n", train_data.total_tokens());

    // 5. Training loop
    int n_steps = 100;
    if (argc >= 3) {
        n_steps = std::atoi(argv[2]);
    }

    printf("Training for %d steps...\n", n_steps);
    printf("%-8s %-12s %-12s %-10s\n", "Step", "Loss", "Perplexity", "Time(ms)");
    printf("-------- ------------ ------------ ----------\n");

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < n_steps; ++step) {
        auto step_start = std::chrono::high_resolution_clock::now();

        auto result = engine.train_steps(1, train_data);

        auto step_end = std::chrono::high_resolution_clock::now();
        double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

        if (result.is_err()) {
            fprintf(stderr, "Error at step %d: %s\n", step, result.error().c_str());
            return 1;
        }

        float loss = result.value();
        float perplexity = std::exp(std::min(loss, 20.0f));  // Cap to avoid overflow

        // Print every step for first 10, then every 10 steps
        if (step < 10 || step % 10 == 0 || step == n_steps - 1) {
            printf("%-8d %-12.4f %-12.1f %-10.1f\n", step, loss, perplexity, step_ms);
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(total_end - total_start).count();

    printf("\nTraining complete in %.1f seconds (%.1f steps/sec)\n", total_s, n_steps / total_s);

    // 6. Final evaluation
    train_data.reset();
    auto eval_result = engine.evaluate(train_data, 5);
    if (eval_result.is_ok()) {
        printf("Final validation loss: %.4f (perplexity: %.1f)\n",
               eval_result.value(), std::exp(std::min(eval_result.value(), 20.0f)));
    }

    // 7. Save checkpoint
    auto ckpt_path = std::filesystem::current_path() / "test_checkpoint.rnet";
    auto save_result = engine.save_checkpoint(ckpt_path);
    if (save_result.is_ok()) {
        printf("Checkpoint saved: %s\n", ckpt_path.string().c_str());
    }

    printf("\n=== Test Complete ===\n");
    return 0;
}
