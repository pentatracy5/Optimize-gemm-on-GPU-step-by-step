#pragma once

constexpr bool PROFILE = false;
constexpr bool PROFILEREF = false;
constexpr int WARMUP = 2;
constexpr int NREPEATS = PROFILE ? 128 : 16;
constexpr float TOLERANCE = 1e-5; // relative tolerance used by check_result

constexpr float ALPHA = 1.0f;
constexpr float BETA = 0.0f;

constexpr int ONE = 1;
constexpr int THREADS_PER_BLOCK = (ONE << 8);           // at least (ONE << 5), maximum (ONE << 10)
constexpr int VEC_SIZE = (ONE << 2);                    // (ONE << 0), (ONE << 1) or (ONE << 2)
constexpr int RM = (ONE << 1) * VEC_SIZE;
constexpr int RN = (ONE << 1) * VEC_SIZE;
constexpr int BL = (ONE << 1) * VEC_SIZE;
constexpr int BM = (ONE << 4) * RM;                     // (ONE << x) here should be less than THREADS_PER_BLOCK
constexpr int BN = THREADS_PER_BLOCK / (BM / RM) * RN;
