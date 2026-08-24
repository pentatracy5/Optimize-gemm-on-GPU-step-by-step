#pragma once

constexpr bool PROFILE = false;
constexpr bool PROFILEREF = false;
constexpr int WARMUP = 2;
constexpr int NREPEATS = PROFILE ? 128 : 16;
constexpr float TOLERANCE = 1e-5; // relative tolerance used by check_result

constexpr float ALPHA = 1.0f;
constexpr float BETA = 0.0f;

constexpr int VEC_SIZE = 4;
constexpr int RM = 1 * VEC_SIZE;
constexpr int RN = 1 * VEC_SIZE;
constexpr int BL = 2 * VEC_SIZE;
constexpr int BM = 16 * RM;
constexpr int BN = 16 * RN;
