#pragma once

constexpr bool PROFILE = false;
constexpr bool PROFILEREF = false;
constexpr int WARMUP = 2;
constexpr int NREPEATS = PROFILE ? 128 : 16;
constexpr float TOLERANCE = 1e-5; // relative tolerance used by check_result

constexpr int VEC_SIZE = 4;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BL = 8;
constexpr int RM = 4;
constexpr int RN = 4;