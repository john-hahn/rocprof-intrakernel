// isa_classifier.cpp — Classify AMD GPU ISA instructions into categories.
//
// Used to build static instruction mix profiles from disassembled code objects.

#include "isa_classifier.h"

#include <algorithm>
#include <cctype>

namespace rikp {

InsnClass classify_instruction(const std::string& mnemonic) {
  if (mnemonic.empty()) return InsnClass::OTHER;

  // Extract the base mnemonic (before any suffix like _e32, _e64).
  std::string base = mnemonic;
  std::transform(base.begin(), base.end(), base.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // Vector ALU
  if (base.compare(0, 2, "v_") == 0) {
    // Vector memory operations
    if (base.find("_load") != std::string::npos) return InsnClass::VMEM_LOAD;
    if (base.find("_store") != std::string::npos) return InsnClass::VMEM_STORE;
    return InsnClass::VALU;
  }

  // Scalar ALU and control
  if (base.compare(0, 2, "s_") == 0) {
    // Scalar memory
    if (base.find("s_load") == 0) return InsnClass::SMEM_LOAD;
    if (base.find("s_store") == 0) return InsnClass::SMEM_STORE;
    // Branches
    if (base.find("s_branch") == 0 || base.find("s_cbranch") == 0) return InsnClass::BRANCH;
    // Wait counts
    if (base.find("s_waitcnt") == 0) return InsnClass::WAITCNT;
    // Barriers
    if (base == "s_barrier") return InsnClass::BARRIER;
    // Message/sendmsg
    if (base.find("s_sendmsg") == 0) return InsnClass::SENDMSG;
    // End program
    if (base == "s_endpgm") return InsnClass::OTHER;
    return InsnClass::SALU;
  }

  // LDS operations
  if (base.compare(0, 3, "ds_") == 0) return InsnClass::LDS;

  // Buffer/global/flat memory
  if (base.compare(0, 7, "buffer_") == 0) {
    if (base.find("_load") != std::string::npos) return InsnClass::VMEM_LOAD;
    if (base.find("_store") != std::string::npos) return InsnClass::VMEM_STORE;
    return InsnClass::VMEM_LOAD;  // atomic/other buffer ops
  }
  if (base.compare(0, 7, "global_") == 0) {
    if (base.find("_load") != std::string::npos) return InsnClass::VMEM_LOAD;
    if (base.find("_store") != std::string::npos) return InsnClass::VMEM_STORE;
    return InsnClass::VMEM_LOAD;
  }
  if (base.compare(0, 5, "flat_") == 0) {
    if (base.find("_load") != std::string::npos) return InsnClass::VMEM_LOAD;
    if (base.find("_store") != std::string::npos) return InsnClass::VMEM_STORE;
    return InsnClass::VMEM_LOAD;
  }

  // Export
  if (base.compare(0, 4, "exp ") == 0 || base == "exp") return InsnClass::EXPORT;

  // WMMA/matrix operations
  if (base.find("v_wmma") == 0 || base.find("v_mfma") == 0) return InsnClass::MATRIX;

  return InsnClass::OTHER;
}

const char* insn_class_name(InsnClass cls) {
  switch (cls) {
    case InsnClass::VALU:       return "VALU";
    case InsnClass::SALU:       return "SALU";
    case InsnClass::VMEM_LOAD:  return "VMEM_LOAD";
    case InsnClass::VMEM_STORE: return "VMEM_STORE";
    case InsnClass::SMEM_LOAD:  return "SMEM_LOAD";
    case InsnClass::SMEM_STORE: return "SMEM_STORE";
    case InsnClass::LDS:        return "LDS";
    case InsnClass::BRANCH:     return "BRANCH";
    case InsnClass::WAITCNT:    return "WAITCNT";
    case InsnClass::SENDMSG:    return "SENDMSG";
    case InsnClass::BARRIER:    return "BARRIER";
    case InsnClass::EXPORT:     return "EXPORT";
    case InsnClass::MATRIX:     return "MATRIX";
    case InsnClass::OTHER:      return "OTHER";
  }
  return "UNKNOWN";
}

}  // namespace rikp
