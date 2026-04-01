#pragma once

#include <string>

namespace rikp {

enum class InsnClass {
  VALU,        // v_* (vector ALU, excluding memory)
  SALU,        // s_* (scalar ALU, excluding memory/branch/waitcnt)
  VMEM_LOAD,   // buffer_load_*, global_load_*, flat_load_*, v_*_load*
  VMEM_STORE,  // buffer_store_*, global_store_*, flat_store_*, v_*_store*
  SMEM_LOAD,   // s_load_*
  SMEM_STORE,  // s_store_*
  LDS,         // ds_*
  BRANCH,      // s_branch, s_cbranch_*
  WAITCNT,     // s_waitcnt*
  SENDMSG,     // s_sendmsg*
  BARRIER,     // s_barrier
  EXPORT,      // export
  MATRIX,      // v_wmma_*, v_mfma_*
  OTHER,
};

// Classify an AMD GPU ISA mnemonic into a category.
InsnClass classify_instruction(const std::string& mnemonic);

// Get human-readable name for an instruction class.
const char* insn_class_name(InsnClass cls);

}  // namespace rikp
