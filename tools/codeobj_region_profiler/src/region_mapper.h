#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "isa_classifier.h"

namespace rikp {

struct InstructionEntry {
  uint64_t pc_offset;
  std::string mnemonic;
  InsnClass cls;
};

struct RegionDef {
  uint16_t id;
  std::string name;
  uint64_t start_pc;
  uint64_t end_pc;
};

struct RegionInsnMix {
  uint16_t region_id = 0;
  uint64_t total = 0;
  // Indexed by static_cast<int>(InsnClass)
  uint64_t by_class[14] = {};
};

class RegionMapper {
 public:
  void add_instruction(uint64_t pc_offset, const std::string& mnemonic, InsnClass cls);
  void define_region(uint16_t id, const std::string& name,
                     uint64_t start_pc, uint64_t end_pc);
  uint16_t lookup_region(uint64_t pc_offset) const;
  RegionInsnMix compute_mix(uint16_t region_id) const;

 private:
  std::vector<InstructionEntry> instructions_;
  std::vector<RegionDef> regions_;
};

}  // namespace rikp
