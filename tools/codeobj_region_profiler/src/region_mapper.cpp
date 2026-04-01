// region_mapper.cpp — Map PC offsets to code regions using disassembly analysis.
//
// This correlates PC samples (from rikp_pcsamp_client) with instruction
// classifications (from isa_classifier) and trace-defined regions.

#include "region_mapper.h"

#include <algorithm>

namespace rikp {

void RegionMapper::add_instruction(uint64_t pc_offset, const std::string& mnemonic, InsnClass cls) {
  instructions_.push_back({pc_offset, mnemonic, cls});
}

void RegionMapper::define_region(uint16_t id, const std::string& name,
                                  uint64_t start_pc, uint64_t end_pc) {
  regions_.push_back({id, name, start_pc, end_pc});
}

uint16_t RegionMapper::lookup_region(uint64_t pc_offset) const {
  for (const auto& r : regions_) {
    if (pc_offset >= r.start_pc && pc_offset < r.end_pc) {
      return r.id;
    }
  }
  return 0;  // "outside" / unmatched
}

RegionInsnMix RegionMapper::compute_mix(uint16_t region_id) const {
  RegionInsnMix mix;
  mix.region_id = region_id;

  for (const auto& inst : instructions_) {
    if (lookup_region(inst.pc_offset) == region_id) {
      ++mix.total;
      ++mix.by_class[static_cast<int>(inst.cls)];
    }
  }

  return mix;
}

}  // namespace rikp
