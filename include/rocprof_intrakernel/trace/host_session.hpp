#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include "rocprof_intrakernel/trace/event.hpp"

namespace rocprof_intrakernel::trace {

struct TraceWriteOptions {
  // Chrome trace expects "ns" when displayTimeUnit="ns".
  // `scale` converts raw realtime clock ticks to ns (or whatever unit you want).
  double scale = 1.0;

  // If true, pair begin/end on host and emit Chrome Complete events (ph:"X").
  bool emit_complete_events = true;

  // If true: pid := cu_id, tid := (block<<6)|wave. Else: pid := block, tid := wave*wavesize.
  bool group_by_cu = true;

  bool emit_summary_json = true;
  uint32_t summary_hist_bins = 128;
  bool summary_dump_by_block_wave = true;

  bool emit_block_region_distributions = false;
  uint32_t block_region_hist_bins = 128;

  uint32_t summary_topk_block_wave_per_region = 0;
  uint32_t block_region_topk_blocks = 0;
  uint32_t block_region_topk_regions_per_block = 0;
};

inline void check_hip(hipError_t e, const char* msg) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", msg, hipGetErrorString(e));
    std::exit(1);
  }
}

static constexpr const char* kChromeTraceCnameList[] = {
    "thread_state_uninterruptible",
    "thread_state_iowait",
    "thread_state_running",
    "thread_state_runnable",
    "thread_state_sleeping",
    "thread_state_unknown",
    "background_memory_dump",
    "light_memory_dump",
    "detailed_memory_dump",
    "vsync_highlight_color",
    "generic_work",
    "good",
    "bad",
    "terrible",
    "black",
    "grey",
    "white",
    "yellow",
    "olive",
    "rail_response",
    "rail_animation",
    "rail_idle",
    "rail_load",
    "startup",
    "heap_dump_stack_frame",
    "heap_dump_object_type",
    "heap_dump_child_node_arrow",
    "cq_build_running",
    "cq_build_passed",
    "cq_build_failed",
    "cq_build_abandoned",
    "cq_build_attempt_runnig",
    "cq_build_attempt_passed",
    "cq_build_attempt_failed",
    "rail_animate",
    "cq_build_attempt_running",
};
static constexpr uint32_t kChromeTraceCnameCount =
    sizeof(kChromeTraceCnameList) / sizeof(kChromeTraceCnameList[0]);

class HostSession {
 public:
  HostSession() = default;
  HostSession(const HostSession&) = delete;
  HostSession& operator=(const HostSession&) = delete;

  ~HostSession() { destroy(); }

  void init(uint32_t per_wave_cap, uint32_t grid_x, uint32_t threads_per_block) {
    destroy();
    if (per_wave_cap == 0 || (per_wave_cap & (per_wave_cap - 1u)) != 0u) {
      std::fprintf(stderr,
                   "rocprof_intrakernel::trace: per_wave_cap must be power-of-2 and >0 (got %u)\n",
                   per_wave_cap);
      std::exit(1);
    }
    per_wave_cap_ = per_wave_cap;
    blocks_ = grid_x;
    // Use the compile-time wave size for host-side calculation.
    // Default to 64 for host code (conservative; matches CDNA).
    waves_per_block_ = (threads_per_block + 63u) / 64u;
    const uint32_t total_slots = blocks_ * waves_per_block_;
    const size_t total_events = size_t(total_slots) * size_t(per_wave_cap_);
    check_hip(hipMalloc(&d_events_, sizeof(Event) * total_events), "hipMalloc events");
    check_hip(hipMalloc(&d_counters_, sizeof(uint32_t) * total_slots), "hipMalloc counters");
    gbuf_ = GlobalBuffer{d_events_, d_counters_};
    reset();
  }

  void init(uint32_t per_wave_cap, uint32_t grid_x, uint32_t threads_per_block,
            uint32_t wave_size) {
    destroy();
    if (per_wave_cap == 0 || (per_wave_cap & (per_wave_cap - 1u)) != 0u) {
      std::fprintf(stderr,
                   "rocprof_intrakernel::trace: per_wave_cap must be power-of-2 and >0 (got %u)\n",
                   per_wave_cap);
      std::exit(1);
    }
    per_wave_cap_ = per_wave_cap;
    blocks_ = grid_x;
    waves_per_block_ = (threads_per_block + wave_size - 1u) / wave_size;
    const uint32_t total_slots = blocks_ * waves_per_block_;
    const size_t total_events = size_t(total_slots) * size_t(per_wave_cap_);
    check_hip(hipMalloc(&d_events_, sizeof(Event) * total_events), "hipMalloc events");
    check_hip(hipMalloc(&d_counters_, sizeof(uint32_t) * total_slots), "hipMalloc counters");
    gbuf_ = GlobalBuffer{d_events_, d_counters_};
    reset();
  }

  void reset() {
    const uint32_t total_slots = blocks_ * waves_per_block_;
    if (d_counters_) {
      check_hip(hipMemset(d_counters_, 0, sizeof(uint32_t) * total_slots), "reset counters");
    }
  }

  GlobalBuffer& global_buffer() { return gbuf_; }
  const GlobalBuffer& global_buffer() const { return gbuf_; }

  void set_region_names(std::vector<std::string> names) { region_names_ = std::move(names); }

  void set_block_filter(std::vector<uint32_t> blocks) {
    block_filter_.clear();
    block_filter_.reserve(blocks.size());
    for (uint32_t b : blocks) block_filter_.insert(b);
  }

  void clear_block_filter() { block_filter_.clear(); }

  static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
      switch (c) {
        case '\"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
          if (c < 0x20) break;
          out.push_back(static_cast<char>(c));
          break;
      }
    }
    return out;
  }

  static void ensure_dir(const std::string& path) {
    if (path.empty()) return;
    std::string cur;
    cur.reserve(path.size() + 8);
    for (size_t i = 0; i < path.size(); ++i) {
      const char c = path[i];
      if (c == '/') {
        if (!cur.empty()) (void)mkdir(cur.c_str(), 0755);
      }
      cur.push_back(c);
    }
    if (!cur.empty()) (void)mkdir(cur.c_str(), 0755);
  }

  static void ensure_parent_dir_for_file(const std::string& file_path) {
    const size_t pos = file_path.find_last_of('/');
    if (pos == std::string::npos) return;
    ensure_dir(file_path.substr(0, pos + 1));
  }

  static std::string strip_json_suffix(const std::string& p) {
    const std::string suf = ".json";
    if (p.size() >= suf.size() && p.compare(p.size() - suf.size(), suf.size(), suf) == 0) {
      return p.substr(0, p.size() - suf.size());
    }
    return p;
  }

  void write_trace(const std::string& path, const TraceWriteOptions& opt = TraceWriteOptions{}) {
    const uint32_t total_slots = blocks_ * waves_per_block_;
    std::vector<uint32_t> counts(total_slots);
    check_hip(hipMemcpy(counts.data(), d_counters_, sizeof(uint32_t) * total_slots,
                        hipMemcpyDeviceToHost),
              "copy counts");

    const size_t total_events = size_t(total_slots) * size_t(per_wave_cap_);
    std::vector<Event> raw(total_events);
    check_hip(hipMemcpy(raw.data(), d_events_, sizeof(Event) * total_events,
                        hipMemcpyDeviceToHost),
              "copy events");

    // Unwind circular buffers: if cnt > cap, only keep last cap events.
    std::vector<Event> ordered;
    ordered.reserve(total_slots * 8);
    for (uint32_t s = 0; s < total_slots; ++s) {
      const uint32_t cnt = counts[s];
      const uint32_t n = std::min(cnt, per_wave_cap_);
      const uint32_t start = (cnt > per_wave_cap_) ? (cnt & (per_wave_cap_ - 1u)) : 0u;
      const size_t slot_base = size_t(s) * size_t(per_wave_cap_);
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t idx = (start + i) & (per_wave_cap_ - 1u);
        ordered.push_back(raw[slot_base + idx]);
      }
    }

    if (!block_filter_.empty()) {
      std::vector<Event> filtered;
      filtered.reserve(ordered.size());
      for (const auto& e : ordered) {
        if (block_filter_.find(uint32_t(e.block)) != block_filter_.end()) filtered.push_back(e);
      }
      ordered.swap(filtered);
    }

    if (ordered.empty()) {
      std::printf("rocprof_intrakernel::trace: 0 events\n");
      return;
    }

    // Deterministic sort.
    auto type_prio = [](uint16_t t) -> uint32_t {
      return (t == 0) ? 0u : (t == 2) ? 1u : 2u;
    };
    std::sort(ordered.begin(), ordered.end(), [&](const Event& a, const Event& b) {
      if (a.ts != b.ts) return a.ts < b.ts;
      const uint32_t ca = uint32_t(unpack_cu(a.wave));
      const uint32_t cb = uint32_t(unpack_cu(b.wave));
      if (ca != cb) return ca < cb;
      if (a.block != b.block) return a.block < b.block;
      const uint32_t wa = uint32_t(unpack_wave(a.wave));
      const uint32_t wb = uint32_t(unpack_wave(b.wave));
      if (wa != wb) return wa < wb;
      const uint32_t pa = type_prio(a.type);
      const uint32_t pb = type_prio(b.type);
      if (pa != pb) return pa < pb;
      if (a.id != b.id) return a.id < b.id;
      return false;
    });

    // Map block -> cu (best-effort).
    std::unordered_map<uint32_t, uint32_t> block_to_cu;
    block_to_cu.reserve(256);
    for (const auto& e : ordered) {
      const uint32_t b = uint32_t(e.block);
      if (block_to_cu.find(b) == block_to_cu.end()) {
        block_to_cu.emplace(b, uint32_t(unpack_cu(e.wave)));
      }
    }

    ensure_parent_dir_for_file(path);
    std::ofstream out(path);
    out << "{\"displayTimeUnit\":\"ns\",\"traceEvents\":[\n";
    bool first = true;
    auto emit = [&](const std::string& json_obj) {
      if (!first) out << ",\n";
      first = false;
      out << json_obj;
    };

    const uint64_t min_ts = ordered[0].ts;

    // Name processes/threads for the Chrome trace viewer.
    std::vector<uint32_t> used_pids;
    used_pids.reserve(256);
    std::unordered_set<uint64_t> used_threads;
    used_threads.reserve(1024);
    for (const auto& e : ordered) {
      const uint32_t b = uint32_t(e.block);
      const uint32_t wave = uint32_t(unpack_wave(e.wave));
      const uint32_t cu = uint32_t(unpack_cu(e.wave));
      const uint32_t pid = opt.group_by_cu ? cu : b;
      const uint32_t tid = opt.group_by_cu ? ((b << 6) | wave) : (wave * 64u);
      used_pids.push_back(pid);
      used_threads.insert((uint64_t(pid) << 32) | uint64_t(tid));
    }
    std::sort(used_pids.begin(), used_pids.end());
    used_pids.erase(std::unique(used_pids.begin(), used_pids.end()), used_pids.end());

    for (uint32_t pid : used_pids) {
      std::string pname;
      if (opt.group_by_cu) {
        pname = "cu " + std::to_string(pid);
      } else {
        auto it = block_to_cu.find(pid);
        if (it != block_to_cu.end())
          pname = "cu " + std::to_string(it->second) + " block " + std::to_string(pid);
        else
          pname = "block " + std::to_string(pid);
      }
      emit("{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":" + std::to_string(pid) +
           ",\"tid\":0,\"args\":{\"name\":\"" + json_escape(pname) + "\"}}");
    }

    std::vector<uint64_t> used_thread_keys;
    used_thread_keys.reserve(used_threads.size());
    for (uint64_t key : used_threads) used_thread_keys.push_back(key);
    std::sort(used_thread_keys.begin(), used_thread_keys.end());

    for (uint64_t key : used_thread_keys) {
      const uint32_t pid = uint32_t(key >> 32);
      const uint32_t tid = uint32_t(key & 0xFFFFFFFFu);
      std::string tname;
      if (opt.group_by_cu) {
        const uint32_t b = tid >> 6;
        const uint32_t wave = tid & 0x3Fu;
        tname = "block " + std::to_string(b) + " wave " + std::to_string(wave);
      } else {
        const uint32_t wave = tid / 64u;
        tname = "wave " + std::to_string(wave);
      }
      emit("{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":" + std::to_string(pid) +
           ",\"tid\":" + std::to_string(tid) +
           ",\"args\":{\"name\":\"" + json_escape(tname) + "\"}}");
    }

    if (!opt.emit_complete_events) {
      // Legacy mode: emit raw B/E/i events.
      std::vector<Event> legacy_events = ordered;
      if (opt.group_by_cu) {
        std::sort(legacy_events.begin(), legacy_events.end(), [&](const Event& a, const Event& b) {
          const uint32_t ba = uint32_t(a.block);
          const uint32_t wa = uint32_t(unpack_wave(a.wave));
          const uint32_t cua = uint32_t(unpack_cu(a.wave));
          const uint32_t pida = cua;
          const uint32_t tida = (ba << 6) | wa;

          const uint32_t bb = uint32_t(b.block);
          const uint32_t wb = uint32_t(unpack_wave(b.wave));
          const uint32_t cub = uint32_t(unpack_cu(b.wave));
          const uint32_t pidb = cub;
          const uint32_t tidb = (bb << 6) | wb;

          if (pida != pidb) return pida < pidb;
          if (tida != tidb) return tida < tidb;
          if (a.ts != b.ts) return a.ts < b.ts;
          const uint32_t pa = type_prio(a.type);
          const uint32_t pb = type_prio(b.type);
          if (pa != pb) return pa < pb;
          if (a.id != b.id) return a.id < b.id;
          return false;
        });
      }

      for (const auto& e : legacy_events) {
        const uint32_t b = uint32_t(e.block);
        const uint32_t wave = uint32_t(unpack_wave(e.wave));
        const uint32_t cu = uint32_t(unpack_cu(e.wave));
        const uint32_t pid = opt.group_by_cu ? cu : b;
        const uint32_t tid = opt.group_by_cu ? ((b << 6) | wave) : (wave * 64u);
        const char* ph = (e.type == 0) ? "B" : (e.type == 1) ? "E" : "i";
        const std::string& raw_name =
            (e.id < region_names_.size()) ? region_names_[e.id] : std::to_string(e.id);
        const std::string name = json_escape(raw_name);
        const char* cname = kChromeTraceCnameList[uint32_t(e.id) % kChromeTraceCnameCount];
        const char* scope = (e.type == 2) ? ",\"s\":\"t\"" : "";

        emit("{\"name\":\"" + name + "\",\"ph\":\"" + std::string(ph) + std::string(scope) +
             "\",\"ts\":" + std::to_string((double(e.ts - min_ts) * opt.scale)) +
             ",\"pid\":" + std::to_string(pid) +
             ",\"tid\":" + std::to_string(tid) +
             ",\"cname\":\"" + std::string(cname) + "\"" +
             ",\"args\":{\"cu\":" + std::to_string(cu) +
             ",\"block\":" + std::to_string(b) +
             ",\"wave\":" + std::to_string(wave) + "}}");
      }
    } else {
      // Complete events mode: pair begin/end on host and emit ph:"X" with explicit dur.
      struct ThreadState {
        std::vector<std::vector<uint64_t>> open;
      };
      struct OutEvent {
        uint64_t ts = 0;
        uint64_t dur = 0;
        uint32_t pid = 0;
        uint32_t tid = 0;
        uint16_t id = 0;
        uint16_t kind = 0;  // 0=X, 1=i
        uint16_t block = 0;
        uint16_t wave = 0;
        uint16_t cu = 0;
        uint16_t _pad = 0;
      };

      struct WelfordAgg {
        uint64_t n = 0;
        double mean = 0.0;
        double m2 = 0.0;
        double min = 0.0;
        double max = 0.0;
        void add(double x) {
          if (n == 0) {
            n = 1; mean = x; m2 = 0.0; min = x; max = x;
            return;
          }
          ++n;
          const double delta = x - mean;
          mean += delta / double(n);
          const double delta2 = x - mean;
          m2 += delta * delta2;
          if (x < min) min = x;
          if (x > max) max = x;
        }
        double var_pop() const { return (n > 0) ? (m2 / double(n)) : 0.0; }
        double var_sample() const { return (n > 1) ? (m2 / double(n - 1)) : 0.0; }
      };

      struct RegionSummary {
        WelfordAgg global;
        std::unordered_map<uint32_t, WelfordAgg> by_block_wave;
        double hist_min = 0.0;
        double hist_max = 0.0;
        std::vector<uint64_t> hist_counts;
      };

      struct BlockRegionSummary {
        WelfordAgg agg;
        double hist_min = 0.0;
        double hist_max = 0.0;
        std::vector<uint64_t> hist_counts;
      };

      std::unordered_map<uint64_t, ThreadState> states;
      states.reserve(used_threads.size() + 16);
      std::vector<OutEvent> out_events;
      out_events.reserve(ordered.size());

      uint64_t unmatched_end = 0;
      for (const auto& e : ordered) {
        const uint32_t b = uint32_t(e.block);
        const uint32_t wave = uint32_t(unpack_wave(e.wave));
        const uint32_t cu = uint32_t(unpack_cu(e.wave));
        const uint32_t pid = opt.group_by_cu ? cu : b;
        const uint32_t tid = opt.group_by_cu ? ((b << 6) | wave) : (wave * 64u);
        const uint64_t key = (uint64_t(pid) << 32) | uint64_t(tid);
        ThreadState& st = states[key];
        if (st.open.size() <= e.id) st.open.resize(size_t(e.id) + 1);

        if (e.type == 0) {
          st.open[e.id].push_back(e.ts);
        } else if (e.type == 1) {
          auto& stack = st.open[e.id];
          if (!stack.empty()) {
            const uint64_t start_ts = stack.back();
            stack.pop_back();
            if (e.ts >= start_ts) {
              out_events.push_back(OutEvent{
                  start_ts, e.ts - start_ts, pid, tid, e.id, 0,
                  uint16_t(b), uint16_t(wave), uint16_t(cu), 0});
            }
          } else {
            ++unmatched_end;
          }
        } else {
          out_events.push_back(OutEvent{
              e.ts, 0, pid, tid, e.id, 1,
              uint16_t(b), uint16_t(wave), uint16_t(cu), 0});
        }
      }

      uint64_t unmatched_begin = 0;
      for (auto& kv : states) {
        for (auto& stack : kv.second.open) unmatched_begin += uint64_t(stack.size());
      }

      if (opt.emit_summary_json) {
        auto cv_json = [](double mean, double var_sample) -> std::string {
          const double abs_mean = std::fabs(mean);
          if (!(abs_mean > 0.0)) return "null";
          const double v = std::max(0.0, var_sample);
          return std::to_string(std::sqrt(v) / abs_mean);
        };

        auto hist_quantile = [](const std::vector<uint64_t>& counts, double lo, double hi,
                                uint64_t n, double q) -> double {
          const uint32_t nb = uint32_t(counts.size());
          if (nb == 0 || n == 0) return lo;
          if (q <= 0.0) return lo;
          if (q >= 1.0) return hi;
          const double denom = double(n);
          const double w = (hi > lo) ? ((hi - lo) / double(nb)) : 0.0;
          double c = 0.0;
          for (uint32_t i = 0; i < nb; ++i) {
            const double p = double(counts[i]) / denom;
            const double prev = c;
            c += p;
            if (c >= q) {
              const double left = lo + w * double(i);
              const double right = left + w;
              if (p <= 0.0) return left;
              double frac = (q - prev) / p;
              if (frac < 0.0) frac = 0.0;
              if (frac > 1.0) frac = 1.0;
              return left + frac * (right - left);
            }
          }
          return hi;
        };

        static const std::vector<int> kPercentileLevels = {
            5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99};

        std::unordered_map<uint32_t, RegionSummary> summaries;
        summaries.reserve(64);

        std::unordered_map<uint64_t, BlockRegionSummary> block_region;
        block_region.reserve(1024);

        for (const auto& oe : out_events) {
          if (oe.kind != 0) continue;
          const uint32_t rid = uint32_t(oe.id);
          RegionSummary& rs = summaries[rid];
          const double dur_scaled = double(oe.dur) * opt.scale;
          rs.global.add(dur_scaled);
          if (opt.summary_dump_by_block_wave) {
            const uint32_t bw = (uint32_t(oe.block) << 6) | (uint32_t(oe.wave) & 0x3Fu);
            rs.by_block_wave[bw].add(dur_scaled);
          }
          if (opt.emit_block_region_distributions) {
            const uint64_t key = (uint64_t(rid) << 32) | uint64_t(uint32_t(oe.block));
            block_region[key].agg.add(dur_scaled);
          }
        }

        const uint32_t bins = (opt.summary_hist_bins == 0) ? 128u : opt.summary_hist_bins;
        for (auto& kv : summaries) {
          RegionSummary& rs = kv.second;
          if (rs.global.n == 0) continue;
          rs.hist_min = rs.global.min;
          rs.hist_max = rs.global.max;
          rs.hist_counts.assign(bins, 0);
        }

        for (const auto& oe : out_events) {
          if (oe.kind != 0) continue;
          const uint32_t rid = uint32_t(oe.id);
          auto it = summaries.find(rid);
          if (it == summaries.end()) continue;
          RegionSummary& rs = it->second;
          if (rs.global.n == 0 || rs.hist_counts.empty()) continue;
          const double x = double(oe.dur) * opt.scale;
          const double lo = rs.hist_min;
          const double hi = rs.hist_max;
          uint32_t idx = 0;
          if (hi > lo) {
            double tf = (x - lo) / (hi - lo);
            if (tf < 0.0) tf = 0.0;
            if (tf > 1.0) tf = 1.0;
            idx = uint32_t(tf * double(rs.hist_counts.size()));
            if (idx >= rs.hist_counts.size()) idx = uint32_t(rs.hist_counts.size() - 1);
          }
          rs.hist_counts[idx] += 1;
        }

        if (opt.emit_block_region_distributions) {
          const uint32_t bbins =
              (opt.block_region_hist_bins == 0) ? 128u : opt.block_region_hist_bins;
          for (auto& kv : block_region) {
            BlockRegionSummary& br = kv.second;
            if (br.agg.n == 0) continue;
            br.hist_min = br.agg.min;
            br.hist_max = br.agg.max;
            br.hist_counts.assign(bbins, 0);
          }
          for (const auto& oe : out_events) {
            if (oe.kind != 0) continue;
            const uint32_t rid = uint32_t(oe.id);
            const uint32_t b = uint32_t(oe.block);
            const uint64_t key = (uint64_t(rid) << 32) | uint64_t(b);
            auto itb = block_region.find(key);
            if (itb == block_region.end()) continue;
            BlockRegionSummary& br = itb->second;
            if (br.agg.n == 0 || br.hist_counts.empty()) continue;
            const double x = double(oe.dur) * opt.scale;
            const double lo = br.hist_min;
            const double hi = br.hist_max;
            uint32_t idx = 0;
            if (hi > lo) {
              double tf = (x - lo) / (hi - lo);
              if (tf < 0.0) tf = 0.0;
              if (tf > 1.0) tf = 1.0;
              idx = uint32_t(tf * double(br.hist_counts.size()));
              if (idx >= br.hist_counts.size()) idx = uint32_t(br.hist_counts.size() - 1);
            }
            br.hist_counts[idx] += 1;
          }
        }

        const std::string summary_path = strip_json_suffix(path) + "_summary.json";
        ensure_parent_dir_for_file(summary_path);
        std::ofstream sout(summary_path);
        sout << "{\n";
        sout << "  \"trace\": \"" << json_escape(path) << "\",\n";
        sout << "  \"displayTimeUnit\": \"ns\",\n";
        sout << "  \"scale\": " << std::to_string(opt.scale) << ",\n";
        sout << "  \"emit_complete_events\": " << (opt.emit_complete_events ? 1 : 0) << ",\n";
        sout << "  \"group_by_cu\": " << (opt.group_by_cu ? 1 : 0) << ",\n";
        sout << "  \"blocks\": " << blocks_ << ",\n";
        sout << "  \"waves_per_block\": " << waves_per_block_ << ",\n";
        sout << "  \"per_wave_cap\": " << per_wave_cap_ << ",\n";
        sout << "  \"summary_filters\": {"
             << "\"summary_hist_bins\": " << opt.summary_hist_bins
             << ", \"summary_dump_by_block_wave\": " << (opt.summary_dump_by_block_wave ? 1 : 0)
             << ", \"summary_topk_block_wave_per_region\": "
             << opt.summary_topk_block_wave_per_region
             << ", \"emit_block_region_distributions\": "
             << (opt.emit_block_region_distributions ? 1 : 0)
             << ", \"block_region_hist_bins\": " << opt.block_region_hist_bins
             << ", \"block_region_topk_blocks\": " << opt.block_region_topk_blocks
             << ", \"block_region_topk_regions_per_block\": "
             << opt.block_region_topk_regions_per_block << "},\n";
        sout << "  \"block_filter\": [";
        if (!block_filter_.empty()) {
          std::vector<uint32_t> bf(block_filter_.begin(), block_filter_.end());
          std::sort(bf.begin(), bf.end());
          for (size_t i = 0; i < bf.size(); ++i) {
            sout << bf[i];
            if (i + 1 != bf.size()) sout << ", ";
          }
        }
        sout << "],\n";
        sout << "  \"unmatched_begin\": " << (unsigned long long)unmatched_begin << ",\n";
        sout << "  \"unmatched_end\": " << (unsigned long long)unmatched_end << ",\n";
        sout << "  \"regions\": [\n";

        auto sanitize_key = [&](const std::string& s) -> std::string {
          std::string o;
          o.reserve(s.size() + 8);
          for (unsigned char c : s) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                            (c >= '0' && c <= '9') || (c == '_');
            o.push_back(ok ? static_cast<char>(c) : '_');
          }
          std::string compact;
          compact.reserve(o.size());
          bool last_us = false;
          for (char c : o) {
            if (c == '_') {
              if (!last_us) compact.push_back(c);
              last_us = true;
            } else {
              compact.push_back(c);
              last_us = false;
            }
          }
          size_t b = 0;
          while (b < compact.size() && compact[b] == '_') ++b;
          size_t e = compact.size();
          while (e > b && compact[e - 1] == '_') --e;
          std::string trimmed = compact.substr(b, e - b);
          return trimmed.empty() ? std::string("region") : trimmed;
        };

        std::vector<uint32_t> region_ids;
        region_ids.reserve(summaries.size());
        for (const auto& kv : summaries) region_ids.push_back(kv.first);
        std::sort(region_ids.begin(), region_ids.end());

        bool first_r = true;
        for (uint32_t rid : region_ids) {
          const auto it = summaries.find(rid);
          if (it == summaries.end()) continue;
          const RegionSummary& rs = it->second;
          if (rs.global.n == 0) continue;
          if (!first_r) sout << ",\n";
          first_r = false;

          const std::string raw_name =
              (rid < region_names_.size()) ? region_names_[rid] : std::to_string(rid);
          const std::string name = json_escape(raw_name);
          const double mean = rs.global.mean;
          const double var_pop = rs.global.var_pop();
          const double var_sample = rs.global.var_sample();

          sout << "    {\"region\": " << rid << ", \"name\": \"" << name << "\""
               << ", \"count\": " << (unsigned long long)rs.global.n
               << ", \"mean_dur\": " << std::to_string(mean)
               << ", \"cv_dur\": " << cv_json(mean, var_sample);

          sout << ", \"percentiles\": {";
          for (size_t pi = 0; pi < kPercentileLevels.size(); ++pi) {
            const double q = double(kPercentileLevels[pi]) / 100.0;
            const double val =
                hist_quantile(rs.hist_counts, rs.hist_min, rs.hist_max, rs.global.n, q);
            sout << "\"p" << kPercentileLevels[pi] << "\": " << std::to_string(val);
            if (pi + 1 != kPercentileLevels.size()) sout << ", ";
          }
          sout << "}";

          sout << ", \"var_dur_pop\": " << std::to_string(var_pop)
               << ", \"var_dur_sample\": " << std::to_string(var_sample)
               << ", \"min_dur\": " << std::to_string(rs.global.min)
               << ", \"max_dur\": " << std::to_string(rs.global.max);

          const uint32_t nb = uint32_t(rs.hist_counts.size());
          sout << ", \"hist\": {\"bins\": " << nb
               << ", \"min\": " << std::to_string(rs.hist_min)
               << ", \"max\": " << std::to_string(rs.hist_max) << ", \"prob\": [";
          const double denom = (rs.global.n > 0) ? double(rs.global.n) : 1.0;
          for (uint32_t i = 0; i < nb; ++i) {
            const double p = double(rs.hist_counts[i]) / denom;
            sout << std::to_string(p);
            if (i + 1 != nb) sout << ", ";
          }
          sout << "]}";

          sout << "}";
        }
        sout << "\n  ],\n";

        if (opt.summary_dump_by_block_wave) {
          sout << "  \"by_block_wave_regions\": {\n";
          bool first_k = true;
          std::unordered_set<std::string> used_keys;
          used_keys.reserve(region_ids.size() * 2);
          for (uint32_t rid : region_ids) {
            const auto it = summaries.find(rid);
            if (it == summaries.end()) continue;
            const RegionSummary& rs = it->second;
            if (rs.global.n == 0) continue;
            if (rs.by_block_wave.empty()) continue;

            const std::string raw_name =
                (rid < region_names_.size()) ? region_names_[rid] : std::to_string(rid);
            std::string key = "region_" + sanitize_key(raw_name);
            if (used_keys.find(key) != used_keys.end()) key += "_" + std::to_string(rid);
            used_keys.insert(key);

            struct BwEntry {
              uint32_t key = 0;
              const WelfordAgg* agg = nullptr;
            };
            std::vector<BwEntry> entries;
            entries.reserve(rs.by_block_wave.size());
            for (const auto& kv2 : rs.by_block_wave)
              entries.push_back(BwEntry{kv2.first, &kv2.second});
            std::sort(entries.begin(), entries.end(), [](const BwEntry& a, const BwEntry& b) {
              const uint64_t na = a.agg ? a.agg->n : 0;
              const uint64_t nb = b.agg ? b.agg->n : 0;
              if (na != nb) return na > nb;
              return a.key < b.key;
            });
            const size_t n_emit =
                (opt.summary_topk_block_wave_per_region == 0)
                    ? entries.size()
                    : std::min<size_t>(entries.size(),
                                       size_t(opt.summary_topk_block_wave_per_region));

            if (!first_k) sout << ",\n";
            first_k = false;
            sout << "    \"" << json_escape(key) << "\": {\n";
            sout << "      \"region\": " << rid << ",\n";
            sout << "      \"name\": \"" << json_escape(raw_name) << "\",\n";
            sout << "      \"by_block_wave\": [\n";
            for (size_t i = 0; i < n_emit; ++i) {
              const uint32_t bw = entries[i].key;
              const uint32_t b = bw >> 6;
              const uint32_t w = bw & 0x3Fu;
              const WelfordAgg& a = *entries[i].agg;
              const double vs = a.var_sample();
              sout << "        {\"block\": " << b << ", \"wave\": " << w
                   << ", \"count\": " << (unsigned long long)a.n
                   << ", \"mean_dur\": " << std::to_string(a.mean)
                   << ", \"cv_dur\": " << cv_json(a.mean, vs)
                   << ", \"var_dur_pop\": " << std::to_string(a.var_pop())
                   << ", \"var_dur_sample\": " << std::to_string(vs)
                   << ", \"min_dur\": " << std::to_string(a.min)
                   << ", \"max_dur\": " << std::to_string(a.max) << "}";
              if (i + 1 != n_emit) sout << ",\n";
            }
            if (n_emit) sout << "\n";
            sout << "      ]\n";
            sout << "    }";
          }
          sout << "\n  }\n";
        } else {
          sout << "  \"by_block_wave_regions\": null\n";
        }

        sout << "}\n";
        std::printf("rocprof_intrakernel::trace: summary -> %s\n", summary_path.c_str());
      }

      // Sort output events for contiguous thread output.
      std::sort(out_events.begin(), out_events.end(),
                [](const OutEvent& a, const OutEvent& b) {
                  if (a.pid != b.pid) return a.pid < b.pid;
                  if (a.tid != b.tid) return a.tid < b.tid;
                  if (a.ts != b.ts) return a.ts < b.ts;
                  if (a.dur != b.dur) return a.dur > b.dur;
                  return a.id < b.id;
                });

      for (const auto& oe : out_events) {
        const std::string& raw_name =
            (oe.id < region_names_.size()) ? region_names_[oe.id] : std::to_string(oe.id);
        const std::string name = json_escape(raw_name);
        const char* cname = kChromeTraceCnameList[uint32_t(oe.id) % kChromeTraceCnameCount];
        const double ts_out = double(oe.ts - min_ts) * opt.scale;

        if (oe.kind == 0) {
          const double dur_out = double(oe.dur) * opt.scale;
          emit("{\"name\":\"" + name + "\",\"ph\":\"X\"" +
               ",\"ts\":" + std::to_string(ts_out) +
               ",\"dur\":" + std::to_string(dur_out) +
               ",\"pid\":" + std::to_string(oe.pid) +
               ",\"tid\":" + std::to_string(oe.tid) +
               ",\"cname\":\"" + std::string(cname) + "\"" +
               ",\"args\":{\"cu\":" + std::to_string(oe.cu) +
               ",\"block\":" + std::to_string(oe.block) +
               ",\"wave\":" + std::to_string(oe.wave) + "}}");
        } else {
          emit("{\"name\":\"" + name + "\",\"ph\":\"i\",\"s\":\"t\"" +
               ",\"ts\":" + std::to_string(ts_out) +
               ",\"pid\":" + std::to_string(oe.pid) +
               ",\"tid\":" + std::to_string(oe.tid) +
               ",\"cname\":\"" + std::string(cname) + "\"" +
               ",\"args\":{\"cu\":" + std::to_string(oe.cu) +
               ",\"block\":" + std::to_string(oe.block) +
               ",\"wave\":" + std::to_string(oe.wave) + "}}");
        }
      }

      if (unmatched_end) {
        std::fprintf(
            stderr,
            "rocprof_intrakernel::trace: warning: %llu unmatched end events ignored\n",
            (unsigned long long)unmatched_end);
      }
      if (unmatched_begin) {
        std::fprintf(
            stderr,
            "rocprof_intrakernel::trace: warning: %llu unmatched begin events ignored\n",
            (unsigned long long)unmatched_begin);
      }
    }

    out << "\n]}\n";
    std::printf("rocprof_intrakernel::trace: %zu events -> %s\n", ordered.size(), path.c_str());
  }

  void destroy() {
    if (d_events_) hipFree(d_events_);
    if (d_counters_) hipFree(d_counters_);
    d_events_ = nullptr;
    d_counters_ = nullptr;
    gbuf_ = GlobalBuffer{};
    per_wave_cap_ = 0;
    waves_per_block_ = 0;
    blocks_ = 0;
  }

 private:
  GlobalBuffer gbuf_{};
  Event* d_events_ = nullptr;
  uint32_t* d_counters_ = nullptr;
  uint32_t per_wave_cap_ = 0;
  uint32_t waves_per_block_ = 0;
  uint32_t blocks_ = 0;
  std::vector<std::string> region_names_;
  std::unordered_set<uint32_t> block_filter_;
};

}  // namespace rocprof_intrakernel::trace
