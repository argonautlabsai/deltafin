#include <metal_stdlib>
using namespace metal;

struct DeltafinRouteMailboxT1 {
  long expert_ids[16];
  uint weight_bits[16];
};

kernel void deltafin_route_mailbox_t1(
    device const long* expert_ids [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device DeltafinRouteMailboxT1* output [[buffer(2)]],
    uint edge [[thread_position_in_grid]]) {
  if (edge < 16) {
    output->expert_ids[edge] = expert_ids[edge];
    output->weight_bits[edge] = as_type<uint>(weights[edge]);
  }
}

// Pilot-hint mailbox (sync-E removal): copies one pilot prediction's ids and
// scores into host-shared memory so the consumer never issues a .to(kCPU)
// stream drain. Sized for the pilot maxima (64 positions x width 32); the
// header is written by thread 0 and the publisher's event signal orders the
// whole struct before any host read.
struct DeltafinPilotMailboxHeader {
  uint layer_index;
  uint expert_count;
  uint position_count;
  uint width;
};

struct DeltafinPilotMailboxRows {
  DeltafinPilotMailboxHeader header;
  long expert_ids[64 * 32];
  float choice_scores[64 * 32];
};

kernel void deltafin_pilot_mailbox_rows(
    device const long* expert_ids [[buffer(0)]],
    device const float* choice_scores [[buffer(1)]],
    device DeltafinPilotMailboxRows* output [[buffer(2)]],
    constant DeltafinPilotMailboxHeader& header [[buffer(3)]],
    uint slot [[thread_position_in_grid]]) {
  const uint total = header.position_count * header.width;
  if (slot < total) {
    output->expert_ids[slot] = expert_ids[slot];
    output->choice_scores[slot] = choice_scores[slot];
  }
  if (slot == 0) {
    output->header = header;
  }
}
