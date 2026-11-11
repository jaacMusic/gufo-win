#ifndef GUFO_CORE_POOL_H
#define GUFO_CORE_POOL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/* enum for pool_pin return values */
typedef enum PoolPin
{
  POOL_OK = 0,
  POOL_WAIT = 1,
  POOL_FULL = -1,
} PoolPin;

/* per-expert memory unit: three parts (gate, up, down FF) from the model files.
   each part is read from a file range, aligned to POOL_ALIGN. */
#define POOL_ALIGN 256

typedef enum PoolUnitState
{
  POOL_COLD = 0,     /* unit is in its files, not in GPU memory */
  POOL_READING,      /* read is in progress into GPU memory */
  POOL_PINNED        /* unit memory is resident in GPU, GPU can read it */
} PoolUnitState;

/* a unit's read range from one file: offset + bytes, aligned. */
typedef struct PoolRange
{
  uint64_t offset;
  uint64_t bytes;
} PoolRange;

/* one unit of expert memory (gate + up + down matrices), with up to
   POOL_PARTS read ranges laid back-to-back, each aligned. */
#define POOL_PARTS 3

typedef struct PoolUnit
{
  PoolRange part[POOL_PARTS];
  int n_parts;
  PoolUnitState state;
} PoolUnit;

/* memory-pool policy: units, budget, and the allocator / fetcher.
   the pool manages a budget of GPU memory for resident experts;
   when the budget is full, the least-wanted unit goes cold. */
typedef struct Pool
{
  PoolUnit *units;      /* array of n_units */
  size_t n_units;
  uint64_t budget;      /* total GPU bytes may hold */
  uint64_t spare;       /* bytes free beyond units' own memory */
  uint64_t read_align;  /* alignment for part reads, power of 2 */
  int *wanted_clock;    /* circular clock for LRU: wanted[0] is soonest */
  size_t wanted_pos;    /* next slot to advance */
} Pool;

/* create a pool of expert units with a GPU memory budget.
   units: array of PoolUnit describing each unit's file ranges.
   n_units: number of units.
   budget: total GPU memory (bytes) the pool may hold.
   alloc / free: GPU memory operations (may be NULL for host-side only).
   read_align: alignment for part reads (0 = no alignment).
   returns NULL with reason in err if allocation fails. */
Pool *pool_create(PoolUnit *units, size_t n_units, uint64_t budget,
                  void (*alloc)(uint64_t bytes), void (*free)(void *dev),
                  uint64_t read_align, char err[256]);

/* destroy a pool, freeing all unit memory. */
void pool_destroy(Pool *p);

/* pin a unit: its memory stays resident until released.
   returns POOL_OK if pinned, POOL_WAIT if cold/reading (starts read),
   POOL_FULL if no room even evicting wanted units. */
int pool_pin(Pool *p, size_t unit_idx);

/* release one hold on a pinned unit.
   `next`: when the caller expects to want it again (its clock position). */
void pool_release(Pool *p, size_t unit_idx, int next);

/* mark a unit as wanted at clock position `next`.
   starts/corrects its read if cold. */
void pool_want(Pool *p, size_t unit_idx, int next);

/* check if a unit is currently pinned and its device pointers are valid. */
bool pool_dev(const Pool *p, size_t unit_idx, const void *dev[3]);

/* advance the internal clock: called after a unit is released so the LRU
   order reflects recent usage. the forward's layers come round in a fixed
   order, so this keeps the recency that matters for eviction. */
void pool_now(Pool *p);

/* return the current state of one unit. */
PoolUnitState pool_state(const Pool *p, size_t unit_idx);

/* return total bytes of GPU memory held by the pool (units + spares). */
uint64_t pool_bytes(const Pool *p);

/* return the number of units currently reading. */
size_t pool_reading(const Pool *p);

#endif