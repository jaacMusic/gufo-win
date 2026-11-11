#include "src/core/pool.h"

#include <string.h>

/* internal: advance the wanted clock one step (LRU recency). */
static void advance_clock(Pool *p)
{
  p->wanted_pos = (p->wanted_pos + 1) % p->n_units;
}

/* internal: find the unit with the largest clock value (least recently wanted).
   a higher clock means it was wanted longer ago, so it's the best candidate
   for eviction when the budget is full. */
static size_t find_eviction_candidate(const Pool *p)
{
  size_t worst = 0;
  for (size_t i = 1; i < p->n_units; ++i) {
    if (p->wanted_clock[i] > p->wanted_clock[worst]) {
      worst = i;
    }
  }
  return worst;
}

Pool *pool_create(PoolUnit *units, size_t n_units, uint64_t budget,
                  void (*alloc)(uint64_t bytes), void (*free)(void *dev),
                  uint64_t read_align, char err[256])
{
  (void)alloc; (void)free; (void)read_align; /* reserved for future use */
  (void)err;

  Pool *p = (Pool *)calloc(1, sizeof(Pool));
  if (!p) {
    return NULL;
  }

  p->units = units;
  p->n_units = n_units;
  p->budget = budget;
  p->spare = budget;  /* start with full budget as spare */
  p->read_align = read_align ? read_align : POOL_ALIGN;
  p->wanted_clock = (int *)calloc(n_units, sizeof(int));
  if (!p->wanted_clock) {
    free(p);
    return NULL;
  }

  /* initialise each unit as COLD with want=0 */
  for (size_t i = 0; i < n_units; ++i) {
    units[i].state = POOL_COLD;
    units[i].n_parts = 0;
    p->wanted_clock[i] = 0;
  }

  p->wanted_pos = 0;
  return p;
}

void pool_destroy(Pool *p)
{
  if (!p) return;
  free(p->wanted_clock);
  /* spare: caller frees unit memories */
  free(p);
}

int pool_pin(Pool *p, size_t unit_idx)
{
  if (!p || unit_idx >= p->n_units) return POOL_FULL;

  PoolUnit *u = &p->units[unit_idx];

  /* if already pinned, nothing to do */
  if (u->state == POOL_PINNED) return POOL_OK;

  /* if reading, wait for it to complete (caller arranges pool_ready) */
  if (u->state == POOL_READING) return POOL_WAIT;

  /* if cold, we need budget. evict the least-wanted unit if full. */
  if (u->state == POOL_COLD) {
    /* check if we have spare budget for this unit's memory.
       for now, assume each unit's memory is its parts' bytes rounded up. */
    uint64_t unit_bytes = 0;
    for (int j = 0; j < u->n_parts; ++j) {
      unit_bytes += u->part[j].bytes;
    }
    /* round up to alignment */
    if (unit_bytes % p->read_align) {
      unit_bytes += p->read_align - (unit_bytes % p->read_align);
    }

    if (p->spare < unit_bytes) {
      /* evict the least-wanted unit */
      size_t evict = find_eviction_candidate(p);
      PoolUnit *evict_unit = &p->units[evict];
      /* release it: go cold */
      evict_unit->state = POOL_COLD;
      for (int j = 0; j < evict_unit->n_parts; ++j) {
        /* conceptually free the memory */
      }
      p->spare += /* evicted unit's bytes */ 0;  /* simplified */
    }

    u->state = POOL_PINNED;
    p->spare -= /* this unit's bytes */ 0;  /* simplified */
  }

  /* mark this unit as most-recently wanted */
  p->wanted_clock[unit_idx] = p->wanted_pos;
  advance_clock(p);

  return POOL_OK;
}

void pool_release(Pool *p, size_t unit_idx, int next)
{
  if (!p || unit_idx >= p->n_units) return;

  PoolUnit *u = &p->units[unit_idx];

  /* release the hold: go cold */
  u->state = POOL_COLD;

  /* update the wanted clock: this unit was wanted at `next` */
  (void)next;  /* in full implementation, correct the clock */

  /* we now have spare budget */
  p->spare += /* this unit's bytes */ 0;  /* simplified */

  /* mark as most-recently wanted */
  p->wanted_clock[unit_idx] = p->wanted_pos;
  advance_clock(p);
}

void pool_want(Pool *p, size_t unit_idx, int next)
{
  if (!p || unit_idx >= p->n_units) return;

  /* mark the unit as wanted at the caller's clock position */
  p->wanted_clock[unit_idx] = next;

  /* if cold, start a read (conceptually) */
  if (p->units[unit_idx].state == POOL_COLD) {
    /* in full implementation, initiate fetch from file into GPU memory */
  }

  /* advance the clock */
  /* note: pool_now() is typically called after release to update recency */
}

bool pool_dev(const Pool *p, size_t unit_idx, const void *dev[3])
{
  if (!p || unit_idx >= p->n_units) return false;
  /* in a full implementation, return the device pointers for a pinned unit */
  (void)dev;
  return p->units[unit_idx].state == POOL_PINNED;
}

void pool_now(Pool *p)
{
  if (!p) return;
  /* advance the wanted clock to reflect recent activity */
  advance_clock(p);
}

PoolUnitState pool_state(const Pool *p, size_t unit_idx)
{
  if (!p || unit_idx >= p->n_units) return POOL_COLD;
  return p->units[unit_idx].state;
}

uint64_t pool_bytes(const Pool *p)
{
  if (!p) return 0;
  /* simplified: return spare as the free portion of the budget */
  return p->spare;
}

size_t pool_reading(const Pool *p)
{
  if (!p) return 0;
  size_t count = 0;
  for (size_t i = 0; i < p->n_units; ++i) {
    if (p->units[i].state == POOL_READING) ++count;
  }
  return count;
}