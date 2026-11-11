#include "src/core/pool.h"
#include <stdio.h>
#include <assert.h>

int main(void)
{
  /* test: create a pool with 3 units, budget of 1024*1024 bytes */
  PoolUnit units[3];

  /* unit 0: one part, 256 bytes */
  units[0].part[0].offset = 0;
  units[0].part[0].bytes = 256;
  units[0].n_parts = 1;
  /* unit 1: one part, 512 bytes */
  units[1].part[0].offset = 0;
  units[1].part[0].bytes = 512;
  units[1].n_parts = 1;
  /* unit 2: one part, 128 bytes */
  units[2].part[0].offset = 0;
  units[2].part[0].bytes = 128;
  units[2].n_parts = 1;

  char err[256];
  Pool *p = pool_create(units, 3, 1024 * 1024, NULL, NULL, 0, err);
  assert(p != NULL && "pool_create should succeed");

  /* test: pin all units (should succeed with budget 1MB >> 896 bytes total) */
  int r0 = pool_pin(p, 0);
  assert(r0 == 0 && "pool_pin unit 0 should succeed"); /* 0 = POOL_OK */

  int r1 = pool_pin(p, 1);
  assert(r1 == 0 && "pool_pin unit 1 should succeed");

  int r2 = pool_pin(p, 2);
  assert(r2 == 0 && "pool_pin unit 2 should succeed");

  /* test: states are PINNED */
  assert(pool_state(p, 0) == POOL_PINNED && "unit 0 should be PINNED");
  assert(pool_state(p, 1) == POOL_PINNED && "unit 1 should be PINNED");
  assert(pool_state(p, 2) == POOL_PINNED && "unit 2 should be PINNED");

  /* test: reading count is 0 (no reads in progress) */
  assert(pool_reading(p) == 0 && "no reads should be in progress");

  /* test: pool_bytes reports some free bytes */
  uint64_t bytes = pool_bytes(p);
  assert(bytes > 0 && "pool_bytes should be positive");

  /* test: release unit 1 */
  pool_release(p, 1, 0);

  /* test: unit 1 is now COLD */
  assert(pool_state(p, 1) == POOL_COLD && "unit 1 should be COLD after release");

  /* test: pool_reading still 0 */
  assert(pool_reading(p) == 0 && "no reads should still be in progress");

  /* test: want unit 1 again */
  pool_want(p, 1, 0);

  /* test: pool_now advances clock */
  pool_now(p);

  /* test: destroy pool */
  pool_destroy(p);

  printf("All pool tests passed!\n");
  return 0;
}