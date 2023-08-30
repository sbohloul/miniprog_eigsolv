# Performance analysis

## Number of blocks per process

In this analysis, block size `mb x nb` and process grid `p x q` are fixed, and global matrix size `m x n` is scaled in a way that the of blocks per process increases homogeneously. Parameters selection enforces ideal load-balancing.

**Important parmeters:**

- `m = (nprow * mb ) * scale_factor`
- `n = (nprow * nb ) * scale_factor`

The following code snippet systematically increases number of blocks per process by, from `1` to `(max_size_scaling - 1)` by increasing the matrix size, accordingly:

```python
max_size_scaling = 9
for i in range(1, max_size_scaling):
    num_mblocks = i
    num_nblocks = i
    num_blocks_per_proc.append(num_mblocks * num_nblocks)
    m = (nprow * mb) * num_mblocks
    n = (npcol * nb) * num_nblocks

    # ...
```

**Simple performance model:**

Assuming

- Time for local operation on each `mb x nb` data block as `t_oper`
- Time for communication when each process holds only one block as `t_comm`
- Each process holds `N_block` blocks

Then for ideal linear behavior

$$ 
T_{tot} = \sum_{i}^{N_{proc}} (N_{b}t_{i, oper} + N_{b}t_{i, comm}) = N_{b} \sum_{i}^{N_{proc}} (t_{i, oper} + t_{i, comm}) = N_{b}N_{proc}(t_{oper} + t_{comm})
$$

Following factors weaken the linear behavior

- Communication overhead as `N_block` increases

**How to run the benchmark**
