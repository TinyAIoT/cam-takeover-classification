# Example Output

After the flashing you should see the output at idf monitor:

```
I (1458) dl::Model: model:main_graph, version:0

I (1458) dl::Model: /features/features.0/features.0.0/Conv: Conv
I (1468) dl::Model: /features/features.1/conv/conv.0/conv.0.0/Conv: Conv
I (1478) dl::Model: /features/features.1/conv/conv.1/Conv: Conv
I (1478) dl::Model: /features/features.2/conv/conv.0/conv.0.0/Conv: Conv
I (1488) dl::Model: /features/features.2/conv/conv.1/conv.1.0/Conv: Conv
I (1498) dl::Model: /features/features.2/conv/conv.2/Conv: Conv
I (1508) dl::Model: /features/features.3/conv/conv.0/conv.0.0/Conv: Conv
I (1508) dl::Model: /features/features.3/conv/conv.1/conv.1.0/Conv: Conv
I (1518) dl::Model: /features/features.3/conv/conv.2/Conv: Conv
I (1528) dl::Model: /features/features.3/Add: Add
I (1528) dl::Model: /features/features.4/conv/conv.0/conv.0.0/Conv: Conv
I (1538) dl::Model: /features/features.4/conv/conv.1/conv.1.0/Conv: Conv
I (1548) dl::Model: /features/features.4/conv/conv.2/Conv: Conv
I (1548) dl::Model: /features/features.5/conv/conv.0/conv.0.0/Conv: Conv
I (1558) dl::Model: PPQ_Operation_0: RequantizeLinear
I (1568) dl::Model: /features/features.5/conv/conv.1/conv.1.0/Conv: Conv
I (1568) dl::Model: /features/features.5/conv/conv.2/Conv: Conv
I (1578) dl::Model: /features/features.5/Add: Add
I (1588) dl::Model: /features/features.6/conv/conv.0/conv.0.0/Conv: Conv
I (1588) dl::Model: PPQ_Operation_1: RequantizeLinear
I (1598) dl::Model: /features/features.6/conv/conv.1/conv.1.0/Conv: Conv
I (1608) dl::Model: /features/features.6/conv/conv.2/Conv: Conv
I (1608) dl::Model: /features/features.6/Add: Add
I (1618) dl::Model: /features/features.7/conv/conv.0/conv.0.0/Conv: Conv
I (1628) dl::Model: /features/features.7/conv/conv.1/conv.1.0/Conv: Conv
I (1628) dl::Model: /features/features.7/conv/conv.2/Conv: Conv
I (1638) dl::Model: /features/features.8/conv/conv.0/conv.0.0/Conv: Conv
I (1648) dl::Model: /features/features.8/conv/conv.1/conv.1.0/Conv: Conv
I (1658) dl::Model: /features/features.8/conv/conv.2/Conv: Conv
I (1658) dl::Model: /features/features.8/Add: Add
I (1668) dl::Model: /features/features.9/conv/conv.0/conv.0.0/Conv: Conv
I (1678) dl::Model: /features/features.9/conv/conv.1/conv.1.0/Conv: Conv
I (1678) dl::Model: /features/features.9/conv/conv.2/Conv: Conv
I (1688) dl::Model: /features/features.9/Add: Add
I (1688) dl::Model: /features/features.10/conv/conv.0/conv.0.0/Conv: Conv
I (1698) dl::Model: /features/features.10/conv/conv.1/conv.1.0/Conv: Conv
I (1708) dl::Model: /features/features.10/conv/conv.2/Conv: Conv
I (1718) dl::Model: /features/features.10/Add: Add
I (1718) dl::Model: /features/features.11/conv/conv.0/conv.0.0/Conv: Conv
I (1728) dl::Model: /features/features.11/conv/conv.1/conv.1.0/Conv: Conv
I (1738) dl::Model: /features/features.11/conv/conv.2/Conv: Conv
I (1738) dl::Model: /features/features.12/conv/conv.0/conv.0.0/Conv: Conv
I (1748) dl::Model: /features/features.12/conv/conv.1/conv.1.0/Conv: Conv
I (1758) dl::Model: /features/features.12/conv/conv.2/Conv: Conv
I (1768) dl::Model: /features/features.12/Add: Add
I (1768) dl::Model: /features/features.13/conv/conv.0/conv.0.0/Conv: Conv
I (1778) dl::Model: PPQ_Operation_2: RequantizeLinear
I (1778) dl::Model: /features/features.13/conv/conv.1/conv.1.0/Conv: Conv
I (1788) dl::Model: /features/features.13/conv/conv.2/Conv: Conv
I (1798) dl::Model: /features/features.13/Add: Add
I (1798) dl::Model: /features/features.14/conv/conv.0/conv.0.0/Conv: Conv
I (1808) dl::Model: /features/features.14/conv/conv.1/conv.1.0/Conv: Conv
I (1818) dl::Model: /features/features.14/conv/conv.2/Conv: Conv
I (1828) dl::Model: /features/features.15/conv/conv.0/conv.0.0/Conv: Conv
I (1838) dl::Model: /features/features.15/conv/conv.1/conv.1.0/Conv: Conv
I (1838) dl::Model: /features/features.15/conv/conv.2/Conv: Conv
I (1848) dl::Model: /features/features.15/Add: Add
I (1848) dl::Model: /features/features.16/conv/conv.0/conv.0.0/Conv: Conv
I (1868) dl::Model: /features/features.16/conv/conv.1/conv.1.0/Conv: Conv
I (1868) dl::Model: /features/features.16/conv/conv.2/Conv: Conv
I (1878) dl::Model: /features/features.16/Add: Add
I (1878) dl::Model: /features/features.17/conv/conv.0/conv.0.0/Conv: Conv
I (1888) dl::Model: /features/features.17/conv/conv.1/conv.1.0/Conv: Conv
I (1888) dl::Model: /features/features.17/conv/conv.2/Conv: Conv
I (1918) dl::Model: /features/features.18/features.18.0/Conv: Conv
I (1938) dl::Model: /GlobalAveragePool: GlobalAveragePool
I (1938) dl::Model: PPQ_Operation_3: Transpose
I (1938) dl::Model: /Flatten: Flatten
I (1938) dl::Model: /classifier/classifier.1/Gemm: Gemm
I (1958) MemoryManagerGreedy: Maximum mermory size: 1705984

Inference in  1298812 us.I (3298) bird_cls: category: pica_pica, score: 0.145075

I (3298) bird_cls: category: erithacus_rubecula, score: 0.112984

I (3308) bird_cls: category: sciurus_vulgaris, score: 0.093667

I (3308) bird_cls: category: passer_domesticus, score: 0.093667

I (3318) bird_cls: category: cyanistes_caeruleus, score: 0.082661

I (3328) bird_cls: Best: pica_pica (score: 0.145075)
I (3328) MEM: Free heap at end of loop: 5044884 bytes
I (3338) main_task: Returned from app_main()
```