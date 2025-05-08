# Long Sequence Performance

- The table below shows the pre-training performance of the LLAMA2-7B and LLAMA3-8B models on H100 and B200 GPUs, respectively, with CP (context parallelism), 
and compares it against the results without CP at various input sequence lengths. 
The detailed model-parallel configurations and the achieved performance are shown in the training results with CP. 
In non-CP training runs, we use the most performant model- and data-parallel configurations without CP given the memory capacity constraint of the each GPU system.

## LLAMA3-8B (FP8) - B200

  - Container: [NeMo25.04.rc2](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)
  - System: DGX-B200

<table>
  <thead>
    <tr>
      <th rowspan="2" class="top-border">SeqLen (K)</th>
      <th rowspan="2" class="top-border"># of GPUs</th>
      <th rowspan="2" class="top-border">Batch Size</th>
      <th rowspan="1" class="top-border">Without CP</th>
      <th colspan="5" class="top-border">With CP</th>
      <th rowspan="2" class="top-border">Speedup with CP/without CP</th>
    </tr>
    <tr>
      <th>TFLOPS / GPU</th>
      <th>TP</th>
      <th>PP</th>
      <th>DP</th>
      <th>CP</th>
      <th>TFLOPS / GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8</td>
      <td>8</td>
      <td>512</td>
      <td>1,671</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1,671</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>16</td>
      <td>256</td>
      <td>1,717</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1,717</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>32</td>
      <td>32</td>
      <td>128</td>
      <td>1,549</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>1,624</td>
      <td class="speedup">1.05</td>
    </tr>
    <tr>
      <td>64</td>
      <td>64</td>
      <td>64</td>
      <td>1,481</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1,600</td>
      <td class="speedup">1.08</td>
    </tr>
    <tr>
      <td>128</td>
      <td>128</td>
      <td>32</td>
      <td>1,438</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1,588</td>
      <td class="speedup">1.10</td>
    </tr>
    <tr>
      <td>256</td>
      <td>256</td>
      <td>16</td>
      <td>1,162</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>1,590</td>
      <td class="speedup">1.37</td>
    </tr>
    <tr>
      <td>512</td>
      <td>512</td>
      <td>8</td>
      <td>607</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>8</td>
      <td>1,619</td>
      <td class="speedup">2.67</td>
    </tr>
    <tr>
      <td>1024</td>
      <td>1024</td>
      <td>4</td>
      <td>-<sup>1)</sup></td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>16</td>
      <td>1,608</td>
      <td class="speedup">-</td>
    </tr>
  </tbody>
</table>

<sup> 1) Since the maximum TP size is limited by the number of query groups (8 in LLAMA3-8B), 
even with full activation recomputation it is impossible to run the LLAMA3-8B model on a 1024K token sequence without CP due to the GPU memory constraints.</sup>

## LLAMA2-7B (FP8) - H100

  - Container: [NeMo24.03.01.framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags)
  - System: DGX-H100


<table>
  <thead>
    <tr>
      <th rowspan="2" class="top-border">SeqLen (K)</th>
      <th rowspan="2" class="top-border"># of GPUs</th>
      <th rowspan="2" class="top-border">Batch Size</th>
      <th rowspan="1" class="top-border">Without CP</th>
      <th colspan="5" class="top-border">With CP</th>
      <th rowspan="2" class="top-border">Speedup with CP/without CP</th>
    </tr>
    <tr>
      <th>TFLOPS / GPU</th>
      <th>TP</th>
      <th>PP</th>
      <th>DP</th>
      <th>CP</th>
      <th>TFLOPS / GPU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>1024</td>
      <td>768</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>768</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>8</td>
      <td>512</td>
      <td>730</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>730</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>16</td>
      <td>256</td>
      <td>660</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>660</td>
      <td class="speedup">1.00</td>
    </tr>
    <tr>
      <td>32</td>
      <td>32</td>
      <td>128</td>
      <td>595</td>
      <td>2</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>610</td>
      <td class="speedup">1.03</td>
    </tr>
    <tr>
      <td>64</td>
      <td>64</td>
      <td>64</td>
      <td>534</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>2</td>
      <td>574</td>
      <td class="speedup">1.07</td>
    </tr>
    <tr>
      <td>128</td>
      <td>128</td>
      <td>32</td>
      <td>424</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>4</td>
      <td>555</td>
      <td class="speedup">1.31</td>
    </tr>
    <tr>
      <td>256</td>
      <td>256</td>
      <td>16</td>
      <td>392</td>
      <td>4</td>
      <td>1</td>
      <td>8</td>
      <td>8</td>
      <td>549</td>
      <td class="speedup">1.40</td>
    </tr>
    <tr>
      <td>512</td>
      <td>512</td>
      <td>8</td>
      <td>104</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>16</td>
      <td>549</td>
      <td class="speedup">5.28</td>
    </tr>
    <tr>
      <td>1024</td>
      <td>1024</td>
      <td>4</td>
      <td>26.5</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>32</td>
      <td>536</td>
      <td class="speedup">20.23</td>
    </tr>
  </tbody>
</table>