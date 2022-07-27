## PegasusTools

Utilities and applications for D-Wave annealing problems. 
(Under highly diabatic development)

### Requirements

 * D-Wave Ocean SDK
 * Numpy
 * Scipy
 * Pandas
 * Pytables
 * NetworkX
 * Pyyaml

To install the editable (development) package, run `pip install -e .`

### Examples

Anneal a simple ferromagnetic chain gadget embedded in a single Pegasus unit cell, automatically
repeating the embedding throughout the hardware graph to gather a large number of samples.
```shell
pgt-cell-anneal -v --tf 20 --rand-gauge --reps 3 ./examples/fm_gadget.txt ./examples/fm_gadget_out
```
Output: `fm_gadget_out_samps.csv`
```csv
blabel,num_occurrences,energy
0,92514,-7.5
252,9697,-6.5
254,10900,-6.5
255,12172,-6.5
192,743,-6.0
224,729,-6.0
128,34,-5.5
.....
```
---
Generate a L=5 QAC topology from the default D-Wave sampler, 
and plot the logical topology.
```shell
pgt-qac-top
-L 5
--plot qac_L5_dwa4.1.pdf
--labels pqac_L5_labels.json
pqac_L5.txt
```
The main output is the adjacency list `pqac_L5.txt`. The additional output
`pqac_L5_labels.json` maps the linear indices of the L=5 topology to their coordinates
on the hardware graph. See the comments in `qac.PegasusQACGraph` for details.
---
Use the above QAC topology to generate a random spin glass instance, using a fixed
random seed (which includes an instance index `-n 0`)
```shell
pgt-gen -n 0  --seed 1148089361  pqac_L5.txt s28 qac_s28_L5_0.txt
```
The `s28` instance class are Sidon-28 instances with a maximum range of 28.
The output file `qac_s28_L5_0.txt` is a (relatively) standard three-column text
specification of the instance (`i j Jij`/`i i hi`).
---
To sample the QAC instance with the D-Wave sampler, run
```shell
pgt-qac --tf 3.0  -n 64 --reps 10 -R  --scale-j=28.0  --qac-penalty=0.2 --qac-mode=qac \
    --qac-mapping pqac_L5_labels.json  qac_s28_L5_0.txt qac_p0.2_L5_0_tf3.0
```
This command runs 10 repetitions (`--reps`) of the anneal, each with a random 
hardware-level gauge (`-R`) and 64 samples per repetition (`-n`), using a linear anneal schedule 
of 3 us (`--tf`). 
---
Some advanced schedules constructions are supported.
For a pause and ramp schedule, specify `pr <t1> <sp> <tp> <tr>`,
where `t1` is the initial anneal time, `sp` is the pause point, 
`tp` is the pause time, and `tr` is the ramp time, e.g.
```shell
pgt-anneal --schedule pr 1.0 0.3 20.0 1.0 # ...
```
For a boundary cancelation (ramped beta) schedule, specify the primary anneal
time with `--tf` and the schedule parameters with `beta <a> <b> <sq>`.