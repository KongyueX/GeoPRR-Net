# Conditional source-view saved-number and execution audit

Integrity **PASS**: 42230 checks, 0 failed; 15780 floating saved-value comparisons, abs tolerance1e-12, maximum difference1.4210854715202004e-14. Scientific verdict **NOT_RETAINED / source_reading_joint_failed**. Reversal native failed every one of the five simultaneous predicates against both identity15 and detached control. Its CF was correctly not run and remains null/0calls.

The operational critic effect passed: control4540/5496=0.8260553129548762, reversal2561/5496=0.4659752547307133, reduction in distance from chance=0.2920305676855895. This does not rescue the failed reading result.

| Arm/view | Mean %FS | Acc1/2/5 counts | Median %FS | P90 %FS | P95 %FS | Tail30 %FS | Max %FS | >1/2/5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity/native | 2.5431218244 | 102/182/276 | 1.5494510531 | 4.0915115178 | 5.4043346643 | 11.8009553314 | 53.7970364094 | 190/110/16 |
| identity/CF | 2.6353762865 | 104/189/275 | 1.4384098351 | 4.1645436361 | 5.7685494423 | 12.9459618763 | 54.2472898960 | 188/103/17 |
| detached_control/native | 2.6759538619 | 92/167/267 | 1.7107360065 | 4.4575661421 | 7.0123346150 | 11.6840900977 | 46.8278169632 | 200/125/25 |
| detached_control/CF | 2.8617163082 | 100/162/265 | 1.6120150685 | 4.8130086064 | 6.8449869752 | 12.8705381726 | 50.0665783882 | 192/130/27 |
| reversal/native | 3.3223066845 | 86/166/254 | 1.7772205174 | 5.8395713568 | 7.7916396223 | 16.0971364006 | 81.1839252710 | 206/126/38 |
| reversal/CF | null | null | null | null | null | null | null | null |

Each executed observation has292/292 rows and73 complete four-row groups; all inputs/scalars are valid, so failures0. Valid0/1 endpoints are accepted without failure. Exact source-ID order/groups/targets match in all five saved observations. Identity native protects102 cases and identityCF104; detached native92 and detachedCF100, each derived separately.

| Comparison | Images improved/worse/same | Mean groups improved/worse/same | <=1 protected -> >5 | 1% rescue/new | 2% rescue/new | 5% rescue/new |
| --- | --- | --- | --- | --- | --- | --- |
| control_native_vs_identity_native | 135/156/1 | 34/39/0 | 3/102 | 51/61 | 41/56 | 4/13 |
| control_CF_vs_identity_CF | 127/165/0 | 27/46/0 | 5/104 | 54/58 | 41/68 | 4/14 |
| reversal_native_vs_identity_native | 139/153/0 | 30/43/0 | 11/102 | 50/66 | 62/78 | 9/31 |
| reversal_native_vs_control_native | 142/150/0 | 37/36/0 | 6/92 | 47/53 | 70/71 | 16/29 |
| identity_CF_vs_native | 137/152/3 | 33/40/0 | 0/102 | 24/22 | 23/16 | 4/5 |
| control_CF_vs_native | 126/164/2 | 31/42/0 | 0/92 | 26/18 | 12/17 | 4/6 |

Both arms passed the original constant0.5 learning gate (half-constant mean=0.1250446370304666, half-constant tail=0.23692208052865074). Control also passed online critic>=.65. These prerequisites allowed controlCF and fresh reversal; they did not make control better than identity. Native reversal mean/tail/max/>5/protection all failed against both baselines.

Both arms completed15 scheduled epochs,82440 training presentations and1290 combined backward attempts each. Control1290 updates/0skip; reversal1289/1skip (epoch1). Formal totals:165756 CNN-point images/2595 calls each,164880 critic examples/2580 calls,2580 backward attempts=2579 updates+1skip. Pref is separate:8unique/3CNN-point, critic4/1,1back/0opt,12decodes including4IDCPU redecodes. Metadata5496/Z2748each/687parents/1374quartets/2748pairs, maximum consumed FP32 paired difference0 at fixed1e-6 tolerance,0RGB/model/GPU.

| Stage | Parent child lifetime s | Child module entry s | Train s | Native s | CF s |
| --- | --- | --- | --- | --- | --- |
| metadata | 1.769112106 | 1.192189701 | None | None | None |
| preflight | 18.499992587 | 17.481254861 | None | None | None |
| detached_control | 480.223776806 | 478.085258789 | 464.3296225592494 | 5.172072678804398 | None |
| control_cf | 14.058982871 | 12.505854830 | None | None | 8.913073733448982 |
| reversal | 478.855564538 | 476.574181709 | 461.7383591271937 | 5.336626909673214 | None |
| finalize | 2.351133034 | 1.991769660 | None | None | None |

Whole supervisor=996.8495061174035s; launch-receipt-to-supervisor-finish UTC=996.861378s; summed parent child lifetimes=995.7585619427264s. Both-arm training=926.0679816864431s. Peak observed CUDA allocated/reserved=8264907776/11349786624bytes; peak CPU=5662527488bytes. Critic FP32 region CUDA events=0.30189404548704624s, hook host=0.09653088822960854s. Current four checkpoint files total440096720bytes by stat only. Checkpoint I/O, point-only, total GPU compute and final screen/queue write times remain null.

Saved source0opt witness confirms8517906 parameters/237 tensors/468 states, critic82113/4 and original77BN; cached462 retained,237 parameters unchanged at0opt, strict468 self-reload, BNtrain+1/eval0, h/pointF16 and critic/logit/BCEF32, detached conditioning, finite nonzero stem/neck/point and critic first/last weights. NEWCPU seven synthetic checks passed in0.9023926854133606s after warm imports with no realMNV4/RGB/weights/CUDA and one toy-state read. No checks were rerun.

## Limitations

- Provided-label saved-value audit. Production normalized_target to FP32 consumer binding and paired difference0 are saved witnesses; no new physical label derivation. C3 original149/148/1 raw-derivation qualification remains unresolved and is not upgraded to PASS.

- Prepared metadata contains nested train/ID/OOD contracts and was privately parsed/copied; only training fields drive y/Z. No ID/OOD manifest or pixels were opened during metadata preparation.

- One seed, 15 scheduled epochs and1290 backward attempts each. Effective optimizer updates differ1290 vs1289 because reversal epoch1 had one normal AMP skip; no compensation or scaler adjustment.

- Online source critic counts include all attempted5496 presentations per epoch, including AMP skip; these are co-trained statistics, not held-out or frozen-feature probe accuracy. The critic effect does not prove conditional mutual information zero, causality, significance, a new blind test, convergence or real-domain generalization.

- Joint gradient clipping includes critic in both arms; detached control does not establish exact historical identity weight trajectory. Same formal initialization is supported by fresh seed/order and forked-RNG code plus saved initialization metadata, not a fresh cross-arm initial-tensor comparison.

- Pref witnesses cached462 initial states,237 parameters unchanged at0opt,468 strict self-reload and BN77 train+1/eval0. This audit did not reopen historical pixels or full weights and does not upgrade their certification.

- Float comparison uses absolute1e-12; IDs, groups, row order and counts use exact equality. Forward numerical sum is computed in FP32 per batch, so its averaged saved value need not equal the sum of separately averaged primitives at1e-12; the observed residuals are reported without rewriting numbers.

- Child module timers begin before torch import but exclude final JSON/down-time; supervisor includes child process lifetime through finalization and excludes screen/queue final writes and caller lifetime. Finalize1.991769660s is not whole-pipeline time. Phase and forward wall timings are not total GPU compute.

- CUDA event scope includes concat, critic, primitive, BCE and forward sum; it is not pure critic cost. Hook host time is separate. Checkpoint I/O, point-only and total GPU compute times are unmeasured null. Current checkpoint files were statted only; neither their bytes nor state were loaded. Cold model-load-only time is not isolated. CPU peaks use per-stage process RUSAGE_SELF RSS and do not sum loader worker memory or whole-host memory.

- Post-exit readiness uses the already saved00:57:52.503437UTC parent monitor witness. Final-child flags were written before parent exit and remain false; they are not a later liveness failure. This auditor made no new process/GPU query.

- ReversalCF was not run because native joint failed; all its results remain null and0calls. Neither controlCF nor identityCF is substituted. Source03 and actual Industrial/RF calls remain0; no model is promoted.

- The local recomputation script depends on private collected validation rows and is excluded from the public archive. Public normalized results report this one audit; the public bundle is not advertised as independently rerunnable from hidden context.
