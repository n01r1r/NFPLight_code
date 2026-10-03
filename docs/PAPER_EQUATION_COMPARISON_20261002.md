# Paper-equation comparison — 2026-10-02

The user clarified that “full reproduction” means mathematical reproduction, not reinstating the denoiser. Keep FP32, the original net_g_real.pth, identity copies, the five 10/30cm groups, and both photometry. MatSynth results are archived and excluded from reports.

## Frozen comparison

Freshly prepare the DNGs into artifacts/fabric_capture_20261002_paper_equations. Both branches use the same selected frames, rectified/compensated/color-transformed [1,6,256,256] FP32 linear-RGB tensor. Baseline is the unmodified author-code equations; comparison evaluates paper equations (4) and (5).

- Eq. (4): RGB-mean intensity at pixel [H/2,W/2], i.e. [128,128], for each input. K_L = I_N(center)/I_F(center). A nonpositive far center is an explicit error, not an epsilon gain.
- Eq. (5): scalar RGB means first, then absolute difference; numerator = abs(I_N - K_L*K_theta*I_F). No clipping of the scaled far term for raw mathematical relation.
- C_M = cos(theta_N)*abs(cos(theta_N)-cos(theta_F)); no global-maximum normalization or denominator floor in raw RM.
- Identical nominal normalized geometry d_N=4, d_F=12, rho=sqrt((x-127.5)^2+(y-127.5)^2)/128 in both branches. This isolates equation changes; it does not calibrate real distances or correct registration. At even 256 resolution all sampled rho are nonzero. Reject zero/invalid denominator instead of hiding the center singularity.
- Raw RM is saved separately. For checkpoint inference only, retain the author 33-channel packing, log epsilon, inverse min-max normalization, appearance clipping and saturation mask. These are explicitly labeled checkpoint adaptation, not printed Eq. (5). Same estimator, no retraining. This is an equation perturbation under fixed learned weights, not an accuracy validation or a claim of retraining paper features.
- Preserve raw original near/far inputs and record input clipping; paper formulas consume the same LDR-compatible tensor as baseline. Do not silently fix large near-far registration residuals or reinterpret the nominal geometry as measured physical geometry.

## Acceptance

2026-10-02 report correction: paper-mode `signed_diff` and `abs_diff` now expose the scalar `[1,1,256,256]` RM intermediates (RGB means before subtraction, then absolute value), matching `paper_signed_difference` / `paper_numerator`. Earlier saved paper archives retain their historical RGB diagnostic fields; refreshed figures read the authoritative `paper_*` arrays instead. Raw RM, estimator features and predictions are unchanged by this trace correction. Author-code traces remain upstream-compatible; report-only `near_minus_scaled_far` aliases negate their saved `signed_diff` and its statistics so only near-minus-scaled-far is shown. The author path still clips scaled far and averages channelwise absolute values; it is explicitly distinct from printed Eq. (5). Independent checks cover the signed intermediate as well as the numerator, denominator and RM.

Small deterministic algebraic tests: analytic diffuse cancellation, algebraic Eq. (5) identity under its assumptions, no RGB absolute-value-before-mean bias, unclipped gained far behavior, invalid central intensity/denominator, finite FP32 outputs. Independently recompute each saved RM/gain/numerator/denominator from NumPy FP32. Validate NPZ round trips, checkpoint/source/code hashes, identical input populations, and report assets. Report raw prediction difference by component (MAE/RMSE/max in raw tanh units), relation difference, gain and saturation/registration limitations. No material accuracy ranking without GT.
