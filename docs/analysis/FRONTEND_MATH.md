---
title: "The Circuit-4 front-end by hand: clean input $x$ and noisy input $x+n$"
date: 2026-10-29
geometry: margin=2.2cm
fontsize: 11pt
header-includes:
  - \usepackage{amsmath,amssymb}
---

**Purpose.** Derive, step by step from the gates, what the trained front-end (M2/M3/M5) outputs for an input image $x$, then for the same image plus Gaussian noise $n$, and compare with M4. Every equation below is checked against the real `quanv_torch.py` (the copy used in v5–v15.5, identical in every round folder) by `verify_frontend_math.py`; results in §6.

# 1. Setup: which pixel goes where

**Input.** One image $x\in[0,1]^{28\times 28}$, entries $x[r,c]$. No resize: the front-end takes the raw $28\times28$ image.

**Patches.** Non-overlapping $2\times2$ patches, stride 2, give a $14\times14$ grid. For output position $(i,j)$, $i,j=0,\dots,13$:
$$
\begin{pmatrix} p_1 & p_2\\ p_3 & p_4\end{pmatrix}
=\begin{pmatrix} x[2i,2j] & x[2i,2j+1]\\ x[2i+1,2j] & x[2i+1,2j+1]\end{pmatrix}.
$$

**Qubits.** Four qubits, split into two **independent 2-qubit blocks** (no gate spans them, so the 4-qubit state is a product of two 2-qubit states):

- block A = top row: qubit $a \leftarrow p_1$, qubit $b \leftarrow p_2$
- block B = bottom row: qubit $a \leftarrow p_3$, qubit $b \leftarrow p_4$

Both blocks use the **same six weights** $w_0,\dots,w_5$.

**Gate sequence per block** (from `_block_unitary`, basis order $|q_a q_b\rangle$):
$$
R_Y(\pi x_a)\otimes R_Y(\pi x_b)\;\to\;R_Y(w_0)\otimes R_Y(w_1)\;\to\;\mathrm{CRZ}(w_2)_{b\to a}\;\to\;R_Y(w_3)\otimes R_Y(w_4)\;\to\;\mathrm{CRZ}(w_5)_{a\to b}\;\to\;\text{measure }Z_a,Z_b .
$$

**Output.** $y\in[-1,1]^{4\times14\times14}$, channels in code order
$$
y[0,i,j]=\langle Z_a\rangle_A,\quad y[1,i,j]=\langle Z_b\rangle_A,\quad y[2,i,j]=\langle Z_a\rangle_B,\quad y[3,i,j]=\langle Z_b\rangle_B .
$$

**Gates used.**
$$
R_Y(\theta)=\begin{pmatrix}\cos\frac\theta2 & -\sin\frac\theta2\\ \sin\frac\theta2 & \cos\frac\theta2\end{pmatrix},\qquad
\mathrm{CRZ}(w)_{b\to a}=\mathrm{diag}\big(1,\;e^{-iw/2},\;1,\;e^{+iw/2}\big)\ \text{on } |00\rangle,|01\rangle,|10\rangle,|11\rangle .
$$
(Control $b$; when $b=1$, target $a$ gets $R_Z(w)=\mathrm{diag}(e^{-iw/2},e^{iw/2})$.)

# 2. Clean input: $x \mapsto y$, by hand

Work on one block with pixels $(x_a,x_b)$. Start in $|00\rangle$.

## Step 1–2. Encoding and the first trainable layer merge

$R_Y$ rotations about the same axis add: $R_Y(w_0)R_Y(\pi x_a)=R_Y(\pi x_a+w_0)$. Define the **effective angles**
$$
\boxed{t_a=\pi x_a+w_0,\qquad t_b=\pi x_b+w_1.}
$$
Each qubit is now $R_Y(t)|0\rangle=\cos\frac t2|0\rangle+\sin\frac t2|1\rangle$. Write $c_a=\cos\frac{t_a}2,\ s_a=\sin\frac{t_a}2$ (same for $b$). The product state is
$$
|\psi_1\rangle=c_ac_b|00\rangle+c_as_b|01\rangle+s_ac_b|10\rangle+s_as_b|11\rangle .
$$
So $w_0,w_1$ act only as **phase offsets** on the pixel angle.

## Step 3. The entangling gate CRZ$(w_2)$

It multiplies each amplitude by a phase. With $\gamma=w_2/2$:
$$
|\psi_2\rangle=c_ac_b|00\rangle+c_as_b\,e^{-i\gamma}|01\rangle+s_ac_b|10\rangle+s_as_b\,e^{+i\gamma}|11\rangle .
$$
This is entangled whenever $\sin\gamma\neq0$ and both qubits are in superposition. Populations, and therefore $\langle Z_a\rangle,\langle Z_b\rangle$, are unchanged. Only coherences change.

## Step 4. Each qubit's Bloch vector after CRZ

The next gates act on one qubit each, and then each qubit is measured separately. So we only need each qubit's **reduced** Bloch components $\langle X\rangle,\langle Z\rangle$, read off the density matrix: $\langle Z\rangle=\rho_{00}-\rho_{11}$ and $\langle X\rangle=2\,\mathrm{Re}\,\rho_{01}$.

**Qubit $b$.** Here $\rho^{(b)}_{01}=\sum_{q_a}\psi_{q_a0}\,\psi^*_{q_a1}=c_b s_b\big(c_a^2e^{+i\gamma}+s_a^2e^{-i\gamma}\big)$. Hence
$$
\langle Z_b\rangle_2=\cos t_b,\qquad \langle X_b\rangle_2=2c_bs_b\,(c_a^2+s_a^2)\cos\gamma=\sin t_b\cos\tfrac{w_2}{2}.
$$
The imaginary part, $\langle Y_b\rangle_2=-2\,\mathrm{Im}\,\rho^{(b)}_{01}=-\sin t_b\sin\frac{w_2}2\cos t_a$, carries the dependence on qubit $a$. As Step 5 shows, an $R_Y$ followed by a $Z$ measurement cannot see $Y$.

**Qubit $a$.** Here $\rho^{(a)}_{01}=\sum_{q_b}\psi_{0q_b}\psi^*_{1q_b}=c_as_a\big(c_b^2+s_b^2e^{-2i\gamma}\big)$. Hence
$$
\langle Z_a\rangle_2=\cos t_a,\qquad \langle X_a\rangle_2=\sin t_a\big(c_b^2+s_b^2\cos w_2\big)=\sin t_a\,\kappa(x_b),
$$
$$
\boxed{\kappa(x_b)=\cos^2\tfrac{w_2}2+\sin^2\tfrac{w_2}2\,\cos t_b }\qquad\big(\text{using } c_b^2+s_b^2\cos w_2 = 1-2s_b^2\sin^2\gamma\big).
$$
Physically: when $b=0$ (probability $c_b^2$), $a$ is untouched; when $b=1$ (probability $s_b^2$), $a$'s coherence is rotated by $w_2$. $\kappa$ is the average of these two cases.

## Step 5. Second trainable layer and measurement

Rotating a Bloch vector by $R_Y(w)$ maps $(\langle X\rangle,\langle Z\rangle)\mapsto(\langle X\rangle\cos w+\langle Z\rangle\sin w,\ \langle Z\rangle\cos w-\langle X\rangle\sin w)$. So
$$
\langle Z\rangle_{\text{out}}=\cos w\,\langle Z\rangle-\sin w\,\langle X\rangle .
$$
The final CRZ$(w_5)$ is diagonal, and it sits directly before a diagonal ($Z$) measurement. Its phases cancel in $|\psi|^2$, so **$w_5$ drops out exactly** (the known dead parameter).

## Result: the front-end in closed form

$$
\boxed{\;y_b(x_b)=\cos w_4\cos(\pi x_b+w_1)\;-\;\sin w_4\cos\tfrac{w_2}2\,\sin(\pi x_b+w_1)\;}\tag{F1}
$$
$$
\boxed{\;y_a(x_a,x_b)=\cos w_3\cos(\pi x_a+w_0)\;-\;\sin w_3\,\kappa(x_b)\,\sin(\pi x_a+w_0)\;}\tag{F2}
$$

**Amplitude–phase form.** Both are shifted cosines of their own pixel:
$$
y_b=A_b\cos(\pi x_b+\delta_b),\quad A_b=\sqrt{\cos^2w_4+\sin^2w_4\cos^2\tfrac{w_2}2},\quad \delta_b=w_1+\operatorname{atan2}\!\big(\sin w_4\cos\tfrac{w_2}2,\ \cos w_4\big),
$$
$$
y_a=A_a(x_b)\cos\big(\pi x_a+\delta_a(x_b)\big),\quad A_a=\sqrt{\cos^2w_3+\sin^2w_3\,\kappa^2},\quad \delta_a=w_0+\operatorname{atan2}(\sin w_3\,\kappa,\ \cos w_3).
$$

**Expanded so the entanglement term is visible:**
$$
y_a=\underbrace{\cos w_3\cos t_a-\sin w_3\cos^2\tfrac{w_2}2\sin t_a}_{\text{own pixel only}}\;\underbrace{-\;\sin w_3\sin^2\tfrac{w_2}2\,\sin t_a\cos t_b}_{\text{cross-pixel term (from CRZ)}} .
$$
The cross term $\sin t_a\cos t_b$ is the **only** place where one pixel affects another channel. $y_b$ depends on its own pixel alone.

**Full map for patch $(i,j)$:**
$$
y[0,i,j]=y_a(p_1,p_2),\quad y[1,i,j]=y_b(p_2),\quad y[2,i,j]=y_a(p_3,p_4),\quad y[3,i,j]=y_b(p_4).
$$

## Properties that matter for noise

1. **Degree-1 trigonometric polynomial.** Each output contains only the frequencies $\{0,\pm\pi\}$ in each pixel (Schuld, Sweke & Meyer 2021). It is a fixed classical function with five effective weights.
2. **Black and white are antipodal.** $\cos(\pi+\theta)=-\cos\theta$ and $\sin(\pi+\theta)=-\sin\theta$, so $y_b(1)=-y_b(0)$ and $y_a(1,x_b)=-y_a(0,x_b)$.
   Black–white **contrast** of channel $b$: $y_b(1)-y_b(0)=-2A_b\cos\delta_b$.
3. **Slope:** $\partial y_b/\partial x_b=-\pi A_b\sin(\pi x_b+\delta_b)$. At the pixel values MNIST uses ($x\approx0$ or $1$) the slope is $\mp\pi A_b\sin\delta_b$. It is **zero iff $\delta_b\in\{0,\pi\}$**, which is also exactly where the contrast is largest.
4. **Blank patch** ($x\equiv0$): $y_b=\cos w_4\cos w_1-\sin w_4\cos\frac{w_2}2\sin w_1$, a constant set by the weights (not 0).

## Comparison: M4 (no front-end)

`F.interpolate(28→14, bilinear, align_corners=False)` at an exact factor of 2 samples every patch at its centre. That **equals the $2\times2$ mean** (verified, V6). The result is replicated to 4 channels:
$$
y^{\text{M4}}[c,i,j]=\tfrac14(p_1+p_2+p_3+p_4),\qquad c=0,1,2,3 .
$$
This is linear, with slope $\tfrac14$ per pixel (1 for the mean) everywhere, including at 0 and 1.

# 3. Noisy input: $x+n \mapsto \tilde y$, by hand

Noise $n[r,c]\sim\mathcal N(0,\sigma^2)$, i.i.d. per pixel, added to the $28\times28$ image **before** the front-end. Write $\theta_b=\pi x_b+\delta_b$ (the clean phase) and $\nu=\pi n$ (the noise expressed as an angle).

## 3.1 Exact, for any single noise draw (no approximation, no clipping)

The pixel enters only through $\pi x$, so noise is simply **an extra rotation angle** $\pi n$ on the encoding gate. Using $\cos(\theta+\nu)=\cos\theta\cos\nu-\sin\theta\sin\nu$:
$$
\boxed{\;\tilde y_b=y_b\cos(\pi n_b)\;-\;A_b\sin\theta_b\,\sin(\pi n_b)\;}\tag{N1}
$$
$$
\Delta y_b=\tilde y_b-y_b=\underbrace{-A_b\sin\theta_b\,\sin(\pi n_b)}_{\text{odd in }n}\;\underbrace{-\;y_b\,\big(1-\cos(\pi n_b)\big)}_{\text{even in }n,\ \le 0\text{ in } |\cdot|}.
$$
For channel $a$, both $t_a\to t_a+\pi n_a$ and $t_b\to t_b+\pi n_b$:
$$
\tilde y_a=\cos w_3\cos(t_a+\pi n_a)-\sin w_3\Big[\cos^2\tfrac{w_2}2+\sin^2\tfrac{w_2}2\cos(t_b+\pi n_b)\Big]\sin(t_a+\pi n_a).\tag{N2}
$$

## 3.2 Small noise: Taylor expansion

$$
\Delta y_b=-\pi A_b\sin\theta_b\;n_b\;-\;\tfrac{\pi^2}{2}A_b\cos\theta_b\;n_b^2\;+\;O(n^3).
$$
$$
\Delta y^{\text{M4}}=\tfrac14(n_1+n_2+n_3+n_4)\quad\text{(exactly first order, no higher terms).}
$$
**The key comparison:**

- At a black or white pixel, $\sin\theta_b=\pm\sin\delta_b$. If $\delta_b=0$, the **first-order term vanishes** and the front-end passes only $\frac{\pi^2}{2}n^2$.
- M4 always passes noise at first order.
- With a phase $\delta_b\neq0$, a first-order leak of size $\pi A_b|\sin\delta_b|\,|n|$ comes back.

## 3.3 Averaging over the noise: unclipped Gaussian

Use $\mathbb E[\cos\pi n]=e^{-\pi^2\sigma^2/2}\equiv g$, $\mathbb E[\sin\pi n]=0$ and $\mathbb E[\cos2\pi n]=g^4$.

$$
\boxed{\ \mathbb E[\tilde y_b]=g\,y_b,\qquad \operatorname{Var}[\tilde y_b]=\tfrac{A_b^2}{2}(1-g^2)\big(1-g^2\cos2\theta_b\big).\ }\tag{E1}
$$

- The **mean signal is shrunk toward 0** by $g$: $g=0.95$ at $\sigma=0.1$, $0.64$ at $0.3$, $0.29$ at $0.5$. So contrast is attenuated.
- At a flat point ($\theta_b=0,\pi$): $\operatorname{Var}=\frac{A_b^2}{2}(1-g^2)^2\approx\frac{A_b^2}{2}\pi^4\sigma^4$. That is SD $\propto\sigma^2$, **second order**.
- At a steep point ($\theta_b=\pm\pi/2$): $\operatorname{Var}\approx A_b^2\pi^2\sigma^2$, first order.
- M4: mean unchanged, $\operatorname{Var}=\sigma^2/4$.

For channel $a$ (independent $n_a,n_b$):
$$
\mathbb E[\tilde y_a]=g\Big[\cos w_3\cos t_a-\sin w_3\cos^2\tfrac{w_2}2\sin t_a\Big]\;-\;g^{\,2}\,\sin w_3\sin^2\tfrac{w_2}2\,\sin t_a\cos t_b .\tag{E2}
$$
**The cross-pixel (entangling) term decays as $g^2=e^{-\pi^2\sigma^2}$, twice as fast in the exponent as the single-pixel terms.** Noise removes the CRZ contribution first.

## 3.4 Clipped noise: what v15.5 actually did

The corruption is $\tilde x=\mathrm{clip}(x+n,0,1)$. In MNIST 91% of pixels are within 0.1 of 0 or 1, and 81% are exactly 0. At those pixels clipping makes the noise **one-sided**:

- **background** $x=0$: $\tilde x=u=\max(n,0)$. So $u=0$ with probability $\tfrac12$; otherwise $u$ is half-normal.
- **stroke** $x=1$: $\tilde x=1-v$, $v=\max(-n,0)$, the same distribution as $u$.

(The upper clip at 1 is ignored; $P(n>1)=4\times10^{-4}$ at $\sigma=0.3$.) The needed moments are exact:
$$
\mathbb E[\cos\pi u]=\tfrac12\big(1+e^{-\pi^2\sigma^2/2}\big),\qquad \mathbb E[\sin\pi u]=\tfrac{1}{\sqrt\pi}\,D\!\big(\tfrac{\pi\sigma}{\sqrt2}\big),
$$
where $D$ is the Dawson function. Substituting into (N1) at $\theta_b=\delta_b$ (background) and $\theta_b=\pi+\delta_b$ (stroke):
$$
\mathbb E[\tilde y_b\,|\,x_b=0]=A_b\Big[\cos\delta_b\cdot\tfrac12(1+g)-\sin\delta_b\cdot\tfrac{1}{\sqrt\pi}D(\tfrac{\pi\sigma}{\sqrt2})\Big],
$$
$$
\mathbb E[\tilde y_b\,|\,x_b=1]=-A_b\Big[\cos\delta_b\cdot\tfrac12(1+g)+\sin\delta_b\cdot\tfrac{1}{\sqrt\pi}D(\tfrac{\pi\sigma}{\sqrt2})\Big].
$$

**Normalised mean shift.** Express the mean shift as a fraction of the channel's own black–white contrast, signed so that "toward the other colour" is positive. For the front-end and for M4 (whose contrast is 1):
$$
\boxed{\ \beta_{\text{bg}}=\underbrace{\tfrac14\big(1-e^{-\pi^2\sigma^2/2}\big)}_{\text{squash, }O(\sigma^2)}+\underbrace{\tfrac{\tan\delta_b}{2\sqrt\pi}D\!\big(\tfrac{\pi\sigma}{\sqrt2}\big)}_{\text{tilt, }O(\sigma)},\qquad
\beta_{\text{stroke}}=\tfrac14\big(1-e^{-\pi^2\sigma^2/2}\big)-\tfrac{\tan\delta_b}{2\sqrt\pi}D\!\big(\tfrac{\pi\sigma}{\sqrt2}\big),\qquad
\beta^{\text{M4}}=\tfrac{\sigma}{\sqrt{2\pi}}\ }\tag{C1}
$$

| $\sigma$ | M4 | front-end, $\delta=0$ | $\delta=-20^\circ$ (bg / stroke) | $\delta=+20^\circ$ (bg / stroke) |
|---|---|---|---|---|
| 0.1 | 0.040 | 0.012 | $-0.010$ / 0.034 | 0.034 / $-0.010$ |
| 0.3 | 0.120 | 0.090 | 0.038 / 0.141 | 0.141 / 0.038 |
| 0.5 | 0.200 | 0.177 | 0.123 / 0.231 | 0.231 / 0.123 |

**Reading the table:**

- At $\delta=0$ the front-end transmits 3.3× less mean shift than M4 at $\sigma=0.1$ and 1.3× less at $\sigma=0.3$. The advantage has almost vanished by $\sigma\approx0.7$ (with clipping: 0.248 vs 0.255).
- A non-zero phase helps the background and hurts the stroke, or the reverse. The background is about 80% of pixels, so the sign of $\delta$ that protects the background wins overall.

# 4. Worked example (real seed-42 initial weights)

**Weights** (seed 42, `torch.Generator().manual_seed(42)`, $U(0,2\pi)$):
$w=(5.5435,\ 5.7491,\ 2.4056,\ 6.0275,\ 2.4533,\ 3.7755)$.
For channel $b$ these give $A_b=0.805$ and $\delta_b=+132.9^\circ$. Because $\cos(\theta+\pi)=-\cos\theta$, only $\tan\delta_b=-1.08$ matters for the tilt. So this circuit is **steep** at 0 and 1: $|\sin\delta_b|=0.73$.

**One patch on a stroke edge**, with one noise draw (about $\sigma=0.3$), clipped:
$$
p=\begin{pmatrix}0 & 1\\ 0.5 & 0\end{pmatrix},\quad
n=\begin{pmatrix}0.21 & -0.13\\ 0.05 & -0.30\end{pmatrix},\quad
\tilde p=\mathrm{clip}(p+n)=\begin{pmatrix}0.21 & 0.87\\ 0.55 & 0\end{pmatrix}.
$$

**Clean, top row** $(x_a,x_b)=(0,1)$:

| Step | Quantity | Value |
|---|---|---|
| 1–2 | $t_a=\pi\cdot0+w_0$, $t_b=\pi\cdot1+w_1$ | 5.5435, 8.8907 |
| | $(\cos\frac{t_a}2,\sin\frac{t_a}2)$, $(\cos\frac{t_b}2,\sin\frac{t_b}2)$ | $(-0.9324,\ 0.3615)$, $(-0.2639,\ -0.9646)$ |
| 3 | amplitudes after CRZ on $|00\rangle,|01\rangle,|10\rangle,|11\rangle$ | $0.2460,\ 0.3235-0.8391i,\ -0.0954,\ -0.1254-0.3253i$ |
| 4 | $\langle Z_a\rangle,\langle X_a\rangle$; $\langle Z_b\rangle,\langle X_b\rangle$ | $0.7387,\ 0.4179$; $-0.8608,\ 0.1831$ |
| 5 | $y_a=\cos w_3\langle Z_a\rangle-\sin w_3\langle X_a\rangle$ | **0.820331** |
| | $y_b=\cos w_4\langle Z_b\rangle-\sin w_4\langle X_b\rangle$ | **0.548440** |

**Clean, bottom row** $(0.5, 0)$: $y_a=0.816338$, $y_b=-0.548440$. As expected from §2, the black and white values of $y_b$ are antipodal: $+0.5484$ vs $-0.5484$.

**Noisy:** substitute $\tilde p$ into the same steps (full listing in `worked.log`):

| channel | pixel(s) | $y$ (clean) | $\tilde y$ (noisy) | $\Delta y$ | $\Delta y$ / own contrast | M4: $\Delta$ pixel |
|---|---|---|---|---|---|---|
| 0, $y_a$ top | $0\to0.21$ (bg) | 0.8203 | 0.9721 | +0.152 | **−0.093** (away from ink) | +0.21 (toward ink) |
| 1, $y_b$ top | $1\to0.87$ (stroke) | 0.5484 | 0.2691 | −0.279 | **−0.255** (toward bg) | −0.13 |
| 2, $y_a$ bottom | $0.5\to0.55$ | 0.8163 | 0.7179 | −0.098 | +0.087 | +0.05 |
| 3, $y_b$ bottom | $0\to0$ (noise $-0.3$ clipped) | −0.5484 | −0.5484 | 0 | 0 | 0 |

**Hand values equal `TrainableQuanvFrontEndTorch.forward` to every printed digit.** For M4 the whole patch gives $y=0.375\to\tilde y=0.4075$ on all four channels.

**What this one example shows:** the four noise mechanisms of §3, each visible in one row.

- **Row 4 (clipping).** The negative noise on a black pixel is erased completely, so the noise is one-sided.
- **Row 1 (background, $\tan\delta<0$).** This seed's phase has $\tan\delta<0$, so the background pixel moves *away* from ink. That is the protective tilt in (C1).
- **Row 2 (stroke, same sign).** The same sign makes the dimmed stroke pixel move about twice as far as M4's, in contrast units. That is the stroke penalty in (C1).
- **Row 3 (mid-grey).** Mid-grey pixels pass noise at first order through either map.

# 5. What the derivation says, in plain words

1. **Clean.** Each front-end channel is a shifted cosine of one pixel, $A\cos(\pi x+\delta)$. Channel $a$ adds one small cross term with its row partner. There are no other interactions and no quantum effect beyond what this formula captures.
2. **Noisy.** Pixel noise is just an extra rotation angle $\pi n$. Through a cosine, the change has a first-order part $\propto\sin\delta$ and a second-order part $\propto n^2$. M4 passes all noise at first order.
3. **What reaches the network.** conv1 sums 36 inputs, so the **mean** shift adds up coherently: $36\beta$ against $6\times$ the SD. The clipped background shift (C1) is therefore the quantity that matters. It is second order at $\delta=0$, which explains the deployed-arm robustness.
4. **Where training enters.** Contrast $2A|\cos\delta|$ is maximal exactly where the tilt term vanishes. Clean training pushes toward larger contrast, and so toward a flatter response at 0 and 1.
5. **Entanglement is the most noise-fragile part.** Its contribution to the mean decays as $e^{-\pi^2\sigma^2}$, against $e^{-\pi^2\sigma^2/2}$ for the single-pixel terms (E2).

# 6. Verification against the code

Run against `code/v10_mnist10/quanv_torch.py` in the repo. The file is byte-identical (same md5) across the v5–v10 folders. v16c's gate G-c ties the same `forward` to the Gadi code, and gate G-b ties the init weights to it bitwise.

| Check | What | Max error |
|---|---|---|
| V1 | (F1)–(F2) vs code, 50 random weights × random images | $1.3\times10^{-15}$ |
| V2 | same at the real seed 42–51 init weights | $1.3\times10^{-15}$ |
| V3 | exact noisy form (N1) vs code on $x+n$ | $1.5\times10^{-15}$ |
| V4 | unclipped means (E1) and (E2), including the $g^2$ cross term (MC, $4\times10^5$) | $\le 7.5\times10^{-4}$ (SE-limited) |
| V5 | clipped background and stroke means (Dawson form) vs code, $\sigma=0.1,0.3$ (MC, $1.6\times10^6$) | $\le 2.1\times10^{-4}$ (within 4 SE) |
| V6 | M4's bilinear 28→14 equals the $2\times2$ mean | $1.1\times10^{-16}$ |

**Note.** The first V5 run failed at $\sigma=0.3$ with only $7.8\times10^4$ draws: error $2.8\times10^{-3}$ against a fixed tolerance of $2\times10^{-3}$, with SE $\approx1.6\times10^{-3}$. Exact quadrature, including the ignored clip at 1, agrees with the closed form to $1.2\times10^{-4}$. The test was re-run with 20× the draws and a tolerance of 4 SE + $3\times10^{-4}$, and passes. The formula did not change.

**Real initial phases (channel $b$).** These follow from (F1) and the seed weights, and feed predictions P-T1/P-T2 in `V15_5_THEORY.md`:

| seed | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 | 50 | 51 |
|---|---|---|---|---|---|---|---|---|---|---|
| $A_b$ | 0.805 | 0.980 | 0.929 | 0.965 | **0.063** | 0.939 | 0.949 | 0.838 | 0.838 | 0.761 |
| $\delta_b$ (deg) | +132.9 | +18.3 | +41.4 | −24.1 | +81.6 | −159.9 | +44.4 | +141.0 | −168.9 | −101.8 |
| $\tan\delta_b$ | −1.08 | +0.33 | +0.88 | −0.45 | +6.8 | +0.36 | +0.98 | −0.81 | +0.20 | +4.8 |

$A_b$ reproduces v16c exactly (0.81 for seed 42, 0.063 for the dead seed 46).

**Not covered here:**

- channel $a$'s clipped mean (it involves the cross term with the partner pixel; checked only unclipped, V4);
- how conv1 and the rest of the network respond (that is the empirical part of `V15_5_THEORY.md`).

*Files:* `verify_frontend_math.py` (checks V1–V6), `worked_example.py` (§4). Both take the path to `quanv_torch.py` as their argument.
