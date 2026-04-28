# Das & Giannakis (2019) Reading Notes

Paper: Suddhasattwa Das and Dimitrios Giannakis, **"Delay-coordinate maps and the spectra of Koopman operators"**, *Journal of Statistical Physics* 175, 1107-1145, 2019.

Primary link: <https://arxiv.org/abs/1706.08544>  
DOI: <https://doi.org/10.1007/s10955-019-02272-w>

---

## Module 1: One-Line Summary

This paper shows that using many delay coordinates in a kernel method can isolate the predictable Koopman eigenfunction part of a dynamical system from data.

---

## Module 2: Key Contributions

### What was the problem before?

Koopman theory is attractive because it studies nonlinear systems using linear operators on observables.

The hard part is spectral approximation. Real dynamical systems can have both point spectrum and continuous spectrum. The point spectrum corresponds to coherent, quasiperiodic, predictable components. The continuous spectrum corresponds more to mixing or chaotic behavior.

In practice, it is difficult to approximate Koopman eigenfunctions from data, especially when the system is high-dimensional, partially observed, or has mixed spectrum.

### What is new in this paper?

The paper builds a kernel operator using delay-coordinate maps.

The main idea is:

```text
compare two states by comparing their future observation histories,
not just their instantaneous observations.
```

As the number of delays goes to infinity, the resulting kernel operator commutes with the Koopman operator. Because commuting operators share eigenspaces, this kernel operator can be used to approximate Koopman eigenfunctions.

The paper also shows that the limiting delay-kernel construction filters out the continuous-spectrum part and maps into the point-spectrum subspace.

### Why does it matter?

This gives a theoretical reason why delay coordinates are powerful for Koopman analysis.

Instead of treating delay embedding as just a useful trick, the paper shows that long delay windows can separate the predictable quasiperiodic part of the dynamics from the mixing part.

This is especially important for systems with mixed behavior, where some components are predictable and oscillatory while other components are chaotic.

---

## Module 3: Method and Purpose

### Goal

The goal is to approximate Koopman eigenfunctions from data in a way that has theoretical convergence guarantees.

The paper is not mainly about producing a simple low-dimensional physical model like HAVOK. It is more about understanding the spectral effect of delay-coordinate maps.

### High-Level Approach

The method is roughly:

1. Observe data from a dynamical system through a measurement map `F`.
2. Build delay-coordinate vectors using many future samples of `F`.
3. Define a distance between two states by comparing their delay-coordinate histories.
4. Turn this distance into a kernel.
5. Normalize the kernel to get a Markov kernel operator `P_Q`.
6. Use eigenfunctions of this kernel operator as a basis for approximating Koopman eigenfunctions.
7. Use a Galerkin method with small diffusion regularization to approximate the Koopman generator.

The key limit is:

```text
number of delays Q -> infinity
```

In that limit, the delay-coordinate kernel operator becomes aligned with the Koopman operator's point-spectrum structure.

### Why use this approach?

Instantaneous observations can mix together many processes. Delay histories contain dynamical information because they describe how observations evolve over time.

By comparing long histories, the method keeps the coherent, repeatable part of the dynamics and suppresses the purely mixing part.

---

## Module 5: Important Formulas

### Koopman operator

```math
(U^t f)(x) = f(\Phi^t(x))
```

This says the Koopman operator advances an observable `f` by composing it with the flow map `\Phi^t`.

### Koopman eigenfunction equation

```math
U^t z = e^{i\omega t}z
```

This says a Koopman eigenfunction evolves by a single frequency, which makes it highly predictable.

### Point and continuous spectrum splitting

```math
L^2(X,\mu) = D \oplus D^\perp
```

This splits observables into the point-spectrum part `D` and the remaining continuous-spectrum part `D^\perp`.

### Kernel integral operator

```math
Kf(x) = \int_X k(x,y)f(y)\,d\mu(y)
```

This defines an operator from a similarity kernel `k(x,y)` between points.

### Delay-coordinate distance

```math
d_Q^2(x,y)
=
\frac{1}{Q}
\sum_{q=0}^{Q-1}
\left\|
F(\Phi^{q\Delta t}(x))
-
F(\Phi^{q\Delta t}(y))
\right\|^2
```

This measures how different two states are by comparing their next `Q` observations.

### Delay-coordinate kernel

```math
k_Q(x,y) = e^{-d_Q^2(x,y)/\epsilon}
```

This turns the delay-coordinate distance into a similarity score.

### Koopman generator

```math
Vf = \lim_{t\to 0}\frac{U^t f - f}{t}
```

This is the continuous-time infinitesimal generator of the Koopman operator group.

---

## Important Results and Notes

The main theoretical result is that, as the number of delays becomes infinite, the delay-coordinate kernel operator converges to an operator that commutes with the Koopman operator.

This matters because if two operators commute, their eigenspaces are compatible. That gives a way to use the kernel operator's eigenfunctions as a basis for Koopman spectral approximation.

A second important point is that the limiting operator removes the continuous-spectrum component. So the method is best suited for finding the point-spectrum part of the dynamics.

The paper tests this on mixed-spectrum examples, including product systems where a chaotic or mixing component is combined with a periodic rotation. The method recovers eigenfunctions associated with the periodic part, and these eigenfunctions behave like sinusoidal pairs with frequencies related to the rotation.

The paper also discusses a limitation using the Lorenz-63 system. Purely chaotic systems with only continuous spectrum do not have useful nonconstant Koopman eigenfunctions for this method to recover. In that case, as the number of delays goes to infinity, the nontrivial eigenvalues of the kernel operator are expected to collapse toward zero.

So the important distinction is:

```text
mixed spectrum: the method can recover predictable Koopman eigenfunctions.
purely continuous spectrum: the method has little point-spectrum structure to recover.
```

---

## Final Takeaway

This paper is mainly about what delay coordinates do to Koopman spectral analysis.

The key message is:

```text
Long delay-coordinate kernels filter the data toward the point-spectrum part of the Koopman operator.
```

For HAVOK-related work, this paper is useful because it gives theoretical support for why delay coordinates can reveal coherent, predictable structure from complicated observations.

But it also gives an important warning: if the system is dominated by continuous-spectrum chaotic behavior, delay-coordinate methods may not recover clean autonomous Koopman eigenfunctions. They may still be useful, but the target is different.
