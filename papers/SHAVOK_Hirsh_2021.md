# Hirsh et al. (2021) Reading Notes

Paper: Seth M. Hirsh, Sara M. Ichinaga, Steven L. Brunton, J. Nathan Kutz, and Bingni W. Brunton, **"Structured time-delay models for dynamical systems with connections to Frenet-Serret frame"**, *Proceedings of the Royal Society A* 477, 20210097, 2021.

Primary link: <https://pubmed.ncbi.nlm.nih.gov/35153585/>  
DOI: <https://doi.org/10.1098/rspa.2021.0097>

---

## Module 1: One-Line Summary

This paper explains why HAVOK often produces antisymmetric tridiagonal linear models and uses that insight to build a more stable structured HAVOK method for limited or noisy data.

---

## Module 2: Key Contributions

### What was the problem before?

The original HAVOK paper showed that time-delay coordinates can turn nonlinear or chaotic dynamics into a linear system plus a forcing term.

But there was a strange observation: the learned HAVOK matrix often looked very structured. It was close to antisymmetric, mostly tridiagonal, and had nonzero entries mainly on the sub- and super-diagonals.

This structure was useful, but it was not fully explained. It was unclear whether it was just a numerical coincidence, a property of certain examples, or something deeper about time-delay models.

Another practical problem was that standard HAVOK needed clean and sufficiently long data to recover this structure well. With short or noisy trajectories, the learned model could become less stable and less interpretable.

### What is new in this paper?

The paper connects HAVOK to the **Frenet-Serret frame**, which is a coordinate frame from differential geometry used to describe curves.

The main claim is:

```text
The structured HAVOK matrix is not accidental.
It appears because time-delay SVD coordinates approximate a Frenet-Serret frame.
```

In this view, the nonzero entries on the sub- and super-diagonals of the HAVOK matrix correspond to intrinsic curvatures of the trajectory.

The paper also proposes **structured HAVOK**, or **sHAVOK**. The main change is that sHAVOK uses two time-shifted Hankel matrices and applies SVD separately to them. This encourages the learned model to keep the expected antisymmetric tridiagonal structure.

### Why does it matter?

This paper makes HAVOK more interpretable.

Before, we could observe that the matrix looked structured. After this paper, we have a reason: the structure is connected to curve geometry and intrinsic curvatures.

It also makes HAVOK more practical. sHAVOK gives more stable and accurate models when data are short, noisy, or limited. This is important because real experimental data are usually not long, clean, and perfectly sampled.

---

## Module 3: Method and Purpose

### Goal

The goal is to explain the special matrix structure seen in HAVOK and then use that explanation to improve the algorithm.

In short:

```text
explain the structure first,
then enforce the structure to get better models.
```

### High-Level Approach

The paper follows this path:

1. Start from the standard HAVOK setup with a measured time series.
2. Build a Hankel matrix from time-delay copies of the signal.
3. Apply SVD to get low-dimensional delay coordinates.
4. Show that these coordinates are connected to the Frenet-Serret frame of a trajectory.
5. Use this connection to explain why the linear dynamics matrix should be antisymmetric and tridiagonal.
6. Modify HAVOK into sHAVOK so the learned model better preserves this structure.

The most important conceptual step is that the HAVOK matrix is interpreted geometrically. Its sub- and super-diagonal entries are not just fitted coefficients; they can be read as curvature-like quantities.

### Why use this approach?

The original HAVOK method already worked well, but it did not fully explain why the learned matrices had such clean structure.

The Frenet-Serret connection gives a theoretical explanation and also suggests a practical algorithmic improvement. If the correct model should have this structure, then the algorithm should be designed to preserve it, especially when the data are not ideal.

---

## Module 5: Important Formulas

### Hankel matrix from a time series

```math
\mathbf{H}
=
\begin{bmatrix}
x(t_1) & x(t_2) & \cdots & x(t_n) \\
x(t_2) & x(t_3) & \cdots & x(t_{n+1}) \\
\vdots & \vdots & \ddots & \vdots \\
x(t_m) & x(t_{m+1}) & \cdots & x(t_{m+n-1})
\end{bmatrix}
```

This stacks delayed copies of one measured signal into a matrix that contains short histories of the system.

### SVD of the Hankel matrix

```math
\mathbf{H} = \mathbf{U}\Sigma\mathbf{V}^T
```

This decomposes the delay data into dominant spatial-delay patterns and time-varying coordinates.

### Standard HAVOK forced linear model

```math
\frac{d}{dt}\mathbf{v}(t) = \mathbf{A}\mathbf{v}(t) + \mathbf{B}v_r(t)
```

This models the leading delay coordinates linearly while treating the last coordinate `v_r(t)` as a forcing term.

### Frenet-Serret frame dynamics

```math
\frac{d}{dt}
\begin{bmatrix}
\mathbf{e}_1 \\
\mathbf{e}_2 \\
\mathbf{e}_3 \\
\vdots
\end{bmatrix}
=
\begin{bmatrix}
0 & \kappa_1 & 0 & \cdots \\
-\kappa_1 & 0 & \kappa_2 & \cdots \\
0 & -\kappa_2 & 0 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
\begin{bmatrix}
\mathbf{e}_1 \\
\mathbf{e}_2 \\
\mathbf{e}_3 \\
\vdots
\end{bmatrix}
```

This shows why the natural geometry of a curve leads to an antisymmetric tridiagonal dynamics matrix.

### Structured HAVOK idea

```math
\mathbf{H}_1 = \mathbf{U}_1\Sigma_1\mathbf{V}_1^T,
\qquad
\mathbf{H}_2 = \mathbf{U}_2\Sigma_2\mathbf{V}_2^T
```

sHAVOK applies SVD separately to two time-shifted Hankel matrices so the fitted dynamics better preserve the expected geometric structure.

---

## Important Results and Notes

The most important result is that the paper explains the structured HAVOK matrix.

The key interpretation is:

```text
HAVOK's antisymmetric tridiagonal matrix comes from the Frenet-Serret frame,
and the off-diagonal entries correspond to intrinsic curvatures.
```

The paper also shows that increasing sampling frequency and using more data makes the HAVOK matrix more clearly antisymmetric and tridiagonal.

For sparse sampling, the paper shows that interpolation can help recover the expected structure.

The sHAVOK method is tested on Lorenz, Rössler, and double pendulum systems, plus real-world examples including double pendulum measurements and measles outbreak data.

The main practical result is that sHAVOK produces more structured and more stable models than standard HAVOK when using short trajectories. In the examples, sHAVOK eigenvalues are closer to the long-trajectory reference model than standard HAVOK.

For HAVOK work, this paper is useful because it gives a reason to care about the structure of the learned `A` matrix. If the model is far from antisymmetric and tridiagonal, that may indicate insufficient data, poor sampling, noise, or a rank/delay choice problem.

---

## Final Takeaway

This paper is basically the "why does HAVOK look like that?" paper.

The answer is that HAVOK is not just producing an arbitrary linear model. Under the right conditions, its delay-coordinate basis is connected to the Frenet-Serret frame, and the learned linear matrix reflects the intrinsic geometry of the trajectory.

The practical message is:

```text
The structure of the HAVOK matrix is meaningful,
and enforcing that structure can make the model more stable with limited data.
```

For your project, this matters because it gives a way to judge whether a HAVOK model is geometrically well-behaved, not only whether it has low reconstruction error.
