---
layout: post
title: "Maxwell's Equations, Three Ways"
date: 2026-11-11
---

## What this is

[optical-em-solver](https://github.com/oluseye-akomolede/optical-em-solver/tree/7248447) is a three-dimensional electromagnetic solver platform for optical and RF structures. Give it a meshed model and a material, and it returns the electromagnetic answer for that structure: the frequencies a cavity holds, the field a pulse leaves behind, the bands a periodic pattern supports. Three solvers sit behind one web app: a frequency-domain eigensolver on tetrahedral meshes, a time-domain finite-difference solver on a graded voxel grid, and a resonance extractor that turns a decaying time signal into frequencies and quality factors with error bars. The app takes the upload, estimates the size of the problem and creates a job. The job runs on a small self-hosted Kubernetes cluster, or on an NVIDIA Jetson Orin Nano when it needs a GPU ([README.md, lines 3-17](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/README.md#L3-L17)). It is a working research platform, not a product. Everything I claim about the code below is a claim about that commit.

## What it is for

The platform answers three questions, and they are not the same question. Which optical frequencies does this structure hold? That is an eigenvalue problem, and it is what the frequency solver does. What happens when a short pulse of light hits it, and what radiates away? That is an open-boundary problem in time, and it is what the time-domain solver does. How does light propagate through a periodic array of such structures, and where is the band gap? That is a band-structure problem at each wavevector, and both solvers can answer it, which is the point. Someone designing a cavity, a metasurface or a photonic crystal asks the first and third. Anyone checking how a thin gold film reflects across a pulse band asks the second ([README.md, lines 23-28](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/README.md#L23-L28)). What I wanted was to ask those questions about a real geometry with real material data attached, not only about a toy shape in a paper.

## What it is good at

Only claims a test or a line of the README backs.

A rectangular cavity's spectrum is known in closed form, and the frequency solver returns the first ten modes of a 10 by 7 by 5 millimetre box within 3 per cent, with the accidental and symmetry-required degeneracies in place and no spurious mode among them. In the time domain, a dipole in a closed lossless box produces work and stored energy that agree to within three parts in ten million, a dipole in free space reads its Larmor power, and a dipole above a conducting plane follows the image-dipole law within 0.05 across a sweep of heights. The two discretisations agree with each other: a Bloch-periodic pulse run and the band solver land within 1 per cent at the same wavevector, for both polarisations, and the same machinery reproduces the MPB tutorial's published gap to within 1 per cent of midgap ([README.md, lines 23-28](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/README.md#L23-L28)). The resonance extractor's error bars are calibrated against the Cramér-Rao bound and against simulated noise, with a median actual error of 0.67 to 0.8 of the reported one sigma. Those are the claims I would defend, and each is backed by a test or a benchmark at that commit.

The code, the tests and the benchmarks described here were largely built by AI agents working under my direction. I have not independently re-derived every equation, and where I say a result is checked, the check is in the repository at that commit.

The rest of this post takes one part of that machine in depth: how the solvers turn two curl equations into numbers, what each one can be checked against, and where I know it is weak.

The equations are Maxwell's, in the frequency domain,

$$
\nabla \times \nabla \times \mathbf{E} = \frac{\omega^2}{c^2}\,\varepsilon_r\,\mathbf{E},
$$

and in the time domain,

$$
\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \qquad
\nabla \times \mathbf{B} = \mu_0 \varepsilon_0 \varepsilon_r \frac{\partial \mathbf{E}}{\partial t},
$$

with no free charge, no currents except those I impose, no magnetic response, no gain and no nonlinearity. That covers a fair amount of nanophotonics and nothing at all about electrons. When I want the electronic structure I use a different solver entirely.

## The frequency problem: edge elements and a null space

For a closed cavity, the frequency-domain problem is an eigenvalue problem. Multiply by a test field $$\mathbf{v}$$, integrate over the volume, and integrate the left side by parts:

$$
\int_V (\nabla \times \mathbf{v}) \cdot (\nabla \times \mathbf{E})\,dV
= k^2 \int_V \varepsilon_r\, \mathbf{v} \cdot \mathbf{E}\,dV,
\qquad k = \frac{\omega}{c}.
$$

On a perfectly conducting boundary the tangential electric field vanishes, so the boundary term from integration by parts is zero and the problem needs no boundary condition beyond that. Discretise the field on a tetrahedral mesh, expand $$\mathbf{E}$$ in basis functions, and the equation becomes a generalised matrix eigenvalue problem,

$$
A\mathbf{e} = k^2 B\mathbf{e},
$$

where $$A$$ is the curl-curl matrix and $$B$$ is the mass matrix weighted by $$\varepsilon_r$$. In the code, each tetrahedron contributes a constant curl per basis function and a volume factor $$|\det J|/6$$, and the permittivity multiplies the mass block only, element by element ([mse_fem.py, lines 142-178](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_fem.py#L142-L178)). That is the whole of how multiple materials enter the frequency solver: the operator does not change, the inner product does.

The choice of basis function is the part that matters. The obvious choice on a tetrahedron is a nodal element, with one unknown per vertex per vector component, continuous across elements. It produces spurious modes. The reason is structural: a nodal vector field enforces continuity of every component, which is stronger than Maxwell requires and inconsistent with the function space the curl operator lives in. The correct family is the edge element, introduced by Nédélec in 1980 (Numerische Mathematik 35, 315), with degrees of freedom on edges, tangential continuity across faces, and no constraint on the normal component. Its lowest order form is the Whitney form of 1957 (Whitney, *Geometric Integration Theory*, Princeton University Press). Edge elements have a property I rely on constantly: the gradient of any continuous scalar field is represented exactly in the space. That is what makes the null space exact rather than approximate.

Which brings me to the difficulty. If $$\mathbf{E} = \nabla \phi$$ for any scalar $$\phi$$, then $$\nabla \times \mathbf{E} = 0$$, so $$A\mathbf{e} = 0$$ and the eigenvalue is zero. The gradient fields are not a numerical artefact to be filtered out at the end; they are an exact, enormous null space of the discrete operator, one dimension for every interior vertex. The physical modes I want sit just above it.

You cannot simply ask a shift-invert eigensolver for the smallest eigenvalues. The shift is placed at $$(\pi/D)^2$$, where $$D$$ is the bounding box diagonal and the first physical mode is expected nearby ([mse_eigen.py, line 40](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_eigen.py#L40)), and the factorisation is of $$K = A + sB$$ with a positive shift $$s$$. That matrix is positive definite, since $$A$$ is positive semidefinite and $$B$$ is positive definite, so it factorises even though $$A$$ alone does not. The gradient subspace is then removed by a projector built from the discrete gradient $$G$$ and the mass matrix,

$$
P = I - G\,(G^{\mathsf T} B\, G)^{-1} G^{\mathsf T} B,
$$

so that the solver works in the complement of the null space ([mse_eigen.py, lines 53-61](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_eigen.py#L53-L61)). The weight in that projector is the permittivity-weighted mass matrix, not the vacuum one. This looks like a detail and is not: with the vacuum weight the projector leaks at the $$3\times10^{-2}$$ level, and with the correct weight $$G^{\mathsf T} B P$$ vanishes to machine precision. The eigenvalues change by about $$3\times10^{-3}$$ relative either way, which is exactly why a test that only compared spectra could not tell a correct projector from a wrong one. The test asserts against the operator instead.

After the solve, any eigenvalue below $$10^{-6}$$ of the largest returned one is rejected. Because the null space was projected out before the solve, that test is an assertion rather than a cleanup: there should be nothing there to find ([mse_eigen.py, lines 24-26](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_eigen.py#L24-L26)).

What I can check it against is a rectangular cavity with perfectly conducting walls, where the answer is known in closed form:

$$
k^2 = \left(\frac{m\pi}{a}\right)^2 + \left(\frac{p\pi}{b}\right)^2 + \left(\frac{q\pi}{d}\right)^2,
$$

with at most one of the three indices allowed to be zero. For a box of 10 by 7 by 5 millimetres the first ten in-medium wavenumbers are 0.5478, 0.7025, 0.7721, 0.7721, 0.8336, 0.8336, 0.8886, 0.9510, 0.9955 and 0.9955 per millimetre. The solver must return all ten within 3 per cent, with no zero or spurious mode among them ([test_maxwell_solver.py, lines 200-216](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/tests/test_maxwell_solver.py#L200-L216)). Two of those values are worth naming: 0.7721 is an accidental degeneracy between the (2,1,0) and (0,1,1) modes, and 0.8336 is the transverse-electric and transverse-magnetic pair (1,1,1) that must appear twice. A code that gets the selection rule wrong returns only one of the pair, or returns a spurious mode at zero.

Two more tests pin the units. Refilling the same box with index 1.5 must leave every in-medium wavenumber unchanged to $$10^{-9}$$ and scale every frequency by exactly $$1/1.5$$. Describing the same physical object in centimetres instead of millimetres must give the same hertz. Neither test is interesting when it passes and both are diagnostic when it fails.

## The time problem: staggered grids and a stability limit

The frequency solver gives me modes of a closed or periodic structure. It does not give me a pulse travelling through an open one, and it does not give me the radiation escaping. For that I step Maxwell's equations in time on a Yee grid, the scheme Kane Yee published in 1966 (IEEE Transactions on Antennas and Propagation 14, 302). Each electric component is offset in one direction and each magnetic component in the other two, so every E sample is encircled by four H samples and the other way round. The two curl equations become explicit updates that leapfrog over each other:

$$
\mathbf{B}^{n+1/2} = \mathbf{B}^{n-1/2} - \frac{\Delta t}{\mu_0}\,\nabla \times \mathbf{E}^{n},
\qquad
\mathbf{E}^{n+1} = \mathbf{E}^{n} + \frac{c^2\,\Delta t}{\varepsilon_r}\,\nabla \times \mathbf{B}^{n+1/2}.
$$

The staggering is not cosmetic. It places the differences at the right points in space so that the discrete curl of a discrete gradient is identically zero, which preserves the same null space structure as the continuous problem, and it makes the scheme second order accurate in space and time.

The price is a stability condition. For a uniform grid,

$$
\Delta t \le \frac{1}{c\sqrt{\Delta x^{-2} + \Delta y^{-2} + \Delta z^{-2}}},
$$

and the code takes 0.99 of that limit as a safety factor ([fdtd.py, line 88](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/fdtd.py#L88)). On a non-uniform grid the bound changes along each axis, and with cells that vary across the domain between the metal and the far field, the timestep is set by the smallest cell anywhere. That is the real cost of local refinement in time domain: one bad cell slows the whole run.

Open boundaries are handled with a convolutional perfectly matched layer, the CPML formulation of Roden and Gedney (2000, Microwave and Optical Technology Letters 27, 334); that reference is mine, the code does not cite a paper. It is a stretched-coordinate layer: the coordinate normal to the boundary is continued into the complex plane, and the resulting convolution is carried by a recursive update of auxiliary variables, one per field component for each open axis it is stretched along. In this code the stretching is $$\kappa = 1$$ with $$\alpha = 0$$, the conductivity grading is polynomial of order 2, and the design reflection is $$10^{-12}$$ ([fdtd.py, lines 92-105](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/fdtd.py#L92-L105); the recursion is at [lines 3315-3316](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/fdtd.py#L3315-L3316)). Measured on a one-dimensional reduction of the same update, the worst reflected amplitude is $$3.6\times10^{-3}$$ at 75 degrees incidence with 40 cells per wavelength, and $$3.0\times10^{-4}$$ at normal incidence with 20 cells per wavelength. Those are the numbers I quote when someone asks whether the absorbing boundary is good enough, and the angle matters: grazing incidence is where it is worst.

Metals need a frequency-dependent permittivity, and the update has to stay in the time domain. I use a Lorentz-Drude model,

$$
\varepsilon_r(\omega) = 1 + \sum_j \frac{f_j\,\omega_{p}^2}{\omega_j^2 - \omega^2 - i\gamma_j \omega},
$$

and integrate the resulting polarisation with an auxiliary differential equation stepped alongside the fields ([fdtd.py, lines 504-593](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/fdtd.py#L504-L593)). Three parameter sets for gold are included, fitted to Rakić et al. (Applied Optics 37, 5271, 1998), to the Palik handbook values, and to Johnson and Christy. Each is measured against the data it was fitted to. The Palik fit deviates by at most 1.4951 per cent at 652.5 nanometres, and the Johnson-Christy fit by at most 0.4080 per cent at 892.0 nanometres.

The validation I trust most in the time domain is not a spectrum. It is an energy ledger. A dipole is injected into a closed lossless box and the work done by the source, computed from the field values on either side of the injection at each step, is compared with the energy the grid actually gained. The ratio is 0.9999997. The one-step-out variant of the same formula gives negative 40.2180, which is 40 times too large and the wrong sign, and that is the version a reasonable person writes first ([test_fdtd_purcell.py, lines 25-36](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/tests/test_fdtd_purcell.py#L25-L36)). With the correct formula a dipole in free space reads its Larmor power, and a dipole above a conducting plane follows the image-dipole law within 0.05 of it across a sweep of heights. The fixture's own sweep runs +2.9 per cent at gap 7 and -4.5 per cent at gap 10, which the test attributes to the source's own cell rather than the law, and it prints the sweep rather than hiding it.

A time-domain run also gives Purcell factors in one broadband pulse instead of one continuous-wave run per frequency. In vacuum the factor must be 1 across the band, and it must not drift with the mesh or grow as the square of wavelength over cell size ([test_fdtd_purcell.py, lines 68-90](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/tests/test_fdtd_purcell.py#L68-L90)).

## Getting frequencies back out of a ringdown

A time-domain run does not hand you a frequency. It hands you a decaying signal at a point, a sum of damped sinusoids,

$$
E(t) = \sum_k a_k e^{-\gamma_k t}\cos(2\pi f_k t + \varphi_k),
$$

and you have to estimate $$f_k$$ and $$\gamma_k$$ from a few thousand samples. I use two classical methods and compare them. The first is filter diagonalisation, in the form given by Mandelshtam and Taylor (1997, Journal of Chemical Physics 107, 6756) and implemented in Harminv, which turns the signal into a small matrix eigenproblem through a windowed Fourier basis. The second is the matrix pencil of Hua and Sarkar (1990, IEEE Transactions on Acoustics, Speech and Signal Processing 38, 814), which builds two Hankel matrices from the samples, takes a singular value decomposition to choose the model order, and solves a generalised eigenproblem whose poles are $$u_k = \exp(-2\pi i f_k \Delta t - \gamma_k \Delta t)$$.

The quality factor follows from the pole, $$Q = \pi f/\gamma$$. Both methods can produce a plausible looking list of modes from noise, so the interesting question is not whether they work at high signal to noise. It is what their error bars mean.

I report a one-sigma uncertainty from the Cramér-Rao bound for a damped sinusoid in white noise: the Fisher information for frequency, inverted, using the noise variance estimated from the record itself. That bound has a closed short-record form,

$$
\sigma_f \approx \frac{1}{2\pi T}\sqrt{\frac{6}{\mathrm{SNR}\cdot N}},
$$

which assumes the decay is slow compared with the record length. At $$\gamma T = 0.5$$ it is 20 per cent optimistic ([docs/resonances.md, lines 135-138](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/docs/resonances.md#L135-L138)), and the code uses the full Fisher expression instead.

Then I checked the bars against simulated noise, which is the part I would want a reader to hold me to. Over a sweep of signal to noise ratios and both real and complex records, the median actual error is 0.67 to 0.8 of the reported one sigma. For an efficient Gaussian estimator the expected value is 0.674, so the bars are neither loose nor optimistic. The default three-sigma bar held at every seed from 10 decibels upward on complex records, with the largest excursion in 240 draws at 3.1 sigma. On real records one seed sits at 3.7 to 4.3 sigma at every signal to noise level, which looks like a failure until you notice it is the same noise realisation scaled; against the bound computed from the true noise variance it is 3.4 to 3.5 sigma, a genuine tail draw rather than a broken bar.

The raw estimators are a different story, and this is why the code polishes the result before reporting it. At 20 decibels the raw filter diagonalisation scatters at 2.5 to 3.5 times the bound, with a median of 1.76 sigma on complex records, and the raw capped matrix pencil at about twice the bound, with a median of 1.34. Bars drawn at the bound would not cover them. With the polish on, they do.

At zero decibels the picture breaks down. One seed in 40 was 5.4 sigma off in $$Q$$, because at a per-sample signal to noise ratio of 1 the likelihood in frequency and decay rate is no longer single peaked, and $$Q$$ has a heavy tail when the decay rate is only determined to 12 per cent. The test asserts the three-sigma bound at 0, 10, 20, 40 and 60 decibels with fixed seeds, and the documentation says that a reseeded run can legitimately see a three-sigma excursion once in a few hundred draws. I would rather have that sentence in the docs than a test that quietly passes by widening its own bar.

## When two methods disagree

The strongest check I have is not any single benchmark. It is that two unrelated discretisations of the same equations agree.

The band structure of a two-dimensional photonic crystal is computed two independent ways: by the time-domain solver with a Bloch-periodic pulse, and by the frequency-domain band solver at each wavevector. They must agree within 1 per cent at the same wavevector, for both polarisations, with every excitable band in the pulse window present ([README.md, line 26](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/README.md#L26)). A deliberately wrong mapping from wavevector to phase shifts every line by far more than 1 per cent, which is what makes the check meaningful: it fails on the error a symmetry check cannot see. The same machinery reproduces the published band gap of the MPB tutorial's triangular rod lattice to within 1 per cent of midgap.

The frequency solver's convergence is measured as an order, not as a number. Against a production mesh, the relative error in the band frequency is $$6.9\times10^{-4}$$ at 16 divisions, $$1.7\times10^{-4}$$ at 32, and $$4.3\times10^{-5}$$ at 64. Each halving divides the error by 4.0, so the observed order is 2.00 ([test_pc.py, lines 874-882](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/tests/test_pc.py#L874-L882)). The production mesh at 40 divisions sits at $$1.2\times10^{-4}$$, which is why the test does not use a fixed $$10^{-4}$$ bar: that bar would measure one mesh, not the operator. The plan document for this work is blunt about the ceiling: $$10^{-6}$$ is not achievable on a first-order mesh, and only the zero eigenvalue at the Γ point is exact ([docs/photonic-crystal-plan.md, lines 23-25](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/docs/photonic-crystal-plan.md#L23-L25)). An error that is flat under refinement is the signature of a unit mistake, and this code has had one: a comparison that read raw wavenumbers against wavevectors in units of $$2\pi/a$$, which sat about 1 per cent low at $$\varepsilon = 4$$ and never converged.

There is also a mistake in my own measurement history that I keep as a warning. The mesh's band at $$k_x = 0$$ is not the band edge. Comparing the two gave 5.7 per cent, 0.28285 against a true 0.30000, and it looked like a solver error when it was an instrument error ([test_pc.py, lines 870-873](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/tests/test_pc.py#L870-L873)). Most of the time I have spent debugging this solver has gone into questions of the form "is the thing I am comparing this to the same quantity?"

## Where the mesh is the limit

All three solvers inherit the mesh, and the mesh is the least settled part of the pipeline. The README says so directly: "Mesh generation for arbitrary imported geometry is the least settled part" ([README.md, line 51](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/README.md#L51)).

I can be specific. For a structure with a 12 nanometre layer, I asked the mesher for a minimum feature of 3 nanometres. It returned 605,240 tetrahedra and still left an element 12.0 nanometres tall in that layer. The code says what this is: the sizing field is a target, not a bound ([mse_meshing.py, lines 1817-1821](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_meshing.py#L1817-L1821)). When I split the layer into three sub-slabs, the mesher returned 108,378 tetrahedra with a worst element span of 4.00 nanometres, still above the 3 requested. The sizing field does not refuse the request; it reports a mesh that satisfies a coarser one. That is a real gap between what the interface promises and what the mesher delivers, and it is why the minimum feature is treated as a target rather than a guarantee.

The time-domain grid has its own discipline. Neighbouring cell spacings may differ by at most a factor of 1.3, a layer thinner than the coarse cell gets at least 4 cells across it, and the target growth per cell is 0.2, which leaves margin under the hard ratio ([fdtd_grid.py, lines 35-40](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/fdtd_grid.py#L35-L40)). The grid refuses overlapping parts before any grid exists: parts may touch but not share volume ([fdtd_voxel.py, lines 194-202](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/fdtd_voxel.py#L194-L202)). Voxelisation rays are jittered by about $$10^{-7}$$ of the scene size along three incommensurate directions, because an axis-aligned box has mesh edges exactly on grid lines and a ray through an edge would be counted twice or not at all ([fdtd_voxel.py, lines 14-22](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/fdtd_voxel.py#L14-L22)).

## How it runs

The web app does not solve anything. It takes an upload, estimates the tetrahedron count, and creates a Kubernetes job with a manifest that carries the geometry and a materials table as a file, so the solver and the app cannot disagree about what a part is called ([web/jobs.py, lines 45-52](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/web/jobs.py#L45-L52)). If the actual tetrahedron count exceeds the CPU limit, the solver raises a specific error so the app can re-route the job to the GPU queue; above an absolute cap it raises one I can read ([mse_maxwell_solver.py, lines 399-430](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_maxwell_solver.py#L399-L430)).

The backend is resolved once, in the function that runs the pipeline, and passed explicitly to the eigensolver, so the mesh-count guard and the factorisation cannot disagree about which machine is running ([mse_maxwell_solver.py, lines 244-248](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_maxwell_solver.py#L244-L248)). An unknown value raises rather than defaulting to the CPU, and a GPU run on a machine without the CUDA stack fails with a message naming what is missing, because a silent fall back to the CPU turns "the GPU run was slow" into a mystery with no error in the log ([mse_backend.py, lines 21-30](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_backend.py#L21-L30)). The GPU path routes both of the matrices it factorises to cuDSS, not just the larger one: at 266 thousand tetrahedra the CPU-side factorisation of the second matrix alone takes 19 seconds on the Jetson ([mse_backend.py, lines 5-12](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_backend.py#L5-L12)), and the projection solves dominate the run, so taking only one matrix to the GPU would leave most of the time on the table.

## What I do not know

In the published version the materials library has 34 parameters and 11 of them have no source. A test for that fact is a strict expected failure: the suite stays green while the 11 gaps exist and turns red if the count changes, and its message lists them: the static permittivities of aluminium oxide, hafnium oxide, gold and silver, with the semiconductor values covered by an Ioffe/NSM reference ([test_materials_library.py, lines 136-140](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/tests/test_materials_library.py#L136-L140)). Every one of the 11 is tagged as an assumption in the materials dump, and the count in the test is derived from the library rather than typed in, so it cannot drift from the data without a failure. A library entry without a citation is rejected with an error naming the field.

The resonance extractor is implemented and calibrated, and no production code outside its module and tests calls it, so it is not yet wired into the app. The semiconductor device stack is not in the public sync yet. Neither of those is a physics problem and both are reasons not to describe the platform as complete.

The list of what I would call proven is the one at the top of this post. Everything else is a claim in progress, and the parts I have not checked are the parts I would least like someone to assume.

## References

1. K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media," *IEEE Transactions on Antennas and Propagation* 14, 302 (1966).
2. J. C. Nédélec, "Mixed finite elements in $$\mathbb{R}^3$$," *Numerische Mathematik* 35, 315 (1980).
3. H. Whitney, *Geometric Integration Theory* (Princeton University Press, 1957).
4. J. A. Roden and S. D. Gedney, "Convolution PML (CPML): an efficient FDTD implementation of the CFS-PML for arbitrary media," *Microwave and Optical Technology Letters* 27, 334 (2000).
5. A. D. Rakić, A. B. Djurišić, J. M. Elazar and M. L. Majewski, "Optical properties of metallic films for vertical-cavity optoelectronic devices," *Applied Optics* 37, 5271 (1998).
6. E. D. Palik, *Handbook of Optical Constants of Solids* (Academic Press, 1985), p. 290.
7. P. B. Johnson and R. W. Christy, "Optical constants of the noble metals," *Physical Review B* 6, 4370 (1972).
8. V. A. Mandelshtam and H. S. Taylor, "Harmonic inversion of time signals and its applications," *Journal of Chemical Physics* 107, 6756 (1997).
9. Y. Hua and T. K. Sarkar, "Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in noise," *IEEE Transactions on Acoustics, Speech and Signal Processing* 38, 814 (1990).
10. S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation Theory* (Prentice Hall, 1993), chapter 3.

## Code references

All links point to commit `7248447`:

- Curl-curl and mass element matrices: [`dockerfiles/maxwell_solver/mse_fem.py`](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_fem.py)
- Eigenvalue solve, shift and projector: [`dockerfiles/maxwell_solver/mse_eigen.py`](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/mse_eigen.py)
- Cavity and scaling tests: [`tests/test_maxwell_solver.py`](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/tests/test_maxwell_solver.py)
- Time-domain scheme, absorbing layer and gold dispersion: [`dockerfiles/maxwell_solver/fdtd.py`](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/fdtd.py)
- Energy ledger and Purcell tests: [`tests/test_fdtd_purcell.py`](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/tests/test_fdtd_purcell.py)
- Resonance calibration: [`docs/resonances.md`](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/docs/resonances.md)
- Band convergence and the category-error note: [`tests/test_pc.py`](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/tests/test_pc.py)
- Grid grading rules: [`dockerfiles/maxwell_solver/fdtd_grid.py`](https://github.com/oluseye-akomolede/optical-em-solver/blob/7248447/dockerfiles/maxwell_solver/fdtd_grid.py)
