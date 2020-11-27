# Eigenvectors as fixed points

Credits: Maseim Kenmoe, Ronald Kriemann, Matteo Smerlak, Anton Zadorin

Preprint: https://arxiv.org/abs/2002.12872

## Full spectrum

- Symmetric: compare with Jacobi (`syevj` in cuSOLVER) rather than Hessenberg-based methods `syevX`
- Non-symmetric: much faster than `geev` within its domain

## One eigenpair

- Anderson acceleration: https://users.wpi.edu/~walker/Papers/anderson_accn_algs_imps.pdf
- QW partitioning: Mihalka, Szabados, Surjan, J. Chem. Phys. 146, 124121 (2017)
