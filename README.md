# 3Dalpha-NavierStokesAdvectionNoise

#### In this repository we implement the method from "A uniform particle approximation to the Navier-Stokes-alpha models in three dimensions with advection noise"

Let us consider the following:

$$
d\omega+ £_{u} \omega dt + \sum\limits_{k \in \mathbb{N}} £_{\sigma_k} \omega \circ dW^k_t= \nu \Delta \omega,
$$
$$
u = (\mathbb{1} - \alpha^2 \Delta)^{-1} K \star \omega,
$$
$$
\omega(0, \cdot) = \omega_0.
$$

where $£$ is the usual Lie derivative. In ?? a new method to approximate this system has been proven. We implement it here.
