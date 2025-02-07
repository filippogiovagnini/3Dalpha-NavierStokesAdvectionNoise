# 3Dalpha-NavierStokesAdvectionNoise

#### In this repository we implement the method from "A uniform particle approximation to the Navier-Stokes-alpha models in three dimensions with advection noise"

Let us consider the following:
$$
\begin{cases}
d\omega+ \lie_{u} \omega dt + \sum\limits_{k \in \N} \lie_{\sigma_k} \omega \circ dW^k_t= \nu \Delta \omega, \\
u = (\indicator - \alpha^2 \Delta)^{-1} K \star \omega, \\
\omega(0, \cdot) = \omega_0.
\end{cases}
$$






<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
