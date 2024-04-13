# odeESS

Source code for algorithm introduced in "Rapid Integrator for a Class of Multi-Contact Systems"
DOI:[https://doi.org/10.48550/arXiv.2402.00279](https://doi.org/10.48550/arXiv.2402.00279)

## Usage

The integrator is implemented in the class `odeESS`, contained within
[ess.py](ess.py). In its current implementation, it requires `odedp5.py`,
`numpy`, and`scipy` to run. However, the odeDP5 dependency can be removed by
altering the `dp5` attribute of `odeESS`.

For an N-dimensional state space with M guards, requires
- `f` -- (N,M)->N vector ode mapping state, region to state derivative
- `h` -- N->M vector of the M guard functions
- `Dh` -- jacobian of `h`
- `eps` -- a precision parameter for integration speedup
```
>>> o = odeESS(f, h, Dh, eps)
>>> t, y = o(x0, t0, tf)
```

## Authors

Marion Anderson [marand@umich.edu](emailto:marand@umich.edu), Shai Revzen [shrevzen@umich.edu](emailto:shrevzen@umich.edu)

## License

The license governing this code is the GNU GPLv3.0, as per GPL-LICENSE.md, with one additional provision set forth below.

If you use this code for academic work, you must cite the either the arXiv publication (https://doi.org/10.48550/arXiv.2402.00279) or the peer-reviewed
publication (TBD).

