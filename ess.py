#!/usr/bin/env python3
"""
This software is copyrighted material (C) Marion Anderson 2024

The license governing this code is the GNU GPLv3.0, as per GPL-LICENSE.md,
with one additional provision set forth below.

If you use this code for academic work, you must cite the either the arXiv
publication (https://doi.org/10.48550/arXiv.2402.00279) or the peer-reviewed
publication (TBD).

  OVERVIEW
    This file contains a special-purpose ODE integrator designed to accelerate
    integration of "Event-Selected" ODEs

    If you have matplotlib installed, running this file should generate
    plots of key unit tests.

  ROUTINE LISTINGS
    odeESS -- class implementing integrator algorithm

  SEE ALSO
    odedp5.py -- integrator class implementing default smooth integrator

  REFERENCES
    [1] M Anderson et al. "Rapid Integrator for a Class of Multi-Contact
        Systems". DOI: 10.48550/arXiv.2402.00279
    [2] S A Burden et al. “Event-Selected Vector Field Discontinuities Yield
        Piecewise-Differentiable Flows”. DOI: 10.1137/15M1016588.
    [3] G Council et al. "Representing and computing the B-derivative of an ECr
        vector field's PCr flow". DOI: 10.1115/1.4054481

  EXAMPLE
    For
      f -- (Nx1,Mx1)->Nx1 -- vector ode. maps state, guards -> derivative
      h -- Nx1->Mx1  -- M event functions collected as a vector function
      Dh -- Nx1->MxN -- jacobian function of h
      eps -- h function output threshold to trigger projection speedup

    The code becomes
      >>> o = odeESS(f, h, Dh, eps)
      >>> t, y = o(x0, t0, tf)
"""
import warnings
from copy import deepcopy

from numpy import (any, argmin, array, asarray, diff, dot, hstack, inf,
                   isscalar, linspace, newaxis, nonzero, printoptions, r_,
                   sign, vstack, zeros_like)
from scipy.linalg import det, inv

try:
  from .odedp5 import odeDP5
except ImportError:
  from odedp5 import odeDP5
# virtualenv import protections
try:
  from time import time as now
  assert callable(now)
except AssertionError:
  import time
  now = time.time

__all__ = ['odeESS', 'odeESSNumericWarning']


class odeESSNumericWarning(UserWarning):
  """Convenience warning category. Differs from UserWarning in name only."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)


def clean_formatwarning(message, category, *args, **kwargs):
  """Prevents warnings.warn printing the code for the warning print statement
  https://stackoverflow.com/questions/2187269/print-only-the-message-on-warnings
  """
  return f'{category.__name__}: {str(message)}\n'


warnings.formatwarning = clean_formatwarning  # cleaner warning printing


class odeESS:
  """ESS-aware integrator. Accelerates integration using a projection
    speedup when the trajectory is local to a discontinuity it will cross.

    Event-selected systems (ESS) have an N-dimensional state space
    partitioned by M guards into 2^M regions. The boundary between any 2
    adjacent regions is the zero-level set of the appropriate event function.
    Event functions must be at least once-differentiable. Furthermore, the
    trajectories local to an event function zero-level set must cross said
    zero-level set, for some notion of local.

    INPUTS:
      f -- (Nx1,Mx1)->Nx1 -- vector ode. maps state, guards -> derivative
      h -- Nx1->Mx1  -- M event functions collected as a vector function
      Dh -- Nx1->MxN -- jacobian function of h
      eps -- h function output threshold to trigger projection speedup

    EXAMPLE:
      >>> o = odeESS(f, h, Dh, eps)
      >>> t, y = o(x0, t0, tf)

    LOGGING:
      self.t, self.y, self.proj_log, self.event_log = [], [], [], []
      self.itercount = 0

    LOGGING TIME PERFORMACE:
      odeESS.tref stores the CPU time at the top of the first mainloop
      iteration after at least half the time interval has been integrated. Note
      that "half the time interval" is based off of trajectory (not CPU) time.
      odeESS automatically calculates half the time interval as (t1-t0)/2.

    NOTES:
      Dh can be explicit or numeric. It only needs to be a callable.

      `f` evaluated on `y` will be invoked as `f(y, h(y))`.

      Regular (non-hybrid) integration is accomplished using DOPRI5, a
      5th order integrator. To change, override `self.dp5`.
  """

  def __init__(self, f, h, Dh, eps, itermax=100):
    self.itermax = itermax
    self.f, self.h, self.eps, self.Dh = f, h, eps, Dh
    assert callable(self.f), 'f must be callable'
    assert callable(self.h), 'h must be callable'
    assert self.eps > 0, 'eps must be positive'
    assert callable(self.Dh), 'Dh must be callable'

    self.dp5 = odeDP5(lambda t, y, p: self.f(y, self.h(y)))
    self.dp5.event = self._evt

    # Logging vars
    # - init here for linter/readability
    self.t, self.y, self.proj_log, self.event_log = [None] * 4
    self.itercount = None
    self.tref = None
    self._reset_loggers()

  def _evt(self, t, y, p):  # pylint: disable=unused-argument
    """odeDP5-compatible event function. Checks if any h functions have
      crossed the eps boundary.

      Implemented as private member function for debugging visibility.
    """
    return any((self.h(y[0]) < -self.eps) & (self.h(y[1]) > -self.eps))

  def __call__(self, y0, t0, t1=None, dp5dt=None):
    """Integrate"""
    y0 = asarray(y0, dtype=float)
    self._reset_loggers()
    if t1 is None:
      assert len(t0) > 1, 'Must provide time interval'
      t, tf = t0[0], t0[-1]
    else:
      t, tf = t0, t1
    assert tf > t, 'Sample times must increase'
    self.t, self.y, y, itercount = [t], [y0.copy()], y0.copy(), 0
    crossed_mask = self.h(y0) >= 0  # init without already-crossed guards
    thalf = (tf - t) / 2

    # Integration Mainloop
    while (t < tf) and (itercount < self.itermax):
      # --------------------------------
      #   timing eval (at TOP of loop)
      # --------------------------------
      if (self.tref is None) and (t - t0 > thalf):  # isnan first should be faster
        self.tref = now()

      hy = asarray(self.h(y))
      # -----------------------
      #   Regular Integration
      # -----------------------
      if not any((hy > -self.eps)[~crossed_mask]):
        t, y = self.dp5(y, t, tf, dp5dt)
        t, y = t[1:], y[1:]  # remove duplicates
        if all(crossed_mask):  # enables integration of periodic trajectories
          crossed_mask = self.h(y[-1]) >= 0
      # -------------------
      #   Projection Step
      # -------------------
      else:
        y = y.copy()  # deepcopy to avoid projection step overwrites
        G = asarray(self.f(y, hy))
        e = self.Dh(y)
        eG = dot(e, G)
        dt = -hy / eG
        dt[crossed_mask] = inf  # ignore already-crossed states
        j = argmin(dt)
        dt = dt[j]
        dy = G * dt
        t += dt
        y += dy
        self.proj_log.append([len(self.t) - 1, dt, t] + list(dy))
        crossed_mask[j] = True  # ignore crossed guard from now on

        # Handle dopri5 overshoot
        if dt < 0:
          if isscalar(self.t[-1]):
            self.t.pop()
            self.y.pop()
          else:
            self.t[-1], self.y[-1] = self.t[-1][:-1], self.y[-1][:-1]

        # Numerical conditioning check
        n = nonzero(abs(eG * dt) > self.eps)[0]
        if len(n) > 0:
          with printoptions(precision=3):
            warnings.warn(f'dh>{self.eps:.2g} at y={y-dy} for guard(s) {n}',
                          odeESSNumericWarning)
      # ------------------
      #   Next Iteration
      # ------------------
      self.t.append(t)
      self.y.append(y)
      if not isscalar(t):
        t, y = t[-1], y[-1]
      self.event_log.append([len(self.t) - 1] + list(y))
      itercount += 1
    self.event_log = asarray(self.event_log)
    self.proj_log = asarray(self.proj_log)
    self.t, self.y = hstack(self.t), vstack(self.y)
    return self.t, self.y

  def _reset_loggers(self):
    """Reset logging variables"""
    self.t, self.y, self.proj_log, self.event_log = [], [], [], []
    self.itercount = 0
    self.tref = None


if __name__ == '__main__':
  # region
  print("""Running unit tests:

          (1) Piecewise-constant vector field
          (2) Pathologically non-constant vector field
          (3) Order test on 3D vector field
  """)
  from itertools import combinations
  from itertools import product as iterprod

  from matplotlib.gridspec import GridSpec
  from matplotlib.pyplot import *
  from numpy import *  # pylint: disable=ungrouped-imports,redefined-builtin
  from numpy import asarray, empty
  from scipy import __version__ as scipyv
  from scipy.linalg import det
  from scipy.linalg import expm as scipy_expm
  from scipy.linalg import inv
  from scipy.optimize import bisect

  if scipyv >= '1.9.0':
    expm = scipy_expm
  else:
    def expm(A, out=None):
      """Vectorizes traditional implementation of scipy.linalg.expm.
          INPUT:
          A -- ...N X N  -- Collection of square matrices to exponentiate
          OUTPUT:
          ...N x N
      """
      A = asarray(A)
      if out is None:
        out = empty(A.shape, dtype=A.dtype)
      if len(A.shape) <= 2:
        out[:, ...] = scipy_expm(A)  # [:, ...] to avoid memory realloc
      else:
        for ndx in iterprod(*[range(n) for n in A.shape[:-2]]):
          out[ndx] = scipy_expm(A[ndx])
      return out

  try:
    from itertools import pairwise
  except ImportError:
    from itertools import tee

    def pairwise(iterable):
      """pairwise('ABCDEFG') --> AB BC CD DE EF FG

          New in Python 3.10, so implement here just in case
          https://docs.python.org/3/library/itertools.html#itertools.pairwise
      """
      a, b = tee(iterable)
      next(b, None)
      return zip(a, b)

  def fvec(f, x, m, out=None):
    """vectorized an ode
    INPUTS:
      f -- (x,m) -> xdot -- derivative function
      x -- Nx... -- state vector
      m -- Mx... -- h(x) output; guard functions at `x`
      out -- Nx..., optional -- return-by-ref array
    OUTPUTS:
      Nx... -- state derivative evaluated at each `x`
    NOTES:
      - Striding doesn't work cleanly in higher dimension ndarrays,
        so a combinatorial indexing approach is required.
      - To get a full "column" of data in an N-D array, I need to
        pull the data out one element at a time (ugh) and put it into
        a buffer since `x[:,tuple]` returns a giant matrix.
    """
    assert x.shape[1:] == m.shape[1:], 'x,m must have same number of cols'
    if out is None:
      out = zeros_like(x)
    if x.ndim == 1:  # 1-D vector case
      out[...] = f(x, m)
    else:  # vectorized case
      M, N = len(m), len(x)
      mtup, xtup = zeros(M), zeros(N)  # pre-alloc buffer vars
      for ndx in ndindex(x.shape[1:]):
        for i in range(M):
          mtup[i] = m[(i,) + ndx]  # `h(x[:,ndx])`
        for i in range(N):
          xtup[i] = x[(i,) + ndx]  # `x[:,ndx]`
        derv = f(xtup, mtup)
        for i in range(N):
          out[(i,) + ndx] = derv[i]
    return out

  class AffineOrderTester:
    """Evaluate accuracy of numerical integrators on a hybrid system of
      affine ODEs defined as:
        f_i(y) = A_i @ y + B_i
      INPUTS:
        Adict -- dict of A matrices (linear component)
        Bdict -- dict of B matrices (affine component)
        h -- guard function: state -> number
      METHODS:
        solve_closedform, solve_ess, solve_dp5 -- y0, t0, tf -> trajectory
          Solve IVP using the closed form (matrix exponential), odeESS, or
          odeDP5, respectively.
      NOTES:
        Adict, Bdict must have identical keys and be numerically
        compatible.

        Adict, Bdict are indexed by `str(sign(h(y)).astype(int))`.
    """

    def __init__(self, Adict, Bdict, h, Dh):
      # Input Sanitization
      assert Adict.keys() == Bdict.keys(), 'Adict,Bdict keys differ'
      shape = asarray(list(Adict.values())[0]).shape
      for k, v in Adict.items():
        assert det(v) != 0, f'Adict[{k}] singular matrix!'
        assert asarray(
            v).shape == shape, f'Adict[{k}] inconsistent matrix shapes'
        assert len(Bdict[k]) == shape[0], f'Bdict[{k}] bad size'
      assert callable(h), 'h not callable'
      # Assign Vars
      self.Adict = deepcopy(Adict)
      self.Bdict = deepcopy(Bdict)
      self.h = h
      self.Dh = Dh

    @staticmethod
    def m2key(m):
      """Map `h(x)` to string for indexing `Adict`,`Bdict`."""
      m = sign(m)  # vf section is determined by sign
      m[m == 0] = 1  # on boundary is the next section
      key = str(m.astype(int))
      return key

    def _flow_factory(self, y0, m, t0):
      """Get closed-form flow callable for an IVP in a single octant.
        INPUTS:
          y0 -- initial condition
          m -- octant of ODE to use
          t0 -- initial time
        OUTPUTS:
          flow -- vectorized callable -- `y=flow(t)`
        NOTES:
          `m` is separate from `y0` to avoid numerical
          noise. Auto-determining `m` can cause the wrong ODE used
          in the solution.
      """
      A, B = self.Adict[self.m2key(m)], self.Bdict[self.m2key(m)]
      shift = inv(A) @ B
      ytilde = y0 + shift
      def flow(t): return (expm((A[..., newaxis] * (t - t0)).transpose(-1, 0, 1))
                           @ ytilde - shift).squeeze()
      return flow

    def solve_closedform(self, y0, t0, tf):
      """Solve IVP using the closed-form flow
        OUTPUTS:
          tsoln, ysoln -- time, state of closed-form solution
          cross_t  -- closed-form guard impact times
        NOTES:
          We are already know the closed-form flow from Linear
          Systems (see `self._flow_factory` also), so maximize
          accuracy when solving for the next octant:
            1. Flow IVP until tf, ignoring guards
            2. Compute the flow's impact time for ALL guards
            3. We want the guard with the shortest impact time
            4. Flow to said guard
            5. Update t0
            6. Loop
      """
      # Determine Crossing Times
      cross_t, flow_log, m = [-inf], [], self.h(y0)
      nguards = len(m)
      for _ in range(nguards):
        flow = self._flow_factory(y0, m, t0)
        flow_log.append(flow)

        tc, t = [], linspace(t0, tf)
        for i in range(nguards):
          side_of_guard = sign(self.h(flow(t))[i])
          no_sign_change = all(diff(side_of_guard) == 0)
          already_crossed = m[i] >= 0
          if no_sign_change or already_crossed:
            tc.append(inf)  # skip guards that flow won't hit
          else:
            tc.append(bisect(lambda t: self.h(flow(t))[i], t0, tf))

        n = argmin(tc)
        t1 = tc[n]
        m[n] = 1  # update crossing
        t0, y0 = t1, flow(t1)
        cross_t.append(t1)
      # last-mile
      cross_t.append(inf)
      flow_log.append(self._flow_factory(y0, m, t0))
      # Construct Callable

      def flow_out(st):
        out = []
        for i, (t0, t1) in enumerate(pairwise(cross_t)):
          t = st[(t0 <= st) & (st < t1)]
          if len(t) == 0:
            continue
          out.append(flow_log[i](t))
        return vstack(out)
      return flow_out

    def _fsingle(self, x, m):
      """(non-vectorized) state,octant -> derivative"""
      key = self.m2key(m)
      deriv = self.Adict[key] @ x + self.Bdict[key]
      return deriv

    def f(self, x, m):
      """(vectorized) state,octant -> derivative"""
      return fvec(self._fsingle, x, m)

    def solve_ess(self, y0, t0, tf, eps, **esskwargs):
      """Solve IVP using odeESS."""
      o = odeESS(self.f, self.h, self.Dh, eps)
      return o(y0, t0, tf, **esskwargs)

    def solve_dp5(self, y0, t0, tf, itermax=100):
      """Solve IVP using odeDP5."""
      o = odeDP5(lambda t, y, p: self.f(y, self.h(y)))
      o.event = lambda t, y, trj: not array_equal(
          sign(self.h(y[0])), sign(self.h(y[1])))
      tout, yout = o(y0, t0, tf)
      for _ in range(itermax):
        if tout[-1] >= tf:
          break
        t, y = o(yout[-1], tout[-1], tf)
        tout = r_[tout, t]
        yout = r_[yout, y]
      return tout, yout

  def plot2dvf(fn, x, dx, tref, yref, tcomp, ycomp):
    """plot 2d vector field
      INPUTS:
        fn -- figure name
        x -- grid of x coords for quiver plot
        dx -- grid of derivative for quiver plot
        tref -- Nx1 -- reference time values
        yref -- Nx2 -- reference yvalues
        tcomp -- Mx1 -- computation time values
        ycomp -- Mx2 -- computation yvalues
    """
    # Plot
    f = figure(fn)
    f.clf()
    gs = GridSpec(2, 4)
    ax1 = f.add_subplot(gs[:, :2])
    ax4 = f.add_subplot(gs[0, 2:])
    ax5 = f.add_subplot(gs[1, 2:])

    # Parameteric Plot
    ax1.set_title('$x-y$ plane')
    ax1.quiver(*x, *dx, alpha=0.5)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_aspect('equal')
    ax1.plot(*yref.T, 'cs-', lw=0.5, ms=4, label='DP5')
    ax1.plot(*ycomp.T, 'm.-', lw=0.3, ms=5, label='ESS')
    ax1.legend()

    # Time Plot
    dt = diff(tcomp)
    repndx = nonzero(~(dt > 0))[0]
    ax4.set_title('Computation Time Values')
    ax4.plot(tcomp, '.-', label='t')
    ax4.plot(dt, '.-', label='dt')
    l, = ax4.plot(repndx, dt[repndx], '*', label='repeats')
    ax4.plot(repndx, tcomp[repndx], '*', c=l.get_color())
    ax4.grid()
    ax4.legend()
    ax4.set_xlabel('time index [n]')

    # Comparison Plot
    for i in range(2):
      xlbl = '$x$' if i == 0 else '$y$'
      ax5.plot(tref, yref[:, i], 's-', label=xlbl + ' Ref', ms=4)
      ax5.plot(tcomp, ycomp[:, i], '.-', label=xlbl + ' Comp')
    ax5.legend()
    ax5.set_xlabel('time')
    ax5.grid()
    ax5.set_title('Time Series Comparison')

    if not f.get_tight_layout():
      f.set_tight_layout(True)
    return f

  def r2grid(r, n):
    """create grid of points given box size and num samples
      INPUTS:
        r -- how far to go to either side or the origin.
        n -- number of samples along a line
      NOTES:
        x1=linspace(-r,r,n)
    """
    x1 = linspace(-r, r, n)
    x11, x22 = meshgrid(x1, x1)
    x = r_[x11[newaxis], x22[newaxis]]
    return x

  def compute_sols(f, h, x0, tf, ncrossings, itermax=100, **odeESS_kwargs):
    """Compute reference and ESS soln
      INPUTS:
        f -- vector field callable for odeESS
        h -- guard vector callable for odeESS
        x0 -- initial condition
        tf -- runtime of integration
        ncrossings -- expected number of guard crossings (odeDP5 calls)
        itermax -- max number of odeESS iterations
        odeESS_kwargs -- kwargs for odeESS construction
      OUTPUTS:
          odeESS integrator instance
          tref, yref -- time, result of odeDP5
          tcomp, ycomp -- time, result of odeESS
    """
    # Solve Ours
    ocomp = odeESS(f, h, **odeESS_kwargs, itermax=itermax)
    tcomp, ycomp = ocomp(x0, 0, tf)

    # Solve Reference
    o = odeDP5(lambda t, y, p: f(y, h(y)))
    o.event = lambda t, y, trj: not array_equal(
        sign(h(y[0])), sign(h(y[1])))
    tref, yref = o(x0, 0, tf)
    for _ in range(ncrossings):
      if tref[-1] >= tf:
        break
      t, y = o(yref[-1], tref[-1], tf)
      tref = r_[tref, t]
      yref = r_[yref, y]

    return ocomp, tref, yref, tcomp, ycomp

  def test_const():
    """Test single guard crossing"""
    fn = 1

    def F(x, m):  # pylint: disable=unused-argument
      """PCr ode"""
      x, m = asarray(x), asarray(m)
      assert x.shape[1:] == m.shape[1:], f'x:{x.shape} m:{m.shape}'
      # index quadrants
      k = zeros(m.shape[1:], dtype=int)
      # k[(m[0] < 0) & (m[1] < 0)] = 0
      k[(m[0] >= 0) & (m[1] < 0)] = 1
      k[(m[0] >= 0) & (m[1] >= 0)] = 2
      k[(m[0] < 0) & (m[1] >= 0)] = 3
      # eval derv
      dx = zeros_like(x, dtype=float)
      dx[0] = 1.
      dx[1, k == 1] = -0.5  # +- is [1,-1]
      dx[1, k == 3] = 0.5  # -+ is [1,1]
      dx[0, k == 2] = 0.5  # -+ is [0.5,0]
      return dx

    def Dh(x):
      """guard jacobians"""
      out = ones((2,) + x.shape, dtype=float)
      out[1, 1] = -1.
      return out

    affine_offset = 0.1

    def h(x):
      """guard functions"""
      out = zeros_like(x, dtype=float)
      out[0] = x[0] + x[1] + affine_offset
      out[1] = x[0] - x[1] + affine_offset
      return out

    # Solve
    x0 = (-0.45, 0.15)
    eps = 0.15
    ocomp, tref, yref, tcomp, ycomp = compute_sols(
        lambda x, m: fvec(F, x, m), h, x0, 1, 4, itermax=6, Dh=Dh,
        eps=eps)

    # Plot
    x = r2grid(0.5, 10)
    dx = F(x, h(x))
    f = plot2dvf(fn, x, dx, tref, yref, tcomp, ycomp)
    ax = f.axes[0]
    xl, yl = array(ax.get_xlim()), array(ax.get_ylim())
    ax.plot(xl, xl + affine_offset, 'r', lw=3)
    ax.plot(xl, -xl - affine_offset, 'b', lw=3)
    ax.plot(xl, xl + eps + affine_offset, 'k--')
    ax.plot(xl, xl - eps + affine_offset, 'k--')
    ax.plot(xl, -xl + eps - affine_offset, 'k--')
    ax.plot(xl, -xl - eps - affine_offset, 'k--')
    ax.set_xlim(xl)
    ax.set_ylim(yl)

    return ocomp

  def test_nonconst():
    """non-constant vf unit test

    Want dy/dx to be non-constant in the region where dx/dt is constant.
    This allows testing how well the integrator splits the ode into parts.
    """
    fn = 2

    def h(x):
      x = asarray(x)
      out = zeros((2,) + x.shape[1:])
      out[0] = x[0] / 2.
      out[1] = -x[1] / 2.
      return out

    def Dh(x):
      x = asarray(x)
      out = zeros((2,) + x.shape, dtype=float)
      out[0, 0] = 1 / 2.
      out[1, 1] = -1 / 2.
      return out

    def fsingle(x, m):
      """non-vectorized vector ode
      x -- Nx1 -- state vector
      m -- Mx1 -- h(x) output
      """
      m = sign(m)
      m[m == 0] = 1  # on boundary is the next section
      out = zeros_like(m, dtype=float)

      # xdot - constant for simplicity
      if m[1] == -1:
        out[0] = 1
      else:
        out[0] = 0.5

      # ydot - nonconstant to strain integrator
      if array_equal([-1, -1], m):
        out[1] = -0.15 / (x[0]**2 + 0.1)
      else:
        out[1] = -0.5

      return out

    # Solve
    eps = 0.2
    def f(x, m): return fvec(fsingle, x, m)  # vectorized ode
    ocomp, tref, yref, tcomp, ycomp = compute_sols(
        f, h, (-0.4, 0.5), 1, 4, itermax=6, Dh=Dh, eps=eps)

    # Plot
    x = r2grid(0.5, 7)
    dx = f(x, h(x))
    fig = plot2dvf(fn, x, dx, tref, yref, tcomp, ycomp)
    ax = fig.axes[0]
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.hlines(0, *xl, color='b', lw=3, label='$h_1\\equiv0$')
    ax.vlines(0, *yl, color='r', lw=3, label='$h_2\\equiv0$')
    ax.vlines((eps, -eps), *yl, color='k', lw=1.5,
              linestyles=['dashed'], label='$|h(x)|<\\epsilon$')
    ax.hlines((eps, -eps), *xl, color='k', lw=1.5, linestyles=['dashed'])
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.legend(loc='upper right')

    return ocomp

  def test_order():
    """Order test. Uses linear-affine ODEs to achieve closed-form error
      calculations.

      h(x)=[-1,-1,-1] <=> x<0,y<0,z>0
      h(x)=[1,1,1] <=> x>=0,y>=0,z<=0

      NOTES:
        - To avoid fixed points on the guards, ODEs are centered at
          +/-1, whichever is opposite to the current state's orthint.
        - System is diagonlizable: Z exponential, XY centers to
          simplify error calcs.
    """
    fn = 3

    def h(x):
      """vectorized guard functions: 3x... -> 3x..."""
      x = asarray(x)
      out = zeros((3,) + x.shape[1:], dtype=float)
      out[0] = x[0]
      out[1] = x[1]
      out[2] = -x[2]
      return out

    def Dh(x):
      """vectorized guard jacobian: 3x... -> 3x3x..."""
      x = asarray(x)
      out = zeros((3,) + x.shape, dtype=float)
      out[0, 0] = 1
      out[1, 1] = 1
      out[2, 2] = -1
      return out

    # Create ODE
    # - represent as hashmap for easy solving
    # - most ODEs have affine offset: [1, 1, -1]
    octants = [str(array(k)) for k in iterprod(*([[1, -1]] * 3))]
    Adict = {k: zeros((3, 3)) for k in octants}  # init empty
    Bdict = {k: array([1, 1, -1]) for k in octants}

    # Handle Repeating Cases
    for k in octants:
      # switch z from exp decay to exp growth
      if k[-3:] == '-1]':
        Adict[k][-1, -1] = -1  # +z
      else:
        Adict[k][-1, -1] = 3  # -z
      # handle xy A matrix
      kcomp = k.replace('[ ', '[').replace('  ', ' ')  # simplify strs
      if kcomp[:4] == '[1 1':  # +x,+y
        Adict[k][:-1, :-1] = [[10, 0], [0, 1]]
      elif kcomp[:6] == '[-1 -1':  # -x,-y
        Adict[k][:-1, :-1] = [[0, -1], [1, 0]]
      elif kcomp[:5] == '[-1 1':  # -x,+y
        Adict[k][:-1, :-1] = [[0, 1], [-1, 0]]
      elif kcomp[:5] == '[1 -1':  # +x,-y
        Adict[k][:-1, :-1] = [[0, -2], [0.5, 0]]
        Bdict[k][1] = 2
      else:
        raise RuntimeError(f'Invalid ODE key: {k}')

      # sanity check
      assert det(Adict[k]) != 0, f'A_{k} singular!'
    # endregion
    # IVP
    y0 = (-0.4, -0.15, 0.3)
    t0, tf = 0, 0.5
    aot = AffineOrderTester(Adict, Bdict, h, Dh)
    flow = aot.solve_closedform(y0, t0, tf)

    # Error
    r, eps = [], logspace(0.3, -4.4, 40)
    for ee in eps:
      t, y = aot.solve_ess(y0, t0, tf, eps=ee)
      err = flow(t) - y
      r.append(sqrt(mean(sum(err * err.conj(), 1))))
      print(f'Step is {ee:8.3e}  --> {r[-1]:8.3e}')

    # Plot
    # Range of results to fit regression line to
    rng = slice(9, -6)
    # Regress log-log values to obtain order
    p = polyfit(log(eps[rng]), log(r[rng]), 1)
    fig = figure(fn)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.loglog(1 / eps, exp(polyval(p, log(eps))), 'b-', linewidth=3)
    ax.loglog(1 / eps[rng], exp(polyval(p, log(eps[rng]))),
              'g', linewidth=10, color=(0, 1, 0))
    ax.loglog(1 / eps, r, 'o-r')
    ax.set_xlabel('1/$\\epsilon$')
    ax.set_ylabel('error RMS')
    ax.set_title('Order test: %.2g' % p[0])
    ax.grid()

  test_const()
  test_nonconst()
  test_order()

  try:
    get_ipython()
    ion()
  except NameError:
    show()
