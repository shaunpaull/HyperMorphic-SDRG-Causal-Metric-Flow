# HyperMorphic-SDRG-Causal-Metric-Flow
HyperMorphic SDRG Causal Metric Flow: a continual-memory prototype where representation geometry evolves over time, letting the system remember through adaptive metric state rather than fixed storage.

SDRG Causal Metric Flow

A continual-memory prototype where representation geometry evolves over time.

SDRG Causal Metric Flow is a HyperMorphic / State-Dependent Representation Geometry prototype in which the system remembers through its adaptive metric state, rather than through fixed storage alone.

The central idea is simple:

The memory is not only what is stored.
The memory is how the representation space bends over time.

⸻

One-Line Summary

SDRG Causal Metric Flow treats the representation metric as a dynamical causal object, allowing similarity itself to adapt to history, curvature, phase, and online regret.

⸻

Core Idea

Most prediction and retrieval systems use a fixed geometry.

For example, ordinary Euclidean distance uses:

d(x,y)^2 = ||x - y||²

SDRG Causal Metric Flow replaces this with a state-dependent metric:

d_t(z_i,z_j)^2 = (z_i - z_j)^T G_t (z_i - z_j)

where the metric G_t changes over time.

Instead of asking:

How close are two states in a fixed space?

The system asks:

How close are two states under the geometry implied by the current causal history?

⸻

Mathematical Primitive

Let z_t be a causal representation of the system state at time t.

Let there be a finite atlas of metric charts:

{G_c}_{c=1}^C

Each chart represents a different way of measuring similarity.

Examples:

* position-sensitive geometry
* velocity-sensitive geometry
* acceleration-sensitive geometry
* curvature-sensitive geometry
* phase/energy-sensitive geometry
* balanced geometry

The active metric is a mixture of these charts:

G_t = Σ_c q_c(H_t) G_c

where:

* G_t is the active metric at time t
* G_c is metric chart c
* q_c(H_t) is the chart weight
* H_t is the causal history/state through time t

So the metric itself becomes a state variable.

⸻

Causal Chart Routing

The model keeps an online regret/loss estimate for each metric chart:

R_c(t+1) = (1 - α) R_c(t) + α · loss_c(t)

where:

* R_c(t) is the running regret of chart c
* α is the memory update rate
* loss_c(t) is the recent prediction loss of chart c

The chart routing distribution is:

q_c(t) = softmax_c(-η R_c(t) + π_c(z_t))

where:

* η controls regret sensitivity
* π_c(z_t) is a causal prior over charts based on the current state
* q_c(t) determines how much chart c contributes to the metric

The active metric is then:

G_t = Σ_c q_c(t) G_c

This creates a causal metric flow.

⸻

State-Dependent Distance

Given two causal representations z_i and z_j, SDRG Causal Metric Flow measures distance as:

d_t(z_i,z_j)^2 = (z_i - z_j)^T G_t (z_i - z_j)

This means the same two points can be “near” or “far” depending on the current causal state.

That is the SDRG move:

similarity is not fixed
similarity is state-dependent

⸻

Continual Memory Interpretation

This is a continual-memory prototype because the system carries memory in the evolving metric.

The memory is encoded in:

R_c(t)
q_c(t)
G_t

The system remembers:

* which metric charts worked recently
* which geometry matched the current regime
* whether position, velocity, acceleration, phase, or curvature mattered
* how the dynamics changed over time
* which representation distances were useful for prediction

This is not replay memory.

It is not a database.

It is not a neural network weight update.

It is:

continual memory encoded as adaptive representation geometry.

⸻

Causal Features

The prototype builds a causal state vector from the trajectory history.

For each time step, it computes:

z_t = [
  position,
  velocity,
  acceleration,
  speed,
  acceleration magnitude,
  curvature-like cross term,
  velocity-acceleration alignment,
  sin(phase),
  cos(phase)
]

Example feature layout:

Feature	Meaning
x₁, x₂	position
v₁, v₂	velocity
a₁, a₂	acceleration
`	
`	
cross(v,a)	curvature-like turning signal
dot(v,a)	alignment / energy transfer
sin(θ), cos(θ)	phase direction

Only causal information through time t is used.

⸻

Metric Atlas

The prototype uses six metric charts:

Chart	Emphasis
position	current spatial location
velocity	direction and speed
acceleration	second-order change
curvature	turning / local bending
phase_energy	phase, speed, and energy-like state
balanced	all features weighted evenly

Each chart is a positive diagonal metric.

Example:

G_position = diag([high, high, low, low, ...])
G_velocity = diag([low, low, high, high, ...])
G_curvature = diag([... high curvature weights ...])

The active metric is the routed mixture:

G_t = q_1(t)G_1 + q_2(t)G_2 + ... + q_C(t)G_C

⸻

Prediction Method

The benchmark uses metric-based nearest-neighbor prediction.

For each time t, the model predicts the next state using past examples that are close under the active metric:

ŷ_t = weighted_average(y_i for nearest neighbors under d_t)

where:

y_i = x_{i+1}

The key difference is that SDRG changes the metric over time.

Fixed baselines use one static geometry for every time step.

⸻

Benchmark

The included demo generates a synthetic regime-switching dynamical system.

The hidden regimes overlap in position space, so a fixed Euclidean geometry can confuse states that look similar but have different causal futures.

The benchmark compares:

* fixed position-only Euclidean geometry
* fixed augmented geometry
* fixed velocity geometry
* fixed curvature geometry
* SDRG Causal Metric Flow

The goal is not to claim state-of-the-art machine learning.

The goal is to demonstrate a new primitive:

the representation metric can act as a continual causal memory channel

⸻

Outputs

The Colab script prints:

* per-seed MSE results
* aggregate MSE results
* improvement against the best fixed geometry
* formal primitive summary
* test suite results

It also saves plots:

sdrg_metric_flow_outputs/mse_bar.png
sdrg_metric_flow_outputs/router_weights.png
sdrg_metric_flow_outputs/metric_volume.png
sdrg_metric_flow_outputs/trajectory.png
sdrg_metric_flow_outputs/sdrg_metric_flow_results.json

⸻

Example Result Format

Example output:

AGGREGATE RESULTS
position_euclidean        MSE=...
fixed_augmented           MSE=...
fixed_velocity            MSE=...
fixed_curvature           MSE=...
SDRG_causal_metric_flow   MSE=...
SDRG improvement vs best fixed: ...%

The exact numbers may vary by seed and benchmark settings.

⸻

Quick Start

Clone the repository:

git clone https://github.com/YOUR_USERNAME/sdrg-causal-metric-flow.git
cd sdrg-causal-metric-flow

Run the demo:

python sdrg_causal_metric_flow.py

Or paste the full code into Google Colab and run the cell.

Dependencies:

numpy
matplotlib

⸻

Minimal Usage

from sdrg_causal_metric_flow import (
    make_sdrg_sequence,
    build_causal_features,
    standardize,
    SDRGCausalMetricFlow,
)
x, regimes = make_sdrg_sequence(seed=0, T=1400)
F = build_causal_features(x)
Y = x[1:]
Fz, mu, sigma = standardize(F)
model = SDRGCausalMetricFlow(feature_dim=Fz.shape[1])
t = 100
prediction, chart_predictions, weights = model.predict(Fz, Y, t)
print(prediction)
print(weights)

⸻

Formal Definition

Causal representation

Let:

z_t = φ(x_0, x_1, ..., x_t)

be a causal representation of the observed history.

Metric atlas

Let:

𝓖 = {G_1, G_2, ..., G_C}

be a finite set of positive metric charts.

Online regret

For each chart c:

R_c(t+1) = (1 - α)R_c(t) + αℓ_c(t)

where ℓ_c(t) is the prediction loss for chart c.

Causal routing

q_c(t) = exp(-ηR_c(t) + π_c(z_t)) / Σ_j exp(-ηR_j(t) + π_j(z_t))

Metric flow

G_t = Σ_c q_c(t)G_c

State-dependent distance

d_t(z_i,z_j)^2 = (z_i - z_j)^T G_t (z_i - z_j)

This is the primitive:

a representation geometry whose metric evolves as a causal memory state.

⸻

Why This Is HyperMorphic

The HyperMorphic idea is that the representation structure should not be treated as fixed background.

In SDRG Causal Metric Flow:

coordinates are fixed for the demo
but the metric is state-dependent

That means the same representation space can be interpreted through different geometries depending on history.

This is a first step toward:

state-dependent representation geometry

where future systems may adapt:

* metrics
* bases
* coordinates
* modular structure
* memory routing
* thresholds
* reconstruction rules

⸻

What This Is Not Claiming

This prototype does not claim:

* state-of-the-art ML performance
* proof of optimality
* biological memory equivalence
* replacement for neural sequence models
* formal continual-learning benchmark dominance
* production-ready memory architecture

The narrow claim is:

SDRG Causal Metric Flow demonstrates a continual-memory mechanism in which the memory channel is the evolving representation metric itself.

⸻

Suggested Paper Claim

A careful technical claim:

We introduce SDRG Causal Metric Flow, a prototype state-dependent representation geometry in which a finite atlas of metric charts is routed by causal history and online regret. The resulting metric flow acts as a continual-memory channel by changing how similarity is computed over time.

⸻

Suggested Paper Title

SDRG Causal Metric Flow: Continual Memory Through State-Dependent Representation Geometry

⸻

Suggested Citation

@misc{sdrg_causal_metric_flow,
  title  = {SDRG Causal Metric Flow: Continual Memory Through State-Dependent Representation Geometry},
  author = { Shaun Paul},
  year   = {2026},
  note   = {Research prototype}
}

⸻

Roadmap

Planned extensions:

* add real-world time-series datasets
* compare against online linear models
* compare against recurrent neural networks
* add continual-learning baselines
* add forgetting metrics
* add replay and no-replay variants
* extend from diagonal metrics to full SPD metrics
* add learned chart construction
* add chart birth/death
* add curvature regularization
* add metric stability bounds
* add visual phase-space atlas plots
* add paper-ready theorem section

⸻

Bottom Line

SDRG Causal Metric Flow turns similarity into memory:

history changes the metric
the metric changes neighborhood structure
neighborhood structure changes prediction
prediction regret changes future geometry

The central idea is:

the system remembers by changing the geometry through which it interprets the world.
