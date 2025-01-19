# Thompson Sampling algorithm for contextual bandits

Implementation of the Thompson Sampling algorithm for contextual bandits from the paper 
["An Empirical Evaluation of Thompson Sampling"](https://papers.nips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html).

### Repository overview:
#### Algorithm - regualarised logistic regression with batch updates
`LogisticRegression.py` contains the implementation of Thompson sampling (Algorithm 1) in combination with
regularised logistic regression using batch updates (Algorithm 3) as described in Section 4 of the paper.

The implementation differs from the original paper in the Laplace approximation step. Specifically, 
the paper performs update of the covariance matrix as $q_i = q_i + \sum_{j=1}^n x^2_{ij} p_j(1-p_j)$; 
I observed that such an update leads to rapid variance reduction and insufficient exploration for 
larger batch size. To account for this, I added a normalization factor of $const/n$ to the update.

#### Environment - contextual bandit
`env.py` simulates an environment with two reward functions: linear and nonlinear.
- Linear reward: $r(x, a) = sigm(w^T [x, a] + \epsilon)$, where $\epsilon \sim N(0, \sigma^2)$
- Nonlinear reward: $r(x, a) = sigm(w^T [x, a(x)] + \epsilon)$, where $a(x) = a + x^T L(a)$, 
where $L(a)$ are some context-action interaction weights. 

#### Analysis 
`thompson_analysis.ipynb` contains an analysis of the Thompson sampling algorithm on the simulated environments.

#### Other 
`utils.py` contains helper functions for the analysis.

### Usage
```bash
git clone https://github.com/danilprov/thompson-sampling.git
conda env create -f environment.yml
conda activate thompson_alg
```

### Potential improvements
- Bootstrapped Thompson Sampling

### References
- original paper: [An Empirical Evaluation of Thompson Sampling](https://papers.nips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html)
- Logistic regression with BFGS solver: https://stats.stackexchange.com/questions/17436/logistic-regression-with-lbfgs-solver
