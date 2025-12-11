"""
Contextual Thompson Sampling (CTS) bandit for operator selection in ALNS.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CTSConfig:
    """Configuration for Contextual Thompson Sampling."""
    dim: int  # Context dimension
    lambda_reg: float = 1e-3  # Regularization parameter
    noise_var: float = 1.0  # Noise variance for reward


class ContextualTSBandit:
    """
    Contextual Thompson Sampling bandit for multi-armed bandit problem.
    
    Each arm maintains a Bayesian linear regression model:
        r = x^T * theta_a + epsilon
    where epsilon ~ N(0, noise_var)
    
    Posterior for arm a:
        theta_a ~ N(mu_a, Sigma_a)
    where:
        Sigma_a = inv(A_a)
        mu_a = Sigma_a @ b_a
    """
    
    def __init__(self, n_arms: int, config: CTSConfig):
        """
        Initialize CTS bandit.
        
        Args:
            n_arms: Number of arms (operator pairs)
            config: CTS configuration
        """
        self.n_arms = n_arms
        self.config = config
        self.dim = config.dim
        
        # Initialize prior for each arm
        # A_a = lambda_reg * I (regularization matrix)
        # b_a = 0 (initial bias)
        self.A = [np.eye(self.dim) * config.lambda_reg for _ in range(n_arms)]
        self.b = [np.zeros(self.dim) for _ in range(n_arms)]
        
        # Track statistics
        self.arm_counts = [0] * n_arms
        self.total_updates = 0
        
    def select_arm(self, x: np.ndarray, rng: np.random.Generator) -> int:
        """
        Select arm using Thompson Sampling.
        
        For each arm a:
            1. Sample theta_tilde_a ~ N(mu_a, Sigma_a)
            2. Compute r_hat_a = x^T * theta_tilde_a
            3. Select argmax_a r_hat_a
        
        Args:
            x: Context vector (dim,)
            rng: Random number generator
        
        Returns:
            Index of selected arm (0..n_arms-1)
        """
        if x.shape != (self.dim,):
            raise ValueError(f"Context dimension mismatch: expected ({self.dim},), got {x.shape}")
        
        best_arm = 0
        best_reward = float('-inf')
        
        for a in range(self.n_arms):
            # Compute posterior parameters
            try:
                Sigma_a = np.linalg.inv(self.A[a])
                mu_a = Sigma_a @ self.b[a]
            except np.linalg.LinAlgError:
                # If A_a is singular, use identity
                logger.warning(f"Arm {a}: A_a is singular, using identity")
                Sigma_a = np.eye(self.dim) / self.config.lambda_reg
                mu_a = np.zeros(self.dim)
            
            # Sample from posterior: theta_tilde ~ N(mu_a, Sigma_a)
            theta_tilde = rng.multivariate_normal(mu_a, Sigma_a)
            
            # Predict reward: r_hat = x^T * theta_tilde
            r_hat = np.dot(x, theta_tilde)
            
            if r_hat > best_reward:
                best_reward = r_hat
                best_arm = a
        
        return best_arm
    
    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        """
        Update posterior for given arm using online Bayesian linear regression.
        
        Update rules:
            A_a <- A_a + x * x^T
            b_a <- b_a + x * reward
        
        Args:
            arm: Arm index (0..n_arms-1)
            x: Context vector (dim,)
            reward: Observed reward (clipped to [-1, 1] for stability)
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm index: {arm} (must be 0..{self.n_arms-1})")
        
        if x.shape != (self.dim,):
            raise ValueError(f"Context dimension mismatch: expected ({self.dim},), got {x.shape}")
        
        # Clip reward for stability
        reward = np.clip(reward, -1.0, 1.0)
        
        # Update A_a: A_a <- A_a + x * x^T
        self.A[arm] = self.A[arm] + np.outer(x, x)
        
        # Update b_a: b_a <- b_a + x * reward
        self.b[arm] = self.b[arm] + x * reward
        
        # Update statistics
        self.arm_counts[arm] += 1
        self.total_updates += 1
    
    def get_posterior_summary(self) -> dict:
        """
        Get summary of posterior distributions for all arms.
        
        Returns:
            Dictionary with arm statistics
        """
        summary = {
            "total_updates": self.total_updates,
            "arm_counts": self.arm_counts.copy(),
            "arm_means": [],
            "arm_cov_diag": []
        }
        
        for a in range(self.n_arms):
            try:
                Sigma_a = np.linalg.inv(self.A[a])
                mu_a = Sigma_a @ self.b[a]
                summary["arm_means"].append(mu_a.tolist())
                summary["arm_cov_diag"].append(np.diag(Sigma_a).tolist())
            except np.linalg.LinAlgError:
                summary["arm_means"].append([0.0] * self.dim)
                summary["arm_cov_diag"].append([1.0] * self.dim)
        
        return summary

