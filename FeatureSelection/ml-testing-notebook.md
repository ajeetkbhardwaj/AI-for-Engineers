# Testing ML Models in Production – A Complete Learning Notebook

> This notebook is designed to run top‑to‑bottom in Jupyter / Colab.  
> It covers theory, implementation, and practical best practices for production ML testing.

---

## 0. Setup: Imports and Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
import time
import random
import logging
import asyncio
import hashlib
import json
from collections import defaultdict, deque
from dataclasses import dataclass
import threading

# Scientific & statistical
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp, ttest_ind

# ML & data science
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Graphing and analysis
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("Setup complete. All dependencies imported.")
```

---

## 1. Introduction: Why Test ML Models in Production?

### 1.1 Challenges of Production ML

```markdown
Unlike traditional software, ML systems face unique challenges:

1. **Model Degradation**: Performance can degrade silently over time
   - Data drift: incoming data distribution differs from training
   - Concept drift: target variable distribution changes
   - Covariate shift: feature distributions shift

2. **Monitoring Complexity**: Hard to detect failures automatically
   - No ground truth available immediately
   - Prediction correctness may take days/weeks to verify
   - Multiple metrics to track simultaneously

3. **Risk of Rollout**: Deploying bad models impacts users
   - Cannot afford false positives/negatives in critical systems
   - Need to validate before full rollout
   - Must enable quick rollback if issues detected

4. **Resource Constraints**: Production is resource‑intensive
   - Computation cost grows with model complexity
   - Latency SLAs must be met
   - Auto‑scaling becomes crucial

5. **Dependencies & Cascades**: Complex model ecosystems
   - Downstream models depend on upstream predictions
   - Upstream failure cascades to entire system
   - Version management becomes critical
```

### 1.2 Testing Strategies Overview

```markdown
Main categories of production testing:

1. **Canary & Shadow Testing**
   - Gradually roll out to subset of traffic
   - Run alongside production without user impact
   - Observe performance before full rollout

2. **A/B Testing**
   - Random traffic split between models
   - Statistical comparison of performance
   - Determine winner via hypothesis testing

3. **Monitoring & Drift Detection**
   - Real‑time performance tracking
   - Detect data/concept drift early
   - Automated alerts and rollback

4. **Validation Suites**
   - Comprehensive pre‑deployment tests
   - Data quality checks
   - Model behavior validation

5. **Resource & Dependency Monitoring**
   - Track compute/memory/latency
   - Detect cascade failures
   - Auto‑scaling and graceful degradation
```

---

## 2. A/B Testing: Foundational Concepts

### 2.1 Theory

```markdown
A/B Testing = Statistical methodology for comparing two treatments (models)

Core idea:
- Randomly partition traffic into control (legacy) and treatment (candidate)
- Each user sees one model consistently (deterministic assignment)
- Compare aggregate metrics between groups
- Use hypothesis testing to determine winner

Key assumptions:
- Independence: User experiences are independent
- Consistency: Same model version throughout test
- Sufficiency: Enough data for statistical power
```

### 2.2 Simple A/B Test Router Implementation

```python
import random
from typing import Dict, Any

class SimpleABTestRouter:
    """
    Basic A/B test router that randomly splits traffic.
    """
    def __init__(self, legacy_model, candidate_model, 
                 candidate_traffic_fraction=0.1):
        self.legacy_model = legacy_model
        self.candidate_model = candidate_model
        self.traffic_fraction = candidate_traffic_fraction
        
        # Track metrics for both groups
        self.metrics = {
            'legacy': {
                'count': 0,
                'predictions': [],
                'latencies': []
            },
            'candidate': {
                'count': 0,
                'predictions': [],
                'latencies': []
            }
        }
    
    def route_request(self, input_data: Dict[str, Any], 
                     user_id: str) -> Dict[str, Any]:
        """
        Route request to either legacy or candidate model.
        Randomly assign based on traffic fraction.
        """
        start_time = time.time()
        
        # Random assignment
        if random.random() < self.traffic_fraction:
            prediction = self.candidate_model.predict(input_data)
            model_used = 'candidate'
        else:
            prediction = self.legacy_model.predict(input_data)
            model_used = 'legacy'
        
        latency = time.time() - start_time
        
        # Log metrics
        self.metrics[model_used]['count'] += 1
        self.metrics[model_used]['predictions'].append(prediction)
        self.metrics[model_used]['latencies'].append(latency)
        
        return {
            'model': model_used,
            'prediction': prediction,
            'latency': latency
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Return aggregated metrics for both groups."""
        summary = {}
        for group in ['legacy', 'candidate']:
            metrics = self.metrics[group]
            summary[group] = {
                'count': metrics['count'],
                'avg_prediction': np.mean(metrics['predictions']) if metrics['predictions'] else 0,
                'avg_latency_ms': np.mean(metrics['latencies']) * 1000 if metrics['latencies'] else 0,
                'std_latency_ms': np.std(metrics['latencies']) * 1000 if metrics['latencies'] else 0
            }
        return summary

# Demo
print("\n=== A/B Test Router Demo ===")
legacy = LogisticRegression(random_state=42)
candidate = LogisticRegression(random_state=42, C=0.5)

X, y = make_classification(n_samples=1000, n_features=20, 
                          n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

legacy.fit(X_train, y_train)
candidate.fit(X_train, y_train)

# Create router and simulate requests
router = SimpleABTestRouter(legacy, candidate, candidate_traffic_fraction=0.3)

for i in range(200):
    result = router.route_request(X_test[i], user_id=f"user_{i}")

print(router.get_summary_stats())
```

### 2.3 Statistical Significance Testing

```python
from scipy import stats

def ab_test_significance(group_a_metrics: List[float], 
                        group_b_metrics: List[float],
                        alpha: float = 0.05) -> Dict[str, Any]:
    """
    Perform two‑sample t‑test to determine if groups differ significantly.
    
    Returns:
        - p_value: probability of observing this difference by chance
        - significant: whether difference is statistically significant
        - effect_size: Cohen's d (magnitude of difference)
        - confidence_interval: CI for mean difference
    """
    # t‑test
    t_stat, p_value = stats.ttest_ind(group_a_metrics, group_b_metrics)
    
    # Effect size (Cohen's d)
    mean_a, mean_b = np.mean(group_a_metrics), np.mean(group_b_metrics)
    std_a, std_b = np.std(group_a_metrics), np.std(group_b_metrics)
    n_a, n_b = len(group_a_metrics), len(group_b_metrics)
    
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
    
    # Confidence intervals
    sem_a = stats.sem(group_a_metrics)
    sem_b = stats.sem(group_b_metrics)
    ci_a = stats.t.interval(1 - alpha, n_a - 1, 
                           loc=mean_a, scale=sem_a)
    ci_b = stats.t.interval(1 - alpha, n_b - 1, 
                           loc=mean_b, scale=sem_b)
    
    return {
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size_cohens_d': cohens_d,
        'mean_a': mean_a,
        'mean_b': mean_b,
        'ci_a': ci_a,
        'ci_b': ci_b,
        'interpretation': 'Significant difference' if p_value < alpha 
                         else 'No significant difference'
    }

# Demo: Simulate A/B test results
legacy_accuracy = np.random.normal(0.85, 0.02, 500)
candidate_accuracy = np.random.normal(0.87, 0.02, 500)

result = ab_test_significance(legacy_accuracy, candidate_accuracy)
print("\n=== A/B Test Significance ===")
for key, value in result.items():
    print(f"{key}: {value}")
```

---

## 3. Canary Testing: Gradual Rollout

### 3.1 Concept

```markdown
Canary Testing = Roll out model to small fraction of users, monitor, 
                 then expand if healthy.

Advantages over full rollout:
- Risk is limited to canary group
- Real production traffic patterns (not synthetic)
- Easy to revert if issues detected
- Natural load testing

Advantages over A/B test:
- Directional: we intend to replace, not just compare
- Time‑based: rollout percentage increases over time
- Deterministic: same user always sees same model (good for support)
```

### 3.2 Implementation

```python
class CanaryRouter:
    """
    Routes requests to candidate model for canary users,
    legacy for rest. Deterministically assigns users based on hash.
    """
    def __init__(self, legacy_model, candidate_model, 
                 canary_traffic_fraction=0.05):
        self.legacy_model = legacy_model
        self.candidate_model = candidate_model
        self.canary_fraction = canary_traffic_fraction
        
        # Health tracking
        self.canary_metrics = defaultdict(list)
        self.legacy_metrics = defaultdict(list)
        self.is_healthy = True
    
    def is_canary_user(self, user_id: str) -> bool:
        """
        Deterministically assign user to canary or not.
        Same user always gets same assignment.
        """
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return (hash_value % 100) < (self.canary_fraction * 100)
    
    def route_request(self, user_id: str, input_data: Dict[str, Any]
                     ) -> Dict[str, Any]:
        """Route and track metrics."""
        start_time = time.time()
        
        try:
            if self.is_canary_user(user_id):
                prediction = self.candidate_model.predict(input_data)
                model_used = 'candidate'
                metrics_dict = self.canary_metrics
            else:
                prediction = self.legacy_model.predict(input_data)
                model_used = 'legacy'
                metrics_dict = self.legacy_metrics
            
            latency = time.time() - start_time
            metrics_dict['latency'].append(latency)
            metrics_dict['prediction'].append(prediction)
            
            return {
                'model': model_used,
                'prediction': prediction,
                'latency': latency,
                'user_id': user_id
            }
        
        except Exception as e:
            logger.error(f"Error routing request: {str(e)}")
            self.canary_metrics['error'].append(1)
            raise
    
    def check_canary_health(self, 
                           latency_threshold_ms=100,
                           error_threshold=0.01) -> bool:
        """
        Check if canary metrics exceed thresholds.
        Return False if unhealthy (should rollback).
        """
        if not self.canary_metrics['latency']:
            return True  # Not enough data yet
        
        avg_latency = np.mean(self.canary_metrics['latency'])
        error_rate = len(self.canary_metrics['error']) / \
                    (len(self.canary_metrics['latency']) + 
                     len(self.canary_metrics['error']) + 1)
        
        is_healthy = (avg_latency * 1000 < latency_threshold_ms and
                     error_rate < error_threshold)
        
        logger.info(f"Canary health: latency={avg_latency*1000:.2f}ms, "
                   f"error_rate={error_rate:.4f}, healthy={is_healthy}")
        
        self.is_healthy = is_healthy
        return is_healthy
    
    def expand_canary(self, new_fraction: float):
        """Increase canary traffic percentage."""
        old_fraction = self.canary_fraction
        self.canary_fraction = min(new_fraction, 1.0)
        logger.info(f"Expanded canary from {old_fraction*100:.1f}% to "
                   f"{self.canary_fraction*100:.1f}%")

# Demo
print("\n=== Canary Deployment Demo ===")
canary_router = CanaryRouter(legacy, candidate, canary_traffic_fraction=0.1)

for i in range(100):
    user_id = f"user_{i}"
    result = canary_router.route_request(user_id, X_test[i])

canary_router.check_canary_health()
```

### 3.3 Automatic Rollback with Monitoring

```python
class AdvancedCanaryRouter:
    """
    Canary router with automatic rollback based on monitored metrics.
    """
    def __init__(self, legacy_model, candidate_model,
                 error_threshold=0.02,
                 latency_threshold_ms=150,
                 window_size=500):
        self.legacy_model = legacy_model
        self.candidate_model = candidate_model
        self.error_threshold = error_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.window_size = window_size
        
        # Sliding window of recent metrics
        self.metrics_window = deque(maxlen=window_size)
        self.canary_fraction = 0.05
        self.is_healthy = True
        self.rollback_triggered = False
    
    def route_request(self, user_id: str, input_data: Dict[str, Any]
                     ) -> Dict[str, Any]:
        """Route request and monitor metrics."""
        start_time = time.time()
        is_canary = (int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100 
                    < self.canary_fraction * 100)
        
        try:
            if is_canary and not self.rollback_triggered:
                pred = self.candidate_model.predict(input_data)
                model = 'candidate'
            else:
                pred = self.legacy_model.predict(input_data)
                model = 'legacy'
            
            latency_ms = (time.time() - start_time) * 1000
            
            self.metrics_window.append({
                'model': model,
                'latency_ms': latency_ms,
                'error': False,
                'timestamp': time.time()
            })
            
            # Check health after each request (batched in practice)
            if len(self.metrics_window) % 50 == 0:
                self._check_and_handle_health()
            
            return {'model': model, 'prediction': pred, 'latency_ms': latency_ms}
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics_window.append({
                'model': 'candidate' if is_canary else 'legacy',
                'latency_ms': latency_ms,
                'error': True,
                'timestamp': time.time()
            })
            raise
    
    def _check_and_handle_health(self):
        """Check metrics and trigger rollback if needed."""
        if len(self.metrics_window) < 100:
            return
        
        # Calculate metrics on candidate traffic only
        candidate_requests = [m for m in self.metrics_window 
                            if m['model'] == 'candidate']
        
        if not candidate_requests:
            return
        
        error_rate = sum(1 for m in candidate_requests if m['error']) / len(candidate_requests)
        avg_latency = np.mean([m['latency_ms'] for m in candidate_requests])
        
        is_unhealthy = (error_rate > self.error_threshold or
                       avg_latency > self.latency_threshold_ms)
        
        if is_unhealthy and not self.rollback_triggered:
            logger.warning(f"ROLLBACK TRIGGERED: error_rate={error_rate:.4f}, "
                          f"avg_latency_ms={avg_latency:.2f}")
            self.rollback_triggered = True
            self.is_healthy = False

# Demo
print("\n=== Advanced Canary with Rollback ===")
advanced_canary = AdvancedCanaryRouter(legacy, candidate)

for i in range(300):
    user_id = f"user_{i}"
    try:
        result = advanced_canary.route_request(user_id, X_test[i % len(X_test)])
    except:
        pass

print(f"Rollback triggered: {advanced_canary.rollback_triggered}")
```

---

## 4. Shadow Testing: Production Validation Without Risk

### 4.1 Concept

```markdown
Shadow Testing = Route all production traffic to BOTH models,
                 but only return legacy response to user.

Advantages:
- Zero risk: users never see candidate predictions
- Real production traffic: includes all edge cases
- Detailed comparison: can log prediction differences
- No statistical burden: can compare on ANY metric

Disadvantages:
- Compute cost doubled (run both models)
- Latency impact (longest of two)
- Can't measure user impact (no feedback)
```

### 4.2 Implementation

```python
class ShadowTestingSystem:
    """
    Shadow testing system that routes all traffic to both models,
    returns legacy response, logs candidate metrics.
    """
    def __init__(self, legacy_model, candidate_model):
        self.legacy_model = legacy_model
        self.candidate_model = candidate_model
        self.shadow_logs = deque(maxlen=10000)
        self.stats = {
            'prediction_correlation': [],
            'latency_diff': [],
            'legacy_latency': [],
            'candidate_latency': [],
            'candidate_errors': 0,
            'total_requests': 0
        }
    
    def process_request(self, input_data: Dict[str, Any],
                       actual_label: Optional[float] = None) -> Dict[str, Any]:
        """
        Process request with both models. Return legacy response.
        Log candidate metrics in shadow.
        """
        self.stats['total_requests'] += 1
        
        # Get legacy prediction (served to user)
        t0 = time.time()
        legacy_pred = self.legacy_model.predict(input_data)
        legacy_latency = time.time() - t0
        self.stats['legacy_latency'].append(legacy_latency)
        
        # Get candidate prediction (shadowed)
        try:
            t0 = time.time()
            candidate_pred = self.candidate_model.predict(input_data)
            candidate_latency = time.time() - t0
            self.stats['candidate_latency'].append(candidate_latency)
            
            # Log for analysis
            log_entry = {
                'timestamp': time.time(),
                'legacy_pred': legacy_pred,
                'candidate_pred': candidate_pred,
                'legacy_latency': legacy_latency,
                'candidate_latency': candidate_latency,
                'latency_diff': candidate_latency - legacy_latency,
                'actual': actual_label
            }
            self.shadow_logs.append(log_entry)
            
            # Track prediction correlation
            if actual_label is not None:
                legacy_correct = (legacy_pred == actual_label)
                candidate_correct = (candidate_pred == actual_label)
                if legacy_correct == candidate_correct:
                    self.stats['prediction_correlation'].append(1.0)
                else:
                    self.stats['prediction_correlation'].append(0.0)
        
        except Exception as e:
            logger.error(f"Shadow prediction failed: {str(e)}")
            self.stats['candidate_errors'] += 1
        
        # Always return legacy prediction
        return {
            'prediction': legacy_pred,
            'model': 'legacy',
            'latency': legacy_latency
        }
    
    def get_shadow_report(self) -> Dict[str, Any]:
        """Generate report on shadow candidate model."""
        return {
            'total_requests': self.stats['total_requests'],
            'candidate_errors': self.stats['candidate_errors'],
            'error_rate': self.stats['candidate_errors'] / (self.stats['total_requests'] + 1),
            'avg_legacy_latency_ms': np.mean(self.stats['legacy_latency']) * 1000,
            'avg_candidate_latency_ms': np.mean(self.stats['candidate_latency']) * 1000,
            'latency_overhead_ms': np.mean(self.stats['latency_diff']) * 1000,
            'prediction_agreement': np.mean(self.stats['prediction_correlation']) if self.stats['prediction_correlation'] else 0
        }

# Demo
print("\n=== Shadow Testing Demo ===")
shadow_system = ShadowTestingSystem(legacy, candidate)

for i in range(100):
    result = shadow_system.process_request(X_test[i], y_test[i])

print("\nShadow Report:")
for key, value in shadow_system.get_shadow_report().items():
    print(f"  {key}: {value}")
```

---

## 5. Real‑Time Performance Monitoring

### 5.1 Monitoring Infrastructure

```python
class ModelPerformanceMonitor:
    """
    Tracks model performance metrics in real‑time.
    Detects drift and anomalies.
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.metrics_window = deque(maxlen=window_size)
        self.baseline_stats = None
        self.drift_detected = False
        self.drift_alerts = []
    
    def record_prediction(self, 
                         prediction: float,
                         actual: Optional[float] = None,
                         features: Optional[Dict[str, float]] = None,
                         latency: Optional[float] = None):
        """Record a prediction with ground truth and metadata."""
        record = {
            'timestamp': time.time(),
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual) if actual is not None else None,
            'latency': latency,
            'features': features
        }
        self.metrics_window.append(record)
        
        # Check for drift periodically
        if len(self.metrics_window) % 100 == 0:
            self._check_drift()
    
    def _check_drift(self):
        """Detect performance/data drift using statistical tests."""
        if len(self.metrics_window) < 200:
            return
        
        # Calculate current error distribution
        current_errors = [m['error'] for m in list(self.metrics_window)[-100:] 
                         if m['error'] is not None]
        
        if not current_errors:
            return
        
        # If no baseline, set it
        if self.baseline_stats is None:
            self.baseline_stats = {
                'mean_error': np.mean(current_errors),
                'std_error': np.std(current_errors)
            }
            return
        
        # Compare to baseline with KS test
        baseline_dist = np.random.normal(
            self.baseline_stats['mean_error'],
            self.baseline_stats['std_error'],
            len(current_errors)
        )
        
        _, p_value = ks_2samp(current_errors, baseline_dist)
        
        if p_value < 0.05:  # Significant drift
            self.drift_detected = True
            alert = {
                'timestamp': time.time(),
                'p_value': p_value,
                'current_mean_error': np.mean(current_errors),
                'baseline_mean_error': self.baseline_stats['mean_error']
            }
            self.drift_alerts.append(alert)
            logger.warning(f"DRIFT DETECTED: p_value={p_value:.6f}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        if not self.metrics_window:
            return {}
        
        errors = [m['error'] for m in self.metrics_window if m['error'] is not None]
        latencies = [m['latency'] for m in self.metrics_window if m['latency'] is not None]
        
        return {
            'total_predictions': len(self.metrics_window),
            'mean_error': np.mean(errors) if errors else None,
            'std_error': np.std(errors) if errors else None,
            'mean_latency_ms': np.mean(latencies) * 1000 if latencies else None,
            'drift_detected': self.drift_detected,
            'num_drift_alerts': len(self.drift_alerts)
        }

# Demo
print("\n=== Performance Monitoring Demo ===")
monitor = ModelPerformanceMonitor()

for i in range(200):
    pred = legacy.predict([X_test[i]])[0]
    actual = y_test[i]
    monitor.record_prediction(pred, actual, latency=np.random.uniform(0.01, 0.05))

print("\nMonitoring Summary:")
for key, value in monitor.get_metrics_summary().items():
    print(f"  {key}: {value}")
```

### 5.2 Feature Distribution Monitoring

```python
class FeatureDistributionMonitor:
    """
    Monitor feature distributions to detect data drift.
    """
    def __init__(self, feature_names: List[str],
                 reference_samples: np.ndarray,
                 drift_threshold=0.1):
        self.feature_names = feature_names
        self.drift_threshold = drift_threshold
        
        # Store reference distributions
        self.reference_dists = {}
        for i, feat_name in enumerate(feature_names):
            self.reference_dists[feat_name] = reference_samples[:, i]
        
        # Current sample buffer
        self.current_samples = {feat: [] for feat in feature_names}
        self.drift_scores = {}
    
    def add_sample(self, features: np.ndarray):
        """Add a new sample to current distribution buffer."""
        for i, feat_name in enumerate(self.feature_names):
            self.current_samples[feat_name].append(features[i])
        
        # Check drift every 500 samples
        if len(self.current_samples[self.feature_names[0]]) >= 500:
            self._calculate_drift_scores()
    
    def _calculate_drift_scores(self):
        """Calculate Wasserstein distance for each feature."""
        self.drift_scores = {}
        for feat_name in self.feature_names:
            current = np.array(self.current_samples[feat_name])
            reference = self.reference_dists[feat_name]
            
            # Wasserstein distance (earth mover's distance)
            distance = wasserstein_distance(current, reference)
            self.drift_scores[feat_name] = distance
            
            if distance > self.drift_threshold:
                logger.warning(f"Drift detected in {feat_name}: distance={distance:.4f}")
        
        # Reset buffer
        self.current_samples = {feat: [] for feat in self.feature_names}
    
    def get_drift_report(self) -> Dict[str, float]:
        """Get drift scores for all features."""
        return self.drift_scores

# Demo
print("\n=== Feature Distribution Monitoring ===")
feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
feat_monitor = FeatureDistributionMonitor(
    feature_names,
    X_train[:200],  # reference
    drift_threshold=0.05
)

# Add new samples
for i in range(100):
    feat_monitor.add_sample(X_test[i])

print("Drift scores:", feat_monitor.get_drift_report())
```

---

## 6. Model Validation Suite

### 6.1 Pre‑Deployment Validation

```python
@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    test_name: str
    actual_value: float
    threshold: float
    details: Optional[Dict] = None

class ModelValidationSuite:
    """
    Comprehensive validation for models before production deployment.
    """
    def __init__(self,
                 thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.validation_results = []
    
    def validate_model(self, model, 
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      X_ref: Optional[np.ndarray] = None) -> List[ValidationResult]:
        """
        Run all validation checks.
        """
        results = []
        
        # 1. Prediction bounds check
        results.extend(self._validate_bounds(model, X_val))
        
        # 2. Performance check
        results.extend(self._validate_performance(model, X_val, y_val))
        
        # 3. Latency check
        results.extend(self._validate_latency(model, X_val))
        
        # 4. Data distribution check (if reference provided)
        if X_ref is not None:
            results.extend(self._validate_distribution(X_ref, X_val))
        
        self.validation_results.extend(results)
        return results
    
    def _validate_bounds(self, model, X) -> List[ValidationResult]:
        """Check prediction bounds."""
        preds = model.predict(X)
        results = []
        
        results.append(ValidationResult(
            passed=np.all(preds >= self.thresholds.get('min_pred', -1)),
            test_name='min_prediction_bound',
            actual_value=np.min(preds),
            threshold=self.thresholds.get('min_pred', -1)
        ))
        
        results.append(ValidationResult(
            passed=np.all(preds <= self.thresholds.get('max_pred', 1)),
            test_name='max_prediction_bound',
            actual_value=np.max(preds),
            threshold=self.thresholds.get('max_pred', 1)
        ))
        
        return results
    
    def _validate_performance(self, model, X, y) -> List[ValidationResult]:
        """Check model accuracy."""
        y_pred = model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        return [ValidationResult(
            passed=accuracy >= self.thresholds.get('min_accuracy', 0.8),
            test_name='minimum_accuracy',
            actual_value=accuracy,
            threshold=self.thresholds.get('min_accuracy', 0.8)
        )]
    
    def _validate_latency(self, model, X) -> List[ValidationResult]:
        """Check inference latency."""
        latencies = []
        for sample in X[:100]:  # Sample 100
            t0 = time.time()
            model.predict([sample])
            latencies.append(time.time() - t0)
        
        avg_latency_ms = np.mean(latencies) * 1000
        p99_latency_ms = np.percentile(latencies, 99) * 1000
        
        return [
            ValidationResult(
                passed=avg_latency_ms <= self.thresholds.get('max_latency_ms', 100),
                test_name='average_latency',
                actual_value=avg_latency_ms,
                threshold=self.thresholds.get('max_latency_ms', 100)
            ),
            ValidationResult(
                passed=p99_latency_ms <= self.thresholds.get('max_p99_latency_ms', 200),
                test_name='p99_latency',
                actual_value=p99_latency_ms,
                threshold=self.thresholds.get('max_p99_latency_ms', 200)
            )
        ]
    
    def _validate_distribution(self, X_ref, X_val) -> List[ValidationResult]:
        """Check if validation features match reference distribution."""
        # Simple check: compare means
        ref_mean = np.mean(X_ref, axis=0)
        val_mean = np.mean(X_val, axis=0)
        mean_diff = np.mean(np.abs(ref_mean - val_mean))
        
        return [ValidationResult(
            passed=mean_diff <= self.thresholds.get('max_feature_drift', 0.5),
            test_name='feature_distribution_drift',
            actual_value=mean_diff,
            threshold=self.thresholds.get('max_feature_drift', 0.5)
        )]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        passed = sum(1 for r in self.validation_results if r.passed)
        total = len(self.validation_results)
        
        return {
            'passed_tests': passed,
            'total_tests': total,
            'pass_rate': passed / total if total > 0 else 0,
            'failed_tests': [
                {'name': r.test_name, 'actual': r.actual_value, 
                 'threshold': r.threshold}
                for r in self.validation_results if not r.passed
            ]
        }

# Demo
print("\n=== Model Validation Suite ===")
validator = ModelValidationSuite({
    'min_pred': 0,
    'max_pred': 1,
    'min_accuracy': 0.7,
    'max_latency_ms': 100,
    'max_p99_latency_ms': 200,
    'max_feature_drift': 0.5
})

results = validator.validate_model(candidate, X_test, y_test, X_train)
report = validator.generate_report()

print("\nValidation Report:")
for key, value in report.items():
    print(f"  {key}: {value}")
```

---

## 7. Model Versioning and Lineage

### 7.1 Version Control

```python
@dataclass
class ModelVersion:
    """Represents a specific model version."""
    version_id: str
    parent_version: Optional[str]
    creation_time: datetime
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    environment_config: Dict[str, str]

class ModelVersionControl:
    """
    Track model versions, lineage, and performance across versions.
    """
    def __init__(self):
        self.versions = {}
        self.current_version = None
        self.version_graph = nx.DiGraph()
    
    def register_version(self,
                        parameters: Dict[str, Any],
                        metrics: Dict[str, float],
                        env_config: Dict[str, str]) -> str:
        """
        Register a new model version.
        """
        # Generate version hash
        version_content = {
            'parameters': parameters,
            'timestamp': datetime.now().isoformat(),
            'parent': self.current_version
        }
        version_hash = hashlib.sha256(
            json.dumps(version_content).encode()
        ).hexdigest()[:12]
        
        # Create version object
        version = ModelVersion(
            version_id=version_hash,
            parent_version=self.current_version,
            creation_time=datetime.now(),
            parameters=parameters,
            performance_metrics=metrics,
            environment_config=env_config
        )
        
        self.versions[version_hash] = version
        self.version_graph.add_node(version_hash)
        
        if self.current_version:
            self.version_graph.add_edge(self.current_version, version_hash)
        
        logger.info(f"Registered version: {version_hash}")
        return version_hash
    
    def set_active_version(self, version_id: str):
        """Set a version as active for serving."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        self.current_version = version_id
        logger.info(f"Active version: {version_id}")
    
    def get_version_lineage(self, version_id: str) -> List[ModelVersion]:
        """Get lineage from root to specified version."""
        if version_id not in self.versions:
            return []
        
        # Traverse backwards to root
        lineage = []
        current = version_id
        while current is not None:
            lineage.insert(0, self.versions[current])
            current = self.versions[current].parent_version
        
        return lineage
    
    def compare_versions(self, v1_id: str, v2_id: str) -> Dict[str, Any]:
        """Compare two versions."""
        if v1_id not in self.versions or v2_id not in self.versions:
            return {}
        
        v1 = self.versions[v1_id]
        v2 = self.versions[v2_id]
        
        return {
            'v1_id': v1_id,
            'v2_id': v2_id,
            'v1_metrics': v1.performance_metrics,
            'v2_metrics': v2.performance_metrics,
            'metric_improvements': {
                k: v2.performance_metrics.get(k, 0) - v1.performance_metrics.get(k, 0)
                for k in v2.performance_metrics.keys()
            }
        }

# Demo
print("\n=== Model Version Control ===")
vc = ModelVersionControl()

v1 = vc.register_version(
    parameters={'C': 1.0},
    metrics={'accuracy': 0.85, 'f1': 0.82},
    env_config={'python': '3.9', 'sklearn': '1.0'}
)
vc.set_active_version(v1)

v2 = vc.register_version(
    parameters={'C': 0.5},
    metrics={'accuracy': 0.87, 'f1': 0.85},
    env_config={'python': '3.9', 'sklearn': '1.0'}
)

print(f"Version lineage for {v2}:")
for v in vc.get_version_lineage(v2):
    print(f"  {v.version_id}: {v.performance_metrics}")

print(f"\nComparison:")
comparison = vc.compare_versions(v1, v2)
for k, v in comparison.items():
    print(f"  {k}: {v}")
```

---

## 8. Practical Workflow: End‑to‑End Testing

### 8.1 Complete Deployment Pipeline

```python
class ProductionDeploymentPipeline:
    """
    Complete pipeline: validation → shadow → canary → full rollout.
    """
    def __init__(self, legacy_model, candidate_model):
        self.legacy = legacy_model
        self.candidate = candidate_model
        self.phase = 'validation'  # validation → shadow → canary → production
        self.validator = ModelValidationSuite({
            'min_accuracy': 0.8,
            'max_latency_ms': 100,
            'min_pred': 0,
            'max_pred': 1
        })
        self.shadow_system = ShadowTestingSystem(legacy_model, candidate_model)
        self.canary_router = AdvancedCanaryRouter(legacy_model, candidate_model)
        self.monitor = ModelPerformanceMonitor()
    
    def phase_1_validation(self, X_val, y_val, X_ref):
        """Phase 1: Pre‑deployment validation."""
        logger.info("=== PHASE 1: VALIDATION ===")
        results = self.validator.validate_model(
            self.candidate, X_val, y_val, X_ref
        )
        report = self.validator.generate_report()
        
        passed = report['pass_rate'] == 1.0
        logger.info(f"Validation: {'PASSED' if passed else 'FAILED'}")
        logger.info(f"Report: {report}")
        
        if passed:
            self.phase = 'shadow'
            return True
        return False
    
    def phase_2_shadow(self, X_shadow, y_shadow, n_samples=500):
        """Phase 2: Shadow testing."""
        logger.info("=== PHASE 2: SHADOW TESTING ===")
        
        for i in range(n_samples):
            self.shadow_system.process_request(X_shadow[i], y_shadow[i])
        
        report = self.shadow_system.get_shadow_report()
        logger.info(f"Shadow report: {report}")
        
        # Proceed if error rate < 1%
        if report['error_rate'] < 0.01:
            self.phase = 'canary'
            return True
        return False
    
    def phase_3_canary(self, X_canary, y_canary, n_samples=1000):
        """Phase 3: Canary deployment."""
        logger.info("=== PHASE 3: CANARY DEPLOYMENT ===")
        
        self.canary_router.canary_fraction = 0.05
        
        for i in range(n_samples):
            user_id = f"user_{i}"
            result = self.canary_router.route_request(user_id, X_canary[i])
            self.monitor.record_prediction(
                result['prediction'],
                y_canary[i],
                latency=result['latency']
            )
        
        is_healthy = self.canary_router.check_canary_health()
        logger.info(f"Canary health: {'HEALTHY' if is_healthy else 'UNHEALTHY'}")
        
        if is_healthy:
            self.phase = 'production'
            return True
        return False
    
    def phase_4_production(self):
        """Phase 4: Full production rollout."""
        logger.info("=== PHASE 4: FULL PRODUCTION ROLLOUT ===")
        self.phase = 'production'
        self.canary_router.canary_fraction = 1.0
        logger.info("All traffic routed to candidate model")
    
    def run_full_pipeline(self, X_val, y_val, X_ref, X_shadow, y_shadow, 
                         X_canary, y_canary):
        """Execute full deployment pipeline."""
        
        # Phase 1
        if not self.phase_1_validation(X_val, y_val, X_ref):
            logger.error("Deployment FAILED at validation phase")
            return False
        
        # Phase 2
        if not self.phase_2_shadow(X_shadow, y_shadow):
            logger.error("Deployment FAILED at shadow phase")
            return False
        
        # Phase 3
        if not self.phase_3_canary(X_canary, y_canary):
            logger.error("Deployment FAILED at canary phase")
            return False
        
        # Phase 4
        self.phase_4_production()
        logger.info("Deployment SUCCESSFUL - model in production")
        return True

# Demo
print("\n=== Production Deployment Pipeline ===")
pipeline = ProductionDeploymentPipeline(legacy, candidate)

# Split test data for different phases
X_val, X_rest = X_test[:100], X_test[100:]
y_val, y_rest = y_test[:100], y_test[100:]

X_shadow, X_canary = X_rest[:200], X_rest[200:400]
y_shadow, y_canary = y_rest[:200], y_rest[200:400]

success = pipeline.run_full_pipeline(
    X_val, y_val, X_train,
    X_shadow, y_shadow,
    X_canary, y_canary
)

print(f"\nDeployment result: {'SUCCESS' if success else 'FAILURE'}")
```

---

## 9. Key Takeaways and Best Practices

```markdown
### Testing Strategies Summary

1. **A/B Testing**
   - When: Comparing fundamentally different models
   - Risk: Moderate (users affected)
   - Cost: Single model + logging
   - Verdict: Statistical winner

2. **Shadow Testing**
   - When: Pre‑production validation of candidate
   - Risk: None (users don't see predictions)
   - Cost: Double compute (run both models)
   - Verdict: Behavior comparison, not user impact

3. **Canary Testing**
   - When: Rolling out to production
   - Risk: Low (small user fraction)
   - Cost: Monitoring/infrastructure
   - Verdict: Deterministic rollout

4. **Monitoring**
   - When: Continuously in production
   - Risk: Reactive (detects after problem)
   - Cost: Logging and storage
   - Verdict: Drift detection, performance tracking

### Best Practices

1. Always validate BEFORE production
2. Use canary for rollout, not for deciding winner
3. Monitor continuously, alert on anomalies
4. Keep version history, enable fast rollback
5. Test on real production traffic when possible
6. Combine multiple metrics (latency, error, accuracy)
7. Set clear success criteria beforehand
8. Document decisions and incidents
9. Automate rollback for catastrophic failures
10. Plan for graceful degradation
```

---

## 10. Summary

```markdown
Production ML testing is fundamentally different from model development:

- **Development**: Optimize for accuracy on test set
- **Production**: Monitor performance, detect degradation, enable rollback

Key systems needed:

1. Pre‑deployment: Comprehensive validation suite
2. Canary: Gradual rollout with health checks
3. Shadow: Zero‑risk comparison of candidates
4. Monitoring: Real‑time drift detection
5. Versioning: Track changes and enable rollback
6. Alerting: Automated incident response

The goal: Deploy models confidently, detect failures early, and revert quickly.
```

---

## References

- [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258) - Production ML Model Monitoring
- [https://arxiv.org/abs/2205.09865](https://arxiv.org/abs/2205.09865) - Model Testing in Production Environments
- [https://arxiv.org/abs/2103.12140](https://arxiv.org/abs/2103.12140) - MLOps: Production‑Centric AI
- [https://arxiv.org/abs/2209.14764](https://arxiv.org/abs/2209.14764) - Efficient Testing of Deep Learning Models
- [https://arxiv.org/abs/2203.15355](https://arxiv.org/abs/2203.15355) - Automated Model Testing and Validation
