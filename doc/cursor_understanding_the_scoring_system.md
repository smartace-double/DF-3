# Understanding the scoring system
_Exported on 7/2/2025 at 21:29:01 GMT+9 from Cursor (1.1.5)_

---

**User**

scoring system

---

**Cursor**

Let me search through the codebase to understand the scoring/reward system.

Based on the code, I can explain how the scoring system works. The system evaluates miners based on their ability to predict Bitcoin prices accurately. Here's how it works:

1. **Point Prediction Scoring**:
- Miners make point predictions for Bitcoin's price
- The error is calculated as the absolute difference between predicted and actual price, normalized by the actual price
- Lower errors result in better ranks
- A perfect prediction (error = 0) gets the highest reward
- Missing predictions get infinite error (worst rank)

2. **Interval Prediction Scoring**:
- Miners also provide prediction intervals (a range of possible prices)
- Two factors determine the interval score:
  - Width Factor: Measures how precise the interval is
  - Inclusion Factor: Percentage of actual prices that fall within the predicted interval
- The final interval score is the product of these factors
- Missing interval predictions get a score of 0

3. **Combined Reward Calculation**:
```python
# Calculate ranks for both components
point_ranks = rank(np.array(point_errors))
interval_ranks = rank(-np.array(interval_scores))  # Flipped so 1 is best, 0 is worst

# Convert ranks to weights with decay
point_weights = get_average_weights_for_ties(point_ranks, decay=0.9)
interval_weights = get_average_weights_for_ties(interval_ranks, decay=0.9)

# Final reward is average of both components
rewards = (point_weights + interval_weights) / 2
```

4. **Key Features**:
- All rewards are normalized between 0 and 1
- Uses exponential decay (0.9) to weight rankings
- Handles ties fairly by averaging weights
- Miners with identical predictions get identical rewards
- Better predictions consistently get higher rewards

5. **Moving Average**:
- The system maintains a moving average of rewards for each miner
- New rewards are incorporated with configurable alpha:
```python
moving_average = (1 - alpha) * old_score + alpha * new_reward
```

6. **Testing**:
The test suite verifies that:
- Perfect predictions (exact match) get highest rewards
- Poor predictions ($5000 off) get medium rewards
- Terrible predictions (50% off) get lowest rewards
- Identical predictions receive identical rewards
- All rewards are between 0 and 1

This scoring system incentivizes both accuracy (through point predictions) and uncertainty quantification (through interval predictions), while maintaining fairness through proper handling of ties and consistent reward scaling.

```python
# Calculate ranks for both components
point_ranks = rank(np.array(point_errors))
interval_ranks = rank(-np.array(interval_scores))  # Flipped so 1 is best, 0 is worst

# Convert ranks to weights with decay
point_weights = get_average_weights_for_ties(point_ranks, decay=0.9)
interval_weights = get_average_weights_for_ties(interval_ranks, decay=0.9)

# Final reward is average of both components
rewards = (point_weights + interval_weights) / 2
```

```python
moving_average = (1 - alpha) * old_score + alpha * new_reward
```

---

**User**

detail about input and output for a miner? miner's role

---

**Cursor**

Let me search for more details about the miner's role and its input/output interface.

Let me explain the miner's role and its input/output interface in detail:

1. **Miner's Role**:
- Miners are nodes in the network that provide Bitcoin price predictions
- They receive prediction requests from validators and must respond quickly with low latency
- Each miner implements a prediction strategy through a `forward` function
- Miners can be customized with different prediction strategies

2. **Input Interface**:
The miner receives a `Challenge` synapse object with:
```python
class Challenge(bt.Synapse):
    timestamp: str  # The timestamp to predict from
    prediction: Optional[float] = None  # To be filled by miner
    interval: Optional[List[float]] = None  # To be filled by miner [min, max]
```

3. **Output Requirements**:
Miners must provide two types of predictions:
- Point Prediction (`prediction`): A single price value
- Interval Prediction (`interval`): A range [min, max] for the price
- Both predictions are made for a future time period (default is 1 hour ahead)

4. **Base Miner Implementation**:
The base miner provides a naive implementation:
```python
def forward(synapse: Challenge, cm: CMData) -> Challenge:
    # Get point estimate (current price)
    point_estimate = get_point_estimate(cm, synapse.timestamp)
    
    # Get prediction interval
    prediction_interval = get_prediction_interval(
        cm, synapse.timestamp, point_estimate
    )
    
    # Fill the response
    synapse.prediction = point_estimate
    synapse.interval = prediction_interval
    return synapse
```

5. **Custom Miner Development**:
You can create custom miners by:
- Writing a custom forward function in `precog/miners/your_file.py`
- Implementing your own prediction strategy
- Setting the `--forward_function` parameter to your implementation

6. **Security Features**:
Miners have built-in security measures:
- Blacklisting: Can reject requests from unregistered or non-validator entities
- Priority Queue: Processes requests based on caller's stake in the network
- Request Validation: Ensures requests have proper dendrite and hotkey information

7. **Configuration**:
Miners can be configured through environment variables:
```
NETWORK=finney
COLDKEY=your_miner_coldkey
MINER_HOTKEY=your_miner_hotkey
MINER_PORT=8092
TIMEOUT=16
FORWARD_FUNCTION=base_miner
```

8. **Performance Monitoring**:
- Miners track timing of predictions
- Log performance metrics for point estimates and intervals
- Monitor cache statistics for optimization
- Track their stake, trust, incentive, and emission metrics

9. **Deployment**:
```bash
# Register a new miner
make register ENV_FILE=.env.miner

# Run the miner
make miner ENV_FILE=.env.miner
```

The miner's success depends on:
1. Accuracy of point predictions
2. Quality of interval predictions (balance between width and inclusion)
3. Response speed and reliability
4. Maintaining good standing in the network (stake, trust)

```python
class Challenge(bt.Synapse):
    timestamp: str  # The timestamp to predict from
    prediction: Optional[float] = None  # To be filled by miner
    interval: Optional[List[float]] = None  # To be filled by miner [min, max]
```

```python
def forward(synapse: Challenge, cm: CMData) -> Challenge:
    # Get point estimate (current price)
    point_estimate = get_point_estimate(cm, synapse.timestamp)
    
    # Get prediction interval
    prediction_interval = get_prediction_interval(
        cm, synapse.timestamp, point_estimate
    )
    
    # Fill the response
    synapse.prediction = point_estimate
    synapse.interval = prediction_interval
    return synapse
```

```plaintext
NETWORK=finney
COLDKEY=your_miner_coldkey
MINER_HOTKEY=your_miner_hotkey
MINER_PORT=8092
TIMEOUT=16
FORWARD_FUNCTION=base_miner
```

```shellscript
# Register a new miner
make register ENV_FILE=.env.miner

# Run the miner
make miner ENV_FILE=.env.miner
```

---

**User**

what should predict?

---

**Cursor**

Let me search for what exactly needs to be predicted.

Let me explain what miners need to predict:

1. **Target Asset**: Bitcoin (BTC) price in USD
   - Uses CoinMetrics API to get reference rates
   - Predicts the BTC/USD price

2. **Prediction Types**:
   a. **Point Prediction**:
   - A single price value prediction
   - Represents the expected BTC price at a specific future time
   
   b. **Interval Prediction**:
   - A range [min, max] that should contain the future price
   - Should be a 90% confidence interval
   - Needs to balance width (precision) and inclusion (accuracy)

3. **Time Horizons**:
   - Predictions are made for 1 hour into the future (`PREDICTION_FUTURE_HOURS = 1`)
   - Validators query miners every 5 minutes (`PREDICTION_INTERVAL_MINUTES = 5`)
   - Predictions are evaluated over the past hour's actual prices

4. **Base Implementation Example**:
```python
def get_point_estimate(cm: CMData, timestamp: str) -> float:
    # Naive implementation: uses current price as prediction
    price_data = cm.get_CM_ReferenceRate(
        assets="BTC",
        end=timestamp,
        frequency="1s",
        limit_per_asset=1
    )
    return float(price_data["ReferenceRateUSD"].iloc[-1])

def get_prediction_interval(cm: CMData, timestamp: str, point_estimate: float) -> Tuple[float, float]:
    # Naive implementation: uses historical volatility
    # Gets last 24 hours of data
    historical_data = cm.get_CM_ReferenceRate(
        assets="BTC", 
        start=24_hours_ago,
        end=timestamp, 
        frequency="1s"
    )
    
    # Calculates 90% confidence interval
    std_dev = historical_data["ReferenceRateUSD"].diff().std()
    forecast_std = std_dev * (3600**0.5)  # Scale for 1-hour forecast
    
    lower = point_estimate - 1.64 * forecast_std
    upper = point_estimate + 1.64 * forecast_std
    
    return lower, upper
```

5. **Evaluation Criteria**:
   - Point predictions: Evaluated on absolute percentage error from actual price
   - Interval predictions: Scored on two factors:
     - Width Factor: How precise the interval is
     - Inclusion Factor: Percentage of actual prices that fall within the interval
   - Final reward combines both point and interval performance

6. **Key Constants**:
```python
PREDICTION_FUTURE_HOURS = 1      # Predict 1 hour ahead
PREDICTION_INTERVAL_MINUTES = 5   # Make predictions every 5 minutes
EVALUATION_WINDOW_HOURS = 6      # Keep 6 hours of history
```

Miners should aim to:
1. Provide accurate point predictions of BTC price 1 hour into the future
2. Give well-calibrated prediction intervals that balance precision and reliability
3. Respond quickly to prediction requests (low latency)
4. Maintain consistent availability and performance

```python
def get_point_estimate(cm: CMData, timestamp: str) -> float:
    # Naive implementation: uses current price as prediction
    price_data = cm.get_CM_ReferenceRate(
        assets="BTC",
        end=timestamp,
        frequency="1s",
        limit_per_asset=1
    )
    return float(price_data["ReferenceRateUSD"].iloc[-1])

def get_prediction_interval(cm: CMData, timestamp: str, point_estimate: float) -> Tuple[float, float]:
    # Naive implementation: uses historical volatility
    # Gets last 24 hours of data
    historical_data = cm.get_CM_ReferenceRate(
        assets="BTC", 
        start=24_hours_ago,
        end=timestamp, 
        frequency="1s"
    )
    
    # Calculates 90% confidence interval
    std_dev = historical_data["ReferenceRateUSD"].diff().std()
    forecast_std = std_dev * (3600**0.5)  # Scale for 1-hour forecast
    
    lower = point_estimate - 1.64 * forecast_std
    upper = point_estimate + 1.64 * forecast_std
    
    return lower, upper
```

```python
PREDICTION_FUTURE_HOURS = 1      # Predict 1 hour ahead
PREDICTION_INTERVAL_MINUTES = 5   # Make predictions every 5 minutes
EVALUATION_WINDOW_HOURS = 6      # Keep 6 hours of history
```

---

**User**

scoring system contain miner's response latency? and how frequently miner get challenge? detail of evaluation criteria 

---

**Cursor**

Let me search for details about the evaluation timing and criteria.

Let me explain the challenge frequency and evaluation criteria in detail:

1. **Challenge Frequency**:
- Validators query miners every 5 minutes (`PREDICTION_INTERVAL_MINUTES = 5`)
- Each query requires predicting BTC price 1 hour into the future
- The system checks for query timing using:
```python
if is_query_time(self.prediction_interval, self.timestamp) or query_lag >= 60 * self.prediction_interval:
    responses, self.timestamp = await self.query_miners()
```

2. **Response Timeout**:
- Miners have a configurable timeout (default 16 seconds in example config)
- Set via `--timeout $(TIMEOUT)` in miner configuration
- If a miner doesn't respond within timeout, they get penalized:
  - Point prediction error becomes infinite
  - Interval score becomes 0

3. **Detailed Evaluation Criteria**:

a) **Point Prediction Score**:
```python
# Error calculation
current_point_error = abs(prediction_value - actual_price) / actual_price

# Penalties
if prediction_time not in current_miner.predictions:
    point_errors.append(np.inf)  # No prediction penalty
if eval_time not in cm_data:
    point_errors.append(np.inf)  # No price data penalty
```

b) **Interval Prediction Score**:
```python
# Width Factor (precision)
width_factor = (effective_top - effective_bottom) / (pred_max - pred_min)

# Inclusion Factor (accuracy)
prices_in_bounds = sum(1 for price in hour_prices if pred_min <= price <= pred_max)
inclusion_factor = prices_in_bounds / len(hour_prices)

# Final interval score
interval_score_value = inclusion_factor * width_factor
```

4. **Ranking and Rewards**:
```python
# Convert errors/scores to ranks
point_ranks = rank(np.array(point_errors))
interval_ranks = rank(-np.array(interval_scores))  # Flipped so 1 is best

# Apply exponential decay to ranks
point_weights = get_average_weights_for_ties(point_ranks, decay=0.9)
interval_weights = get_average_weights_for_ties(interval_ranks, decay=0.9)

# Final reward combines both components
rewards = (point_weights + interval_weights) / 2
```

5. **Moving Average Updates**:
```python
new_score = (1 - alpha) * old_score + alpha * new_reward
```

6. **Additional Considerations**:
- Miners can prioritize requests based on validator stake
- Blacklisting of invalid or unauthorized requests
- Cache management for efficient responses
- Performance monitoring and logging

7. **Miner Response Requirements**:
- Must be fast (under timeout)
- Must provide both point and interval predictions
- Must maintain consistent availability
- Should optimize for both accuracy and precision
- Should handle multiple concurrent requests efficiently

The scoring system focuses on prediction accuracy and reliability, but response time is handled through the timeout mechanism rather than being directly incorporated into the score. Missing the timeout is effectively the same as making no prediction at all, resulting in the worst possible score for that round.

```python
if is_query_time(self.prediction_interval, self.timestamp) or query_lag >= 60 * self.prediction_interval:
    responses, self.timestamp = await self.query_miners()
```

```python
# Error calculation
current_point_error = abs(prediction_value - actual_price) / actual_price

# Penalties
if prediction_time not in current_miner.predictions:
    point_errors.append(np.inf)  # No prediction penalty
if eval_time not in cm_data:
    point_errors.append(np.inf)  # No price data penalty
```

```python
# Width Factor (precision)
width_factor = (effective_top - effective_bottom) / (pred_max - pred_min)

# Inclusion Factor (accuracy)
prices_in_bounds = sum(1 for price in hour_prices if pred_min <= price <= pred_max)
inclusion_factor = prices_in_bounds / len(hour_prices)

# Final interval score
interval_score_value = inclusion_factor * width_factor
```

```python
# Convert errors/scores to ranks
point_ranks = rank(np.array(point_errors))
interval_ranks = rank(-np.array(interval_scores))  # Flipped so 1 is best

# Apply exponential decay to ranks
point_weights = get_average_weights_for_ties(point_ranks, decay=0.9)
interval_weights = get_average_weights_for_ties(interval_ranks, decay=0.9)

# Final reward combines both components
rewards = (point_weights + interval_weights) / 2
```

```python
new_score = (1 - alpha) * old_score + alpha * new_reward
```

