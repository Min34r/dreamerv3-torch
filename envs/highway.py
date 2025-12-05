import gymnasium
import numpy as np

# Import highway_env to register the environments with gymnasium
import highway_env


# Default reward configuration for different environment types
DEFAULT_REWARD_CONFIGS = {
    "highway": {
        # Speed rewards - encourage maintaining high speed
        "high_speed_reward": 0.4,        # Reward weight for high speed
        "reward_speed_range": [25, 35],  # [min, max] m/s for full speed reward
        
        # Safety rewards
        "collision_reward": -1.0,        # Penalty for collision (terminal)
        "on_road_reward": 0.1,           # Reward for staying on road
        
        # Lane behavior - IMPROVED for better lane changing
        "right_lane_reward": 0.05,       # Small reward for rightmost lane (reduced to encourage overtaking)
        "lane_change_reward": 0.0,       # No penalty for lane changes
        "smart_lane_change_reward": 0.3, # Reward for changing lane to avoid slow vehicle
        "blocked_lane_penalty": 0.2,     # Penalty for staying behind slow vehicle
        "clear_lane_reward": 0.15,       # Reward for being in a clear lane
        
        # Defensive driving
        "safe_distance_reward": 0.1,     # Reward for keeping safe distance
        "min_safe_distance": 15.0,       # Minimum safe distance in meters
        "safe_distance_penalty": 0.3,    # Penalty for being too close
        
        # Look-ahead for lane decisions
        "look_ahead_distance": 50.0,     # Distance to look ahead for obstacles
        "slow_vehicle_threshold": 0.7,   # Consider vehicle slow if < 70% of ego speed
        
        # Normalize rewards to [0, 1] range
        "normalize_reward": True,
    },
    "intersection": {
        "high_speed_reward": 0.2,
        "reward_speed_range": [10, 15],
        "collision_reward": -1.0,
        "on_road_reward": 0.1,
        "arrived_reward": 1.0,           # Bonus for successfully crossing
        "safe_distance_reward": 0.2,
        "min_safe_distance": 10.0,
        "normalize_reward": True,
    },
    "merge": {
        "high_speed_reward": 0.3,
        "reward_speed_range": [20, 28],
        "collision_reward": -1.0,
        "on_road_reward": 0.1,
        "merging_speed_reward": 0.2,     # Reward for matching highway speed
        "safe_distance_reward": 0.15,
        "min_safe_distance": 12.0,
        "normalize_reward": True,
    },
    "roundabout": {
        "high_speed_reward": 0.2,
        "reward_speed_range": [8, 12],
        "collision_reward": -1.0,
        "on_road_reward": 0.15,
        "safe_distance_reward": 0.2,
        "min_safe_distance": 8.0,
        "normalize_reward": True,
    },
    "parking": {
        "collision_reward": -1.0,
        "on_road_reward": 0.0,
        "goal_reward": 1.0,              # Reward for reaching parking spot
        "goal_distance_reward": 0.5,     # Shaped reward based on distance to goal
        "heading_reward": 0.2,           # Reward for correct heading
        "normalize_reward": True,
    },
    "racetrack": {
        "high_speed_reward": 0.5,
        "reward_speed_range": [15, 25],
        "collision_reward": -1.0,
        "on_road_reward": 0.3,
        "lane_centering_reward": 0.2,    # Reward for staying centered
        "normalize_reward": True,
    },
}


class HighwayEnv(gymnasium.Env):
    """
    Wrapper for Highway-Env autonomous driving environments.
    Supports highway-v0, intersection-v0, parking-v0, etc.
    
    Highway-env documentation: https://highway-env.farama.org/
    
    Features advanced reward shaping for better learning:
    - Speed-based rewards
    - Safety distance rewards
    - Lane behavior rewards
    - Collision penalties
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        obs_type="image",
        action_type="discrete",
        seed=None,
        use_reward_shaping=True,
        reward_config=None,
        vehicles_count=50,
        vehicles_density=1.5,
    ):
        """
        Args:
            name: Environment name (e.g., 'highway', 'intersection', 'parking', 'merge', 'roundabout')
            action_repeat: Number of times to repeat each action
            size: Image observation size (width, height)
            obs_type: Type of observation - 'image', 'kinematics', or 'grayscale'
            action_type: Type of action space - 'discrete' or 'continuous'
            seed: Random seed
            use_reward_shaping: Whether to use advanced reward shaping (default True)
            reward_config: Custom reward configuration dict (overrides defaults)
            vehicles_count: Number of vehicles in the environment (default 50)
            vehicles_density: Density of vehicles on road (default 1.5)
        """
        super().__init__()
        self._name = name
        self._action_repeat = action_repeat
        self._size = size
        self._obs_type = obs_type
        self._action_type = action_type
        self._use_reward_shaping = use_reward_shaping
        self._vehicles_count = vehicles_count
        self._vehicles_density = vehicles_density
        self._use_rgb_render = False  # Will be set in _configure_env
        self.reward_range = [-np.inf, np.inf]
        
        # Setup reward configuration
        self._setup_reward_config(name, reward_config)
        
        # Track previous state for reward shaping
        self._prev_lane = None
        self._prev_speed = None
        
        # Map task name to full environment name
        env_name = self._get_env_name(name)
        
        # Create the environment with custom configuration
        self._env = gymnasium.make(env_name, render_mode="rgb_array")
        
        # Configure the environment
        self._configure_env()
        
        if seed is not None:
            self._env.reset(seed=seed)
        
        self._done = True

    def _setup_reward_config(self, name, custom_config=None):
        """Setup reward configuration based on environment type."""
        # Get default config for this environment type
        default_config = DEFAULT_REWARD_CONFIGS.get(name, DEFAULT_REWARD_CONFIGS["highway"])
        
        # Merge with custom config if provided
        self._reward_config = default_config.copy()
        if custom_config:
            self._reward_config.update(custom_config)

    def _get_env_name(self, name):
        """Map short task name to full environment name."""
        env_mapping = {
            "highway": "highway-v0",
            "intersection": "intersection-v1",  # Updated to v1
            "parking": "parking-v0",
            "merge": "merge-v0",
            "roundabout": "roundabout-v0",
            "racetrack": "racetrack-v0",
            "twowayhighway": "two-way-v0",
        }
        # If name is already a full env name, return it
        if "-v" in name:
            return name
        return env_mapping.get(name, f"{name}-v0")

    def _configure_env(self):
        """Configure the highway environment for DreamerV3 training."""
        # Base config - use larger display size for visibility
        # screen_width/height controls pygame window, observation_shape controls NN input
        display_width = 600
        display_height = 300
        
        config = {
            "action": {
                "type": "DiscreteMetaAction" if self._action_type == "discrete" else "ContinuousAction",
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 40,  # Number of steps per episode
            "screen_width": display_width,
            "screen_height": display_height,
            "scaling": 5.5,  # Zoom level for better visibility
        }
        
        # Environment-specific vehicle configuration
        # Different environments use different parameter names!
        env_name = self._name.lower()
        
        if env_name in ("highway", "highway-v0"):
            # Highway: uses vehicles_count and vehicles_density
            config["vehicles_count"] = self._vehicles_count
            config["vehicles_density"] = self._vehicles_density
            config["lanes_count"] = 4
            
        elif env_name in ("intersection", "intersection-v0", "intersection-v1"):
            # Intersection: uses initial_vehicle_count and spawn_probability
            config["initial_vehicle_count"] = min(self._vehicles_count, 20)  # Cap at 20 for intersection
            config["spawn_probability"] = min(self._vehicles_density / 2.0, 0.8)  # Convert density to probability
            
        elif env_name in ("merge", "merge-v0"):
            # Merge: limited configuration, vehicles come from ramp
            # No direct vehicle count control, but we can adjust duration
            config["duration"] = 50
            
        elif env_name in ("roundabout", "roundabout-v0"):
            # Roundabout: vehicles spawn dynamically, limited control
            config["duration"] = 15
            
        elif env_name in ("racetrack", "racetrack-v0"):
            # Racetrack: uses other_vehicles parameter
            # Note: Racetrack has specific action configuration requirements
            config["other_vehicles"] = min(self._vehicles_count, 10)  # Cap for racetrack
            config["controlled_vehicles"] = 1
            # Don't override action type for racetrack - use its default
            if self._action_type == "continuous":
                config["action"] = {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True
                }
            
        elif env_name in ("parking", "parking-v0"):
            # Parking: no other vehicles (goal-conditioned task)
            pass
        
        else:
            # Default: try highway-style config
            config["vehicles_count"] = self._vehicles_count
            config["vehicles_density"] = self._vehicles_density
        
        # Configure observation type
        # highway-env uses "GrayscaleObservation" for images (renders the env)
        if self._obs_type == "image":
            config["observation"] = {
                "type": "GrayscaleObservation",
                "observation_shape": (self._size[0], self._size[1]),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],
            }
            # We'll handle RGB conversion in _process_obs by rendering
            self._use_rgb_render = True
        elif self._obs_type == "grayscale":
            config["observation"] = {
                "type": "GrayscaleObservation",
                "observation_shape": (self._size[0], self._size[1]),
                "stack_size": 1,
                "weights": [0.2989, 0.5870, 0.1140],
            }
            self._use_rgb_render = False
        elif self._obs_type == "kinematics":
            config["observation"] = {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted",
            }
            self._use_rgb_render = False
        
        self._env.unwrapped.configure(config)
        # Reset to apply configuration
        self._env.reset()

    def _compute_shaped_reward(self, base_reward, info, terminated, truncated):
        """
        Compute shaped reward with multiple components for better learning.
        
        This combines:
        1. Speed reward - encourage optimal driving speed
        2. Collision penalty - strong negative reward for crashes
        3. Safe distance reward - maintain safe distance from other vehicles
        4. Lane behavior rewards - right lane preference, smooth lane changes
        5. On-road reward - staying on the road
        6. Progress reward - making forward progress
        7. Comfort reward - penalize harsh acceleration/steering (for continuous action)
        
        Returns:
            Shaped reward (float)
        """
        if not self._use_reward_shaping:
            return base_reward
        
        rc = self._reward_config
        reward = 0.0
        
        try:
            ego_vehicle = self._env.unwrapped.vehicle
            if ego_vehicle is None:
                return base_reward
        except (AttributeError, TypeError):
            return base_reward
        
        # 1. Collision penalty (highest priority)
        if terminated and info.get("crashed", ego_vehicle.crashed if hasattr(ego_vehicle, 'crashed') else False):
            reward += rc.get("collision_reward", -1.0)
            return reward  # Early return on collision
        
        # 2. Speed reward - normalized to target speed range
        try:
            speed = ego_vehicle.speed if hasattr(ego_vehicle, 'speed') else 0.0
            speed_range = rc.get("reward_speed_range", [20, 30])
            min_speed, max_speed = speed_range
            
            if speed < min_speed:
                # Below minimum speed - linearly scale reward
                speed_reward = (speed / min_speed) * rc.get("high_speed_reward", 0.4)
            elif speed <= max_speed:
                # In optimal range - full reward
                speed_reward = rc.get("high_speed_reward", 0.4)
            else:
                # Above maximum - small penalty for excessive speed
                overspeed_ratio = (speed - max_speed) / max_speed
                speed_reward = rc.get("high_speed_reward", 0.4) * max(0, 1 - overspeed_ratio * 0.5)
            
            reward += speed_reward
        except (AttributeError, TypeError):
            pass
        
        # 3. Safe distance reward
        try:
            min_safe_distance = rc.get("min_safe_distance", 15.0)
            road = self._env.unwrapped.road
            
            if road is not None and hasattr(road, 'vehicles'):
                min_distance = float('inf')
                for vehicle in road.vehicles:
                    if vehicle is not ego_vehicle:
                        distance = np.linalg.norm(
                            np.array(vehicle.position) - np.array(ego_vehicle.position)
                        )
                        if distance < min_distance:
                            min_distance = distance
                
                if min_distance < float('inf'):
                    if min_distance >= min_safe_distance:
                        # Safe distance maintained
                        reward += rc.get("safe_distance_reward", 0.1)
                    elif min_distance > 0:
                        # Too close - scale penalty based on how close
                        closeness_ratio = 1 - (min_distance / min_safe_distance)
                        reward -= closeness_ratio * rc.get("safe_distance_penalty", 0.3)
        except (AttributeError, TypeError):
            pass
        
        # 4. Right lane reward (prefer rightmost lane for highway)
        try:
            if rc.get("right_lane_reward", 0.0) != 0:
                lane_index = ego_vehicle.lane_index if hasattr(ego_vehicle, 'lane_index') else None
                if lane_index is not None:
                    # Highway usually has lanes indexed 0, 1, 2, ... 
                    # Higher index = more right
                    road = self._env.unwrapped.road
                    if road is not None and hasattr(road, 'network'):
                        # Normalize lane position
                        try:
                            lane_count = len(road.network.graph.get(lane_index[0], {}).get(lane_index[1], []))
                            if lane_count > 0:
                                right_ratio = lane_index[2] / (lane_count - 1) if lane_count > 1 else 1.0
                                reward += right_ratio * rc.get("right_lane_reward", 0.1)
                        except:
                            pass
        except (AttributeError, TypeError):
            pass
        
        # 5. Lane change penalty (smoother driving)
        try:
            if hasattr(self, '_last_lane_index'):
                current_lane = ego_vehicle.lane_index if hasattr(ego_vehicle, 'lane_index') else None
                if current_lane is not None and self._last_lane_index is not None:
                    if current_lane[2] != self._last_lane_index[2]:
                        reward += rc.get("lane_change_reward", -0.05)
            self._last_lane_index = ego_vehicle.lane_index if hasattr(ego_vehicle, 'lane_index') else None
        except (AttributeError, TypeError):
            self._last_lane_index = None
        
        # 5b. Smart lane change rewards - encourage lane changes when beneficial
        try:
            road = self._env.unwrapped.road
            ego_speed = ego_vehicle.speed if hasattr(ego_vehicle, 'speed') else 0
            ego_lane = ego_vehicle.lane_index if hasattr(ego_vehicle, 'lane_index') else None
            ego_pos = np.array(ego_vehicle.position) if hasattr(ego_vehicle, 'position') else None
            
            if road is not None and ego_lane is not None and ego_pos is not None:
                look_ahead = rc.get("look_ahead_distance", 50.0)
                slow_threshold = rc.get("slow_vehicle_threshold", 0.7)
                
                # Find vehicles ahead in current lane
                vehicle_ahead_in_lane = None
                min_dist_ahead = float('inf')
                
                for vehicle in road.vehicles:
                    if vehicle is ego_vehicle:
                        continue
                    
                    v_pos = np.array(vehicle.position)
                    v_lane = vehicle.lane_index if hasattr(vehicle, 'lane_index') else None
                    
                    # Check if in same lane and ahead
                    if v_lane is not None and ego_lane is not None:
                        if v_lane[0] == ego_lane[0] and v_lane[1] == ego_lane[1] and v_lane[2] == ego_lane[2]:
                            dist = v_pos[0] - ego_pos[0]  # Forward distance
                            if 0 < dist < look_ahead and dist < min_dist_ahead:
                                min_dist_ahead = dist
                                vehicle_ahead_in_lane = vehicle
                
                # Check if blocked by slow vehicle
                if vehicle_ahead_in_lane is not None:
                    v_speed = vehicle_ahead_in_lane.speed if hasattr(vehicle_ahead_in_lane, 'speed') else 0
                    
                    if v_speed < ego_speed * slow_threshold:
                        # There's a slow vehicle ahead - penalize staying in lane
                        self._blocked_by_slow = True
                        reward -= rc.get("blocked_lane_penalty", 0.2)
                        
                        # Check if we just changed lanes to escape
                        if hasattr(self, '_was_blocked') and self._was_blocked:
                            if hasattr(self, '_last_lane_index') and self._last_lane_index is not None:
                                if ego_lane[2] != self._last_lane_index[2]:
                                    # Reward for smart lane change!
                                    reward += rc.get("smart_lane_change_reward", 0.3)
                    else:
                        self._blocked_by_slow = False
                else:
                    # No vehicle ahead - reward for clear lane
                    self._blocked_by_slow = False
                    reward += rc.get("clear_lane_reward", 0.15)
                
                self._was_blocked = getattr(self, '_blocked_by_slow', False)
                
        except (AttributeError, TypeError):
            pass
        
        # 6. On-road reward
        try:
            on_road = ego_vehicle.on_road if hasattr(ego_vehicle, 'on_road') else True
            if on_road:
                reward += rc.get("on_road_reward", 0.1)
            else:
                reward += rc.get("off_road_penalty", -0.5)
        except (AttributeError, TypeError):
            pass
        
        # 7. Forward progress reward
        try:
            if hasattr(self, '_last_position'):
                current_pos = np.array(ego_vehicle.position) if hasattr(ego_vehicle, 'position') else None
                if current_pos is not None and self._last_position is not None:
                    # Reward for forward progress (usually x-direction)
                    forward_progress = current_pos[0] - self._last_position[0]
                    reward += forward_progress * rc.get("progress_reward_scale", 0.01)
            self._last_position = np.array(ego_vehicle.position) if hasattr(ego_vehicle, 'position') else None
        except (AttributeError, TypeError):
            self._last_position = None
        
        # 8. Heading alignment reward (staying aligned with road direction)
        try:
            if hasattr(ego_vehicle, 'heading') and hasattr(ego_vehicle, 'lane'):
                lane = ego_vehicle.lane
                if lane is not None:
                    lane_heading = lane.heading_at(lane.local_coordinates(ego_vehicle.position)[0])
                    heading_error = abs(ego_vehicle.heading - lane_heading)
                    heading_error = min(heading_error, 2 * np.pi - heading_error)  # Handle wrap-around
                    # Reward for good alignment (error near 0)
                    alignment_reward = (1 - heading_error / np.pi) * rc.get("heading_reward", 0.1)
                    reward += alignment_reward
        except (AttributeError, TypeError):
            pass
        
        # 9. Survival reward (small constant reward for staying alive)
        if not terminated:
            reward += rc.get("survival_reward", 0.01)
        
        # 10. Terminal rewards
        if terminated or truncated:
            if not info.get("crashed", False):
                # Successfully completed episode without crashing
                reward += rc.get("success_reward", 0.5)
        
        # Blend shaped reward with original reward
        blend_factor = rc.get("shaped_reward_weight", 0.8)
        final_reward = blend_factor * reward + (1 - blend_factor) * base_reward
        
        # Clip reward to reasonable range
        final_reward = np.clip(final_reward, -2.0, 2.0)
        
        return float(final_reward)

    @property
    def observation_space(self):
        """Return observation space compatible with DreamerV3."""
        if self._obs_type in ("image", "grayscale"):
            # Image observation
            channels = 1 if self._obs_type == "grayscale" else 3
            img_shape = self._size + (channels,)
            spaces = {
                "image": gymnasium.spaces.Box(0, 255, img_shape, dtype=np.uint8),
                "is_first": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
                "is_last": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
                "is_terminal": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
            }
        else:
            # Kinematics observation (vector-based)
            obs_shape = self._env.observation_space.shape
            spaces = {
                "kinematics": gymnasium.spaces.Box(-np.inf, np.inf, obs_shape, dtype=np.float32),
                "is_first": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
                "is_last": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
                "is_terminal": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
            }
        return gymnasium.spaces.Dict(spaces)

    @property
    def action_space(self):
        """Return action space."""
        gymnasium_space = self._env.action_space
        
        if isinstance(gymnasium_space, gymnasium.spaces.Discrete):
            # Discrete action space
            space = gymnasium.spaces.Discrete(gymnasium_space.n)
            space.discrete = True
        elif isinstance(gymnasium_space, gymnasium.spaces.Box):
            # Continuous action space
            space = gymnasium.spaces.Box(
                low=gymnasium_space.low.astype(np.float32),
                high=gymnasium_space.high.astype(np.float32),
                dtype=np.float32
            )
        else:
            raise NotImplementedError(f"Action space {type(gymnasium_space)} not supported")
        
        return space

    def step(self, action):
        """Execute action and return observation, reward, terminated, truncated, info."""
        # Handle one-hot encoded actions
        if hasattr(action, 'shape') and len(action.shape) >= 1 and action.shape[0] > 1:
            action = np.argmax(action)
        elif isinstance(action, np.ndarray):
            action = action.item() if action.size == 1 else action
        
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        done = terminated or truncated
        self._done = done
        
        # Apply reward shaping
        shaped_reward = self._compute_shaped_reward(total_reward, info, terminated, truncated)
        
        # Process observation
        processed_obs = self._process_obs(obs, is_first=False, is_last=done, is_terminal=terminated)
        
        # Add discount info
        if "discount" not in info:
            info["discount"] = np.array(0.0 if terminated else 1.0, dtype=np.float32)
        
        # Store original reward in info for debugging/logging
        info["original_reward"] = total_reward
        info["shaped_reward"] = shaped_reward
        
        return processed_obs, shaped_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            obs, info = self._env.reset(seed=seed, options=options)
        else:
            obs, info = self._env.reset(options=options)
        self._done = False
        
        # Reset reward shaping tracking variables
        self._last_lane_index = None
        self._last_position = None
        self._blocked_by_slow = False
        self._was_blocked = False
        
        return self._process_obs(obs, is_first=True, is_last=False, is_terminal=False), info

    def _process_obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        """Process observation to match DreamerV3 format."""
        if self._obs_type in ("image", "grayscale"):
            # For image observations, always use render to get RGB image
            if self._use_rgb_render or self._obs_type == "image":
                # Render the environment to get RGB image
                image = self._env.render()
            elif isinstance(obs, np.ndarray) and obs.ndim >= 2:
                image = obs
            else:
                # Fallback to render
                image = self._env.render()
            
            # Ensure correct shape and dtype
            if image.ndim == 2:
                image = image[:, :, None]  # Add channel dimension for grayscale
            
            # Resize if needed
            if image.shape[0] != self._size[1] or image.shape[1] != self._size[0]:
                image = self._resize_image(image)
            
            processed_obs = {
                "image": image.astype(np.uint8),
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
            }
        else:
            # Kinematics observation
            if isinstance(obs, np.ndarray):
                kinematics = obs.astype(np.float32)
            else:
                kinematics = np.array(obs, dtype=np.float32)
            
            processed_obs = {
                "kinematics": kinematics,
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
            }
        
        return processed_obs

    def _resize_image(self, image):
        """Resize image to target size."""
        try:
            import cv2
            return cv2.resize(image, self._size, interpolation=cv2.INTER_AREA)
        except ImportError:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            pil_img = pil_img.resize(self._size, PILImage.BILINEAR)
            return np.array(pil_img)

    def render(self, mode="rgb_array"):
        """Render the environment."""
        if mode != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.render()

    def close(self):
        """Close the environment."""
        return self._env.close()

    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            self._env.reset(seed=seed)


class HighwayEnvKinematics(HighwayEnv):
    """
    Highway environment wrapper using kinematics observations (vector-based).
    This is more efficient than image-based observations for some use cases.
    """
    
    def __init__(
        self,
        name,
        action_repeat=1,
        vehicles_count=5,
        features=None,
        action_type="discrete",
        seed=None,
    ):
        """
        Args:
            name: Environment name
            action_repeat: Number of times to repeat each action
            vehicles_count: Number of vehicles to observe
            features: List of features to observe per vehicle
            action_type: 'discrete' or 'continuous'
            seed: Random seed
        """
        # Initialize gymnasium.Env base class
        gymnasium.Env.__init__(self)
        
        self._vehicles_count = vehicles_count
        self._features = features or ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
        
        # Don't call parent __init__ - we'll set up differently
        self._name = name
        self._action_repeat = action_repeat
        self._obs_type = "kinematics"
        self._action_type = action_type
        self.reward_range = [-np.inf, np.inf]
        
        env_name = self._get_env_name(name)
        self._env = gymnasium.make(env_name, render_mode="rgb_array")
        self._configure_kinematics_env()
        
        if seed is not None:
            self._env.reset(seed=seed)
        
        self._done = True

    def _configure_kinematics_env(self):
        """Configure the environment for kinematics observations."""
        config = {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": self._vehicles_count,
                "features": self._features,
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted",
                "normalize": True,
            },
            "action": {
                "type": "DiscreteMetaAction" if self._action_type == "discrete" else "ContinuousAction",
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 40,
        }
        
        self._env.unwrapped.configure(config)
        self._env.reset()

    @property
    def observation_space(self):
        """Return observation space for kinematics."""
        obs_shape = (self._vehicles_count, len(self._features))
        spaces = {
            "vector": gymnasium.spaces.Box(-1.0, 1.0, obs_shape, dtype=np.float32),
            "is_first": gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_last": gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_terminal": gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
        }
        return gymnasium.spaces.Dict(spaces)

    def _process_obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        """Process kinematics observation."""
        if isinstance(obs, np.ndarray):
            vector = obs.astype(np.float32)
        else:
            vector = np.array(obs, dtype=np.float32)
        
        # Flatten if needed for MLP encoder
        if vector.ndim > 1:
            vector = vector.flatten()
        
        return {
            "vector": vector,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }
