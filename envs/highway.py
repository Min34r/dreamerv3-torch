import gymnasium
import gym
import numpy as np

# Import highway_env to register the environments with gymnasium
import highway_env


class HighwayEnv:
    """
    Wrapper for Highway-Env autonomous driving environments.
    Supports highway-v0, intersection-v0, parking-v0, etc.
    
    Highway-env documentation: https://highway-env.farama.org/
    """
    metadata = {}

    def __init__(
        self,
        name,
        action_repeat=1,
        size=(64, 64),
        obs_type="image",
        action_type="discrete",
        seed=None,
    ):
        """
        Args:
            name: Environment name (e.g., 'highway', 'intersection', 'parking', 'merge', 'roundabout')
            action_repeat: Number of times to repeat each action
            size: Image observation size (width, height)
            obs_type: Type of observation - 'image', 'kinematics', or 'grayscale'
            action_type: Type of action space - 'discrete' or 'continuous'
            seed: Random seed
        """
        self._name = name
        self._action_repeat = action_repeat
        self._size = size
        self._obs_type = obs_type
        self._action_type = action_type
        self._use_rgb_render = False  # Will be set in _configure_env
        self.reward_range = [-np.inf, np.inf]
        
        # Map task name to full environment name
        env_name = self._get_env_name(name)
        
        # Create the environment with custom configuration
        self._env = gymnasium.make(env_name, render_mode="rgb_array")
        
        # Configure the environment
        self._configure_env()
        
        if seed is not None:
            self._env.reset(seed=seed)
        
        self._done = True

    def _get_env_name(self, name):
        """Map short task name to full environment name."""
        env_mapping = {
            "highway": "highway-v0",
            "intersection": "intersection-v0",
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

    @property
    def observation_space(self):
        """Return observation space compatible with DreamerV3."""
        if self._obs_type in ("image", "grayscale"):
            # Image observation
            channels = 1 if self._obs_type == "grayscale" else 3
            img_shape = self._size + (channels,)
            spaces = {
                "image": gym.spaces.Box(0, 255, img_shape, dtype=np.uint8),
                "is_first": gym.spaces.Box(0, 1, (), dtype=np.float32),
                "is_last": gym.spaces.Box(0, 1, (), dtype=np.float32),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.float32),
            }
        else:
            # Kinematics observation (vector-based)
            obs_shape = self._env.observation_space.shape
            spaces = {
                "kinematics": gym.spaces.Box(-np.inf, np.inf, obs_shape, dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), dtype=np.float32),
                "is_last": gym.spaces.Box(0, 1, (), dtype=np.float32),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.float32),
            }
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        """Return action space."""
        gymnasium_space = self._env.action_space
        
        if isinstance(gymnasium_space, gymnasium.spaces.Discrete):
            # Discrete action space
            space = gym.spaces.Discrete(gymnasium_space.n)
            space.discrete = True
        elif isinstance(gymnasium_space, gymnasium.spaces.Box):
            # Continuous action space
            space = gym.spaces.Box(
                low=gymnasium_space.low.astype(np.float32),
                high=gymnasium_space.high.astype(np.float32),
                dtype=np.float32
            )
        else:
            raise NotImplementedError(f"Action space {type(gymnasium_space)} not supported")
        
        return space

    def step(self, action):
        """Execute action and return observation, reward, done, info."""
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
        
        # Process observation
        processed_obs = self._process_obs(obs, is_first=False, is_last=done, is_terminal=terminated)
        
        # Add discount info
        if "discount" not in info:
            info["discount"] = np.array(0.0 if terminated else 1.0, dtype=np.float32)
        
        return processed_obs, total_reward, done, info

    def reset(self):
        """Reset the environment."""
        obs, info = self._env.reset()
        self._done = False
        return self._process_obs(obs, is_first=True, is_last=False, is_terminal=False)

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
            "vector": gym.spaces.Box(-1.0, 1.0, obs_shape, dtype=np.float32),
            "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
        }
        return gym.spaces.Dict(spaces)

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
