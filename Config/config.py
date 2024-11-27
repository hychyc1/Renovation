import yaml

class Config:
    def __init__(self, config_dict):
        """
        Initializes the configuration object from a parsed dictionary.

        Args:
        - config_dict (dict): A dictionary parsed from a YAML file.
        """
        # Reward specifications
        reward_specs = config_dict.get('reward_specs', {})
        self.monetary_weight = reward_specs.get('monetary_weight', 1.0)
        self.transportation_weight = reward_specs.get('transportation_weight', 1.0)
        self.POI_weight = reward_specs.get('POI_weight', 1.0)

        # Agent specifications
        agent_specs = config_dict.get('agent_specs', {})
        self.batch_stage = agent_specs.get('batch_stage', False)

        # General parameters
        self.gamma = config_dict.get('gamma', 0.99)
        self.tau = config_dict.get('tau', 0.0)

        # Base information
        self.grid_size = config_dict.get('grid_size', [55, 54])
        self.plot_ratio = config_dict.get('plot_ratio', 0.8)
        self.POI_plot_ratio = config_dict.get('POI_plot_ratio', 0.75)
        self.speed_c = config_dict.get('speed_c', 0)
        self.speed_r = config_dict.get('speed_r', 0)
        self.max_yr = config_dict.get('max_yr', 10)
        self.inflation_rate = config_dict.get('inflation_rate', 0.015)
        self.POI_affect_range = config_dict.get('POI_affect_range', 5)
        self.space_per_person = config_dict.get('space_per_person', 30)
        self.occupation_rate = config_dict.get('occupation_rate', 0.9)
        self.FAR_values = config_dict.get('FAR_values', [])
        self.combinations = config_dict.get('combinations', [])
        self.monetary_compensation_ratio = config_dict.get('monetary_compensation_ratio', 0.5)

        # State encoder specifications
        state_encoder_specs = config_dict.get('state_encoder_specs', {})
        self.grid_attributes = state_encoder_specs.get('grid_attributes', [])
        self.grid_feature_name = state_encoder_specs.get('grid_feature_name', None)
        self.feature_dim = state_encoder_specs.get('feature_dim', None)

        # Policy specifications
        self.policy_specs = config_dict.get('policy_specs', {})

        # Value specifications
        value_specs = config_dict.get('value_specs', {})
        self.value_head_hidden_size = value_specs.get('value_head_hidden_size', [32, 32, 1])

        # Optimization parameters
        self.policy_lr = config_dict.get('policy_lr', 4.0e-4)
        self.value_lr = config_dict.get('value_lr', 4.0e-4)
        self.weight_decay = config_dict.get('weightdecay', 0.0)
        self.eps = config_dict.get('eps', 1.0e-5)

        # PPO coefficients
        self.value_pred_coef = config_dict.get('value_pred_coef', 0.5)
        self.entropy_coef = config_dict.get('entropy_coef', 0.01)
        self.clip_epsilon = config_dict.get('clip_epsilon', 0.2)

        # Training parameters
        self.max_num_iterations = config_dict.get('max_num_iterations', 100)
        self.num_episodes_per_iteration = config_dict.get('num_episodes_per_iteration', 1200)
        self.max_sequence_length = config_dict.get('max_sequence_length', 37)
        self.num_epochs = config_dict.get('num_epochs', 4)
        self.batch_size = config_dict.get('mini_size', 1024)
        self.save_model_interval = config_dict.get('save_model_interval', 10)

    def __repr__(self):
        """
        Returns a string representation of the configuration object for debugging.
        """
        config_items = vars(self).items()
        return '\n'.join([f"{key}: {value}" for key, value in config_items])

    @staticmethod
    def from_yaml(file_path):
        """
        Creates a Config object from a YAML file.

        Args:
        - file_path (str): Path to the YAML file.

        Returns:
        - Config: A configuration object initialized from the YAML file.
        """
        with open(file_path, 'r') as yaml_file:
            config_dict = yaml.safe_load(yaml_file)
        return Config(config_dict)
