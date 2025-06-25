import gymnasium as gym
import numpy as np

class Go1RewardWrapper(gym.Wrapper):
    """
    Кастомный враппер для формирования функции вознаграждения для Go1.
    - Добавляет награду за ориентацию корпуса.
    - Добавляет награду за движение в суставах для стимуляции походки.
    """
    def __init__(self, env, orientation_weight=1.0, joint_velocity_reward_weight=0.05):
        super().__init__(env)
        self.orientation_weight = orientation_weight
        self.joint_velocity_reward_weight = joint_velocity_reward_weight
        # В Ant-v5, если exclude_current_positions_from_observation=False,
        # кватернион (w,x,y,z) находится в qpos[3:7].
        # Obs = concatenate((qpos, qvel)).
        # Поэтому кватернион будет в obs[3:7]
        self.quat_indices = slice(3, 7)
        # В Ant-v5, qvel состоит из 6 (скорость тела) + 12 (скорости суставов)
        self.joint_vel_indices = slice(-12, None)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # --- 1. Базовая награда из среды ---
        # Включает: forward_reward, ctrl_cost, contact_cost, healthy_reward
        base_reward = reward

        # --- 2. Награда за ориентацию ---
        # Извлекаем кватернион ориентации корпуса из наблюдений
        # Порядок в MuJoCo: [w, x, y, z]
        quat = obs[self.quat_indices]
        w, x, y, z = quat
        
        # Вычисляем z-компоненту вектора "вверх" для корпуса. 
        # Это эквивалентно dot(body_up, world_up) и более эффективно.
        # body_up_z = 1 - 2*x**2 - 2*y**2
        # Мы будем вознаграждать за положительное значение, чтобы робот не переворачивался.
        orientation_reward = 1 - 2 * (x**2 + y**2)

        # --- 3. Награда за движение суставов ---
        # Поощряем движение в суставах ног, чтобы избежать "вибрации"
        joint_velocities = self.env.unwrapped.data.qvel[self.joint_vel_indices]
        joint_movement_reward = np.sum(np.abs(joint_velocities))

        # --- Итоговая награда ---
        total_reward = (base_reward +
                        self.orientation_weight * orientation_reward +
                        self.joint_velocity_reward_weight * joint_movement_reward)

        # Сохраняем компоненты награды в info для отладки
        if "reward_components" not in info:
            info["reward_components"] = {}
        info["reward_components"]['reward_base'] = base_reward
        info["reward_components"]['reward_orientation'] = self.orientation_weight * orientation_reward
        info["reward_components"]['reward_joint_movement'] = self.joint_velocity_reward_weight * joint_movement_reward
        
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Враппер не меняет reset, но для полноты картины он должен быть
        return super().reset(**kwargs) 