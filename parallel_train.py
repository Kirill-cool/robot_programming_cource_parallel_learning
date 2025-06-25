import gymnasium
import numpy as np
import os
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from my_eval import EvalCallback
from stable_baselines3.common.logger import configure
from reward_wrappers import Go1RewardWrapper


def make_custom_env(rank, env_id, seed=0):
    def _init():
        if env_id == "go1":
            env = gymnasium.make(
                'Ant-v5',
                xml_file='aliengo_model/scene.xml',
                forward_reward_weight=0.5,
                ctrl_cost_weight=0.05,
                contact_cost_weight=0.5,
                healthy_reward=3.0,
                main_body=1,
                healthy_z_range=(0.34, 0.75),
                include_cfrc_ext_in_observation=False,
                exclude_current_positions_from_observation=False,
                reset_noise_scale=0.1,
                frame_skip=5,
                max_episode_steps=1000,
            )
        else:
            env = gymnasium.make(env_id)
        
        env = Go1RewardWrapper(env, orientation_weight=10.0, joint_velocity_reward_weight=0.07)
        
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="SAC", type=str)
    parser.add_argument("--env_id", default="go1", type=str)
    parser.add_argument("--num_envs", type=int, default=16, help="Количество параллельных сред для загрузки CPU")
    parser.add_argument("--total_timesteps", type=int, default=5_000_000)
    parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Размер Replay Buffer (в RAM)")
    parser.add_argument("--batch_size", type=int, default=256, help="Размер батча для обучения (по бенчмарку rl-zoo)")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--learning_starts", type=int, default=10000, help="Количество шагов до старта обучения (по бенчмарку rl-zoo)")
    parser.add_argument("--run_id", type=str, required=True, help="Уникальный ID для НОВОГО запуска эксперимента")
    parser.add_argument("--load_from_run_id", type=str, default=None, help="ID старого запуска для продолжения обучения")
    parser.add_argument("--load_model_name", type=str, default="best_model.zip", help="Имя файла модели для загрузки (напр. 'best_model.zip')")
    args = parser.parse_args()

    run_path = f"results/{args.env_id}/{args.algo}_parallel/{args.run_id}"
    os.makedirs(run_path, exist_ok=True)

    env_fns = [make_custom_env(i, args.env_id, seed=i) for i in range(args.num_envs)]
    
    # Создаем базовую (не обернутую) векторную среду
    base_vec_env = SubprocVecEnv(env_fns)

    # Если продолжаем обучение, загружаем статистики, иначе создаем новые
    if args.load_from_run_id:
        load_path = f"results/{args.env_id}/{args.algo}_parallel/{args.load_from_run_id}"
        stats_path = os.path.join(load_path, "vec_normalize.pkl")
        if not os.path.exists(stats_path):
            raise ValueError(f"Файл статистики не найден: {stats_path}")
        print(f"Загрузка статистики VecNormalize из: {stats_path}")
        vec_env = VecNormalize.load(stats_path, base_vec_env)
        vec_env.training = True # Переключаем в режим обучения
    else:
        vec_env = VecNormalize(base_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)


    eval_env_raw = make_custom_env(0, args.env_id, seed=1234)()
    eval_vec_env = VecNormalize(SubprocVecEnv([lambda: eval_env_raw]), training=False, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=run_path,
        log_path=run_path,
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(save_freq=50000,
                                             save_path=run_path, name_prefix="rl_model")
    
    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # Если продолжаем, загружаем модель, иначе создаем новую
    if args.load_from_run_id:
        load_path = f"results/{args.env_id}/{args.algo}_parallel/{args.load_from_run_id}"
        model_path = os.path.join(load_path, args.load_model_name)
        if not os.path.exists(model_path):
            raise ValueError(f"Файл модели не найден: {model_path}")
        print(f"Загрузка модели из: {model_path}")
        model = SAC.load(model_path, env=vec_env, device='cuda')
        # Указываем модели, где сохранять новые логи Tensorboard
        model.tensorboard_log = run_path
    else:
        model = SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            tau=0.02,
            gamma=0.98,
            train_freq=(8, "step"),
            gradient_steps=8,
            learning_starts=args.learning_starts,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            device='cuda',
            tensorboard_log=run_path
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback_list,
        # Если продолжаем, не сбрасываем счетчик шагов
        reset_num_timesteps= (args.load_from_run_id is None)
    )

    model.save(f"{run_path}/final_model.zip")
    vec_env.save(f"{run_path}/vec_normalize.pkl")


if __name__ == "__main__":
    main() 