import gymnasium
import argparse
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from reward_wrappers import Go1RewardWrapper

# Функция создания среды, должна быть ИДЕНТИЧНА той, что в parallel_train.py
def make_custom_env(render_mode="human"):
    """Создает кастомную среду Go1."""
    env = gymnasium.make(
    'Ant-v5',
        xml_file='aliengo_model/scene.xml',
        # Параметры должны соответствовать тем, что были при обучении
        forward_reward_weight=0.5,      # Резко снижаем, чтобы убрать спешку
        ctrl_cost_weight=0.05,          # Увеличиваем штраф за резкость для плавности
        contact_cost_weight=0.5,        # Радикально увеличиваем штраф за контакт
        healthy_reward=3.0,             # Максимально поощряем "жизнь"
    main_body=1,
        healthy_z_range=(0.34, 0.75),   # Еще немного поднимаем минимальную высоту
    include_cfrc_ext_in_observation=False,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
        frame_skip=5,
    max_episode_steps=10000,
        # Параметры для рендеринга
        render_mode=render_mode,
        width=1280,
        height=720,
)
    # Делаем ориентацию главным приоритетом
    env = Go1RewardWrapper(env, orientation_weight=10.0, joint_velocity_reward_weight=0.07)
    return env

def main():
    parser = argparse.ArgumentParser(description="Рендеринг обученной модели SAC для Go1")
    parser.add_argument("--run_id", type=str, required=True, help="ID запуска, например 'sac_full_benchmark_v1'")
    args = parser.parse_args()

    # Формируем пути к модели и файлу нормализации
    run_path = f"results/go1/SAC_parallel/{args.run_id}"
    model_path = os.path.join(run_path, "best_model.zip")
    stats_path = os.path.join(run_path, "vec_normalize.pkl")

    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели не найден по пути {model_path}")
        return
    if not os.path.exists(stats_path):
        print(f"Ошибка: Файл статистики нормализации не найден по пути {stats_path}")
        return

    print("Создание среды для рендеринга...")
    # Создаем базовую среду
    eval_env_raw = make_custom_env(render_mode="human")
    
    # Оборачиваем в DummyVecEnv, так как VecNormalize работает с векторными средами
    eval_env_dummy = DummyVecEnv([lambda: eval_env_raw])
    
    # ЗАГРУЖАЕМ СТАТИСТИКИ и оборачиваем среду в VecNormalize
    # training=False, norm_reward=False - ВАЖНО для оценки
    eval_env = VecNormalize.load(stats_path, eval_env_dummy)
    eval_env.training = False
    eval_env.norm_reward = False

    print(f"Загрузка модели из {model_path}...")
    # Загружаем модель SAC
    model = SAC.load(model_path, env=eval_env, device='cpu')

    print("Запуск рендеринга...")
    obs = eval_env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            # VecEnv не требует явного вызова render(), он происходит в step, если render_mode="human"
            
            if done:
                print("Эпизод завершен. Среда будет сброшена автоматически.")

    except KeyboardInterrupt:
        print("\nРендеринг остановлен пользователем.")
    finally:
        eval_env.close()
        print("Среда закрыта.")

if __name__ == "__main__":
    main()


