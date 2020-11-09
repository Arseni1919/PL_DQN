from CONSTANTS import *
from dqn_lightning_module import DQNLightningModule
from dqn_net import DQN


def get_action(state, net):
    state = torch.tensor([state])
    q_values = net(state)
    _, action = torch.max(q_values, dim=1)
    return int(action.item())


def main():
    env = gym.make(ENV)
    state = env.reset()

    model = DQN(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load("example.ckpt"))

    game = 0
    total_reward = 0
    while game < 10:
        # action = env.action_space.sample()
        action = get_action(state, model)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        if done:
            state = env.reset()
            game += 1
            print(f'finished game {game} with a total reward: {total_reward}')
            total_reward = 0
        else:
            state = next_state
    env.close()


if __name__ == '__main__':
    main()
