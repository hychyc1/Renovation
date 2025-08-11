from utils.setup import setup_agent


if __name__ == '__main__':
    agent, cfg = setup_agent()
    # print("START RUNNING", flush=True)
    # """create agent"""
    # agent = PPOAgent(cfg=cfg, dtype=dtype, device=device)
    agent.train(cfg.max_num_iterations)

    # for iteration in range(cfg.max_num_iterations):
    #     train_one_iteration(agent, iteration)

    agent.logger.info('training done!')
