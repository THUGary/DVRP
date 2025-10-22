# DVRP
接口定义:

1. 栅格地图表示,智能体位置和运动都在上面(每次运动一格)

2. generator,输入:环境尺寸与超参数(比如需求数量的限制等),输出:[(x1,y1,t1,c1),...,(xn,yn,tn,cn)],(x,y)是栅格上的坐标,t是需求的出现时间,c代表需求量

3. planner,输入:当前观测[(x1,y1,t1,c1),...,(xm,ym,tm,cm)]+智能体状态[(p1_x,p1_y,s1),...,(pk_x,pk_y,sk)];输出:每个智能体未来一定时间内的目标选择队列[T1,T2,...,Tk]
注意这里m不等于n,因为仅限于观测到当前时刻之前的需求,k是智能体的数量,Ti代表第i个智能体的目标队列,si代表智能体i的剩余空间

4. controller,输入:当前智能体的目标队列Ti;输出:当前智能体的运动ai,也就是选择周围那个邻接点位,通过(deltax,deltay)来表示

5. environment,状态:{智能体状态+depot位置+当前需求},动作是每一个智能体动作的拼接,每个step更新当前所有智能体状态,需求变化(新生成的,被解决的)

6. 使用pygame可视化状态

## Quick start

- 运行一次最小示例（规则基线）：

```bash
python train.py --seed 0
```

- 可视化（需要安装 pygame）：

```bash
pip install pygame
python train.py --seed 0 --render --fps 10
```

- 快速自检（短回合）：

```bash
python test.py
```

## 代码框架（API 草案）

- agent/generator
	- BaseDemandGenerator.reset(seed)
	- BaseDemandGenerator.sample(t) -> [(x,y,t,c), ...]
	- RuleBasedGenerator: 随机生成少量需求
	- NetDemandGenerator: 预留的学习型生成器

- agent/planner
	- BasePlanner.plan(observations, agent_states, depot, t, horizon) -> 每个智能体的目标队列
	- RuleBasedPlanner: 最近需求贪心
	- NetPlanner: 预留的学习型规划器

- agent/controller
	- BaseController.act(current_pos, target_queue) -> (dx, dy)
	- RuleBasedController: 朝当前目标移动一步

- environment
	- GridEnvironment.reset(seed) -> obs
	- GridEnvironment.step(actions) -> (obs, reward, done, info)
	- 观测 obs: {time, depot, agent_states[(x,y,s)], demands[(x,y,t,c)], width, height}

- 入口脚本
	- train.py: 规则基线串起来跑一个回合
	- test.py: 快速 smoke test
	- configs.py: 默认配置，可修改尺寸、智能体数量、生成器参数等

可视化与模型训练的部分预留在 models/ 与 pygame 中，当前骨架可直接运行、便于后续逐步填充。
