from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# Data Augmentation: reflection and axis swap
def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

def rescale_raw_problem(depot_xy:torch.Tensor,
                        node_xy:torch.Tensor,
                        node_demand:torch.Tensor,
                        Up_Bound:float=None,
                        Demand_scaler:float=None):
    '''
    Rescale the raw problem data: scale the coordinates to [0,1],\n
    and scale the demands.
    
    :param depot_xy: (batch, 2)
    :param node_xy: (batch, problem, 2)
    :param node_demand: (batch, problem)
    :param Up_Bound: the upper bound of the coordinates (both x and y).\n
        If the batched problem is sampled in a map of size Up_Bound x Up_Bound, this parameter
        can be provided.
    :param Demand_scaler: the capacity of vehicles used in cvrp.\n
        If not provided, it will be determined by the maximum demand value `d_max` and the problem_size `p` in the batch:\n
        `Demand_scaler=d_max/10*math.ceil(30+p/5) if p>20 else 2*d_max`

    
    :return: scaled_depot_xy, scaled_node_xy, scaled_node_demand, Up_Bound, Demand_scaler
    '''
    # depot_xy.shape: (batch, 2)
    # node_xy.shape: (batch, problem, 2)
    # node_demand.shape: (batch, problem)

    # Ensure inputs are tensors
    depot_xy = torch.as_tensor(depot_xy)
    node_xy = torch.as_tensor(node_xy)
    node_demand = torch.as_tensor(node_demand)

    problem_size=node_xy.shape[1]

    # Compute Up_Bound: maximum coordinate value across all depot and node x/y
    if Up_Bound is None:
        # depot_xy shape: (batch, 2) or (batch, 1, 2)
        # node_xy shape: (batch, problem, 2)
        # Compute scalar maximum across all coordinates (x and y) from depot and nodes
        try:
            max_depot_coord = float(depot_xy.max().item())
        except Exception:
            max_depot_coord = float(depot_xy.reshape(-1).max().item())
        max_node_coord = float(node_xy.reshape(-1).max().item())
        Up_Bound = float(max(max_depot_coord, max_node_coord))
        
    # Compute Demand: maximum demand value across the batch
    if Demand_scaler is None:
        Demand_scaler = float(node_demand.reshape(-1).max())
        Demand_scaler = Demand_scaler / 10.0 * math.ceil(30 + problem_size / 5) if problem_size > 20 else 2.0 * Demand_scaler

    print(f"Up_Bound: {Up_Bound}, Demand_scaler: {Demand_scaler}")

    if Up_Bound <= 0:
        raise ValueError("Up_Bound must be positive.")
    else:
        scaled_depot_xy = depot_xy / float(Up_Bound)
        scaled_node_xy = node_xy / float(Up_Bound)

    if Demand_scaler <= 0:
        raise ValueError("Demand_scaler must be positive.")
    else:
        scaled_node_demand = node_demand / float(Demand_scaler)

    return scaled_depot_xy, scaled_node_xy, scaled_node_demand, Up_Bound, Demand_scaler

def journey2loops(journey: torch.Tensor):
    '''
    Split journey sequences into loops using 0 as the depot separator.

    Input `journey` is expected to have shape (batch, journey_length).
    For each batch row, find indices where node == 0 and slice the sequence
    between those indices. The first element may be 0. Empty segments are ignored.

    Returns a Python list `loops` where `loops[b]` is a list of loops for batch b,
    and each loop is a list of node ids (zeros removed).
    '''
    if journey.dim() != 2:
        raise ValueError('journey must be a 2D tensor with shape (batch, journey_length)')

    batch_size, journey_length = journey.shape
    loops = []

    for b in range(batch_size):
        seq = journey[b].tolist()
        zero_idx = [i for i, v in enumerate(seq) if v == 0]
        b_loops = []

        if not zero_idx:
            nonzeros = [v for v in seq if v != 0]
            if nonzeros:
                b_loops.append(nonzeros)
        else:
            prev = None
            for idx in zero_idx:
                if prev is None:
                    if idx > 0:
                        seg = [v for v in seq[0:idx] if v != 0]
                        if seg:
                            b_loops.append(seg)
                else:
                    seg = [v for v in seq[prev + 1:idx] if v != 0]
                    if seg:
                        b_loops.append(seg)
                prev = idx
            if prev is not None and prev < len(seq) - 1:
                seg = [v for v in seq[prev + 1:] if v != 0]
                if seg:
                    b_loops.append(seg)

        loops.append(b_loops)

    return loops

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)

    
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class VRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        #self.problem_type = env_params['problem_type']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)


        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)

        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']

        self.saved_index = 0

    def load_raw_problems(self, depot_xy:torch.Tensor, 
                          node_xy:torch.Tensor,
                          node_demand:torch.Tensor,
                          Up_Bound:float=None,
                          Demand_scaler:float=None,
                          aug_factor=1):
        scaled_depot_xy, scaled_node_xy, scaled_node_demand, _, _ = rescale_raw_problem(
            depot_xy, node_xy, node_demand, Up_Bound, Demand_scaler)
        self.load_problems(scaled_depot_xy, scaled_node_xy, scaled_node_demand, aug_factor)

    def load_problems(self, depot_xy:torch.Tensor, 
                      node_xy:torch.Tensor, 
                      node_demand:torch.Tensor, 
                      aug_factor=1):
        self.batch_size = depot_xy.shape[0]
        self.problem_size = node_xy.shape[1]
        device = depot_xy.device

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)

            else:
                raise NotImplementedError
        


        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1),device=device)
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand


        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)

        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished


        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################

        self.at_the_depot = (selected == 0)

        #### update load information ###

        demand_list = self.depot_node_demand[:, None, :].expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        self.load -= selected_demand
        self.load[self.at_the_depot] = 1 # refill loaded at the depot


        #### mask nodes if load exceed ###

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.000001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)

        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)
        



        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)


        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq-rolled_seq)**2).sum(3).sqrt()

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        return travel_distances

    def get_node_seq(self):

        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        return gathering_index,ordered_seq

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
#Prompt Model Section
import pickle
from sklearn.preprocessing import normalize
class Prompt(nn.Module):
    def __init__(self, length=1, embed_dim=256, embedding_key='mean', prompt_init='uniform', prompt_pool=False, load_key = True, key_path='./keys_new_16',
                 prompt_key=False, pool_size=None, top_k=None, prompt_size=3,key_div_bound=100, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()

        self.length = length # length of token, for vrp it is set to be 1
        self.embed_dim = embed_dim # embedding size
        self.prompt_pool = prompt_pool # input a costomized prompt pool
        self.embedding_key = embedding_key # ways for calculating embedding keys
        self.prompt_init = prompt_init # ways for prompt initilization
        self.prompt_key = prompt_key # w
        self.prompt_timesused = None
        self.prompt_timesnotused = None
        self.pool_size = pool_size
        self.top_k = top_k
        self.load_key = load_key
        self.prompt_size = prompt_size
        self.key_div_bound = key_div_bound

        self.batchwise_prompt = batchwise_prompt

        self.scalor = None

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, self.prompt_size*6*embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                # shape: (pool_size, length, embedding_size)
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
                # shape: (pool_size, length, embedding_size)

        if self.load_key:
            if not os.path.exists(key_path):
                raise FileNotFoundError(f"Key file not found at {key_path}")
            keys = pickle.load(open(key_path,'rb'))
            self.prompt_key = torch.tensor(keys).to(torch.device('cuda', 0)).float()

            

        else:
            if prompt_key:
                key_shape = (pool_size, 5*embed_dim)
                if prompt_key_init == 'zero':
                    self.prompt_key = nn.Parameter(torch.zeros(key_shape)+torch.tensor([  0.124 ,  1.036 ,  1.387 ,  3.684 ,  4.764 ,  -3.025 ,  3.117 ,  0.093 ,  -0.620 ,  0.971 ,  1.587 ,  -4.532 ,  -0.227 ,  -0.471 ,  2.889 ,  0.505 ,  -0.044 ,  1.105 ,  1.052 ,  -0.651 ,  -0.538 ,  3.675 ,  2.419 ,  -1.551 ,  0.347 ,  -0.728 ,  2.461 ,  0.514 ,  -2.535 ,  0.621 ,  -0.083 ,  -1.701 ,  -0.928 ,  1.109 ,  -2.026 ,  2.628 ,  0.876 ,  -0.180 ,  -0.180 ,  -1.285 ,  0.579 ,  -0.339 ,  -0.005 ,  -0.470 ,  2.301 ,  -1.302 ,  -0.526 ,  -0.014 ,  1.643 ,  5.364 ,  2.212 ,  -0.484 ,  -3.853 ,  0.818 ,  1.090 ,  0.081 ,  0.642 ,  3.392 ,  -0.119 ,  -1.587 ,  2.153 ,  1.787 ,  -0.006 ,  -1.357 ,  -2.900 ,  -2.816 ,  0.251 ,  -1.397 ,  -0.092 ,  0.795 ,  -0.249 ,  -0.216 ,  0.344 ,  1.206 ,  -0.345 ,  -0.596 ,  -0.051 ,  -2.968 ,  -0.202 ,  0.366 ,  2.685 ,  0.993 ,  -0.380 ,  2.676 ,  -1.187 ,  -0.061 ,  -1.178 ,  1.630 ,  -0.284 ,  -0.523 ,  -1.119 ,  -0.450 ,  -0.369 ,  -0.418 ,  0.447 ,  0.117 ,  0.173 ,  0.135 ,  0.262 ,  -0.042 ,  -0.061 ,  1.150 ,  1.656 ,  -1.606 ,  2.254 ,  
                                                                                        -1.972 ,  0.063 ,  -0.531 ,  -0.272 ,  0.067 ,  -1.680 ,  0.636 ,  -0.880 ,  -0.466 ,  -0.546 ,  0.287 ,  0.685 ,  0.873 ,  0.345 ,  0.120 ,  1.483 ,  -1.598 ,  -3.699 ,  2.823 ,  -0.840 ,  -1.466 ,  0.929 ,  -0.228 ,  -0.613 ,  -0.344 ,  0.478 ,  -0.673 ,  -0.064 ,  -1.576 ,  0.646 ,  -0.114 ,  1.043 ,  0.158 ,  -0.545 ,  -0.167 ,  0.484 ,  0.447 ,  -1.539 ,  -1.020 ,  0.745 ,  -0.001 ,  -0.186 ,  0.077 ,  0.253 ,  -0.727 ,  -0.852 ,  0.962 ,  -1.424 ,  -0.920 ,  0.089 ,  0.832 ,  -0.216 ,  1.075 ,  -0.021 ,  0.355 ,  0.122 ,  0.388 ,  0.061 ,  0.257 ,  -0.141 ,  0.456 ,  0.090 ,  0.388 ,  -1.383 ,  0.281 ,  -0.505 ,  0.576 ,  -0.443 ,  -1.386 ,  -0.450 ,  0.218 ,  -0.574 ,  0.641 ,  -0.103 ,  -0.300 , 
                                                                                        -0.180 ,  -0.052 ,  0.483 ,  -0.435 ,  0.071 ,  -1.483 ,  -0.644 ,  0.103 ,  -0.726 ,  0.190 ,  0.595 ,  0.683 ,  -0.541 ,  0.338 ,  -0.543 ,  -0.653 ,  -1.010 ,  -0.277 ,  0.348 ,  0.340 ,  0.176 ,  0.263 ,  0.502 ,  -0.073 ,  0.374 ,  -0.444 ,  0.394 ,  -0.093 ,  -0.121 ,  -0.748 ,  -0.514 ,  1.306 ,  0.285 ,  -0.708 ,  -0.567 ,  -0.455 ,  -0.215 ,  0.112 ,  -0.578 ,  0.229 ,  -0.469 ,  0.469 ,  -0.068 ,  -0.745 ,  -0.618 ,  -0.610 ,  -0.430 ,  -2.769 ,  -0.070 ,  -0.166 ,  0.117 ,  -0.343 ,  -0.781 ,  1.382 ,  -0.925 ,  0.271 ,  0.148 ,  -0.492 ,  -1.443 ,  0.702 ,  -0.610 ,  0.258 ,  -0.426 ,  1.584 ,  -0.885 ,  0.060 ,  0.114 ,  0.281 ,  -1.270 ,  1.767 ,  1.515 ,  0.113 ,  0.558 ,  0.947 ,  -0.155 , 
                                                                                        -0.106 ,  0.013 ,  -0.075 ,  0.145 ,  -0.491 ,  0.192 ,  0.159 ,  -0.181 ,  0.725 ,  0.201 ,  0.391 ,  -0.471 ,  0.069 ,  -0.144 ,  -0.139 ,  -0.061 ,  0.422 ,  -0.034 ,  -0.029 ,  -0.101 ,  0.128 ,  -0.483 ,  0.180 ,  -0.321 ,  0.253 ,  -0.216 ,  -0.157 ,  -0.191 ,  0.069 , 
                                                                                        -0.296 ,  -0.329 ,  0.048 ,  0.307 ,  0.176 ,  0.035 ,  -0.317 ,  0.169 ,  -0.062 ,  0.694 ,  -0.167 ,  -0.059 ,  -0.166 ,  -0.124 ,  -0.047 ,  -0.008 ,  0.160 ,  0.355 ,  0.071 ,  0.001 ,  0.137 ,  0.088 ,  0.083 ,  0.139 ,  0.361 ,  -0.234 ,  -0.099 ,  0.156 ,  0.270 ,  0.341 , 
                                                                                        -0.528 ,  0.271 ,  0.079 ,  -0.053 ,  0.802 ,  -0.102 ,  -0.055 ,  0.048 ,  0.791 ,  -0.042 ,  0.105 ,  0.103 ,  -0.326 ,  -0.127 ,  0.128 ,  -0.083 ,  -0.399 ,  0.609 ,  -0.010 ,  -0.313 ,  0.196 ,  -0.143 ,  0.178 ,  -0.242 ,  -0.467 ,  -0.097 ,  -0.078 ,  -0.270 ,  0.036 ,  -0.086 ,  -0.074 ,  0.462 ,  0.290 ,  0.278 ,  0.019 ,  -0.146 ,  0.202 ,  0.073 ,  0.492 ,  -0.064 ,  0.102 ,  -0.470 ,  0.070 ,  -0.261 ,  -0.024 ,  0.449 ,  -0.014 ,  -0.115 ,  -0.018 ,  -0.306 ,  0.093 ,  -0.071 ,  0.011 ,  -0.627 ,  0.151 ,  0.578 ,  0.034 ,  0.170 ,  0.026 ,  0.040 ,  -0.164 ,  -0.147 ,  -0.362 ,  -0.083 ,  0.173 ,  -0.775 ,  -0.118 ,  0.716 ,  0.405 ,  -0.105 ,  -0.017 ,  0.229 ,  -0.010 ,  -0.080 ,  -0.111 ,  -0.034 ,  0.211 ,  0.192 ,  0.159 ,  -0.190 ,  0.223 ,  0.146 ,  0.113 ,  0.080 ,  -0.166 ,  -0.275 ,  0.108 ,  -0.304 ,  0.173 ,  0.281 ,  -0.208 ,  -0.292 ,  0.107 ,  -0.032 ,  -0.015 ,  -0.238 ,  -0.144 ,  0.258 ,  0.203 ,  -0.003 ,  0.127 ,  -0.130 ,  -0.016 ,  0.644 ,  0.151 ,  0.070 ,  -0.372 ,  0.295 ,  0.335 ,  0.014 ,  -0.192 ,  0.218 ,  0.222 ,  -0.050 ,  -0.089 ,  0.081 ,  -0.035 ,  0.169 ,  0.070 ,  0.275 ,  -0.126 ,  -0.006 ,  0.236 ,  0.186 ,  -0.387 ,  -0.110 ,  -0.216 ,  -0.125 ,  -0.233 ,  0.273 ,  -0.074 ,  0.108 ,  -0.111 ,  -0.144 ,  
                                                                                        0.080 ,  -0.368 ,  -0.255 ,  0.101 ,  0.135 ,  -0.579 ,  0.417 ,  -0.132 ,  0.017 ,  0.040 ,  0.078 ,  0.242 ,  0.314 ,  -0.169 ,  0.005 ,  -0.087 ,  
                                                                                        0.341 ,  -0.466 ,  -0.298 ,  -0.222 ,  -0.118 ,  -0.347 ,  0.277 ,  -0.329 ,  -0.227 ,  -0.035 ,  -0.119 ,  0.359 ,  0.000 ,  0.169 ,  0.014 ,  -0.421 ,  0.102 ,  0.385 ,  0.191 ,  -0.104 ,  0.185 ,  -0.217 ,  0.050 ,  0.132 ,  0.178 ,  0.253 ,  0.205 ,  0.265 ,  0.010 ,  0.267 ,  0.137 ,  0.019 ,  -0.145 ,  0.033 ,  -0.174 ,  -0.221 ,  0.089 ,  0.047 ,  0.396 ,  0.098 ,  0.026 ,  -0.461 ,  0.036 ,  0.061 ,  0.413 ,  -0.451 ,  -0.111 ,  -0.217 ,  0.123 ,  0.225 ,  0.113 ,  0.088 ,  -0.012 ,  -0.045 ,  -0.069 ,  -0.484 ,  0.855 ,  0.262 ,  0.023 ,  0.374 ,  0.151 ,  0.172 ,  -0.345 ,  0.140 ,  -0.741 ,  -0.215 ,  -0.330 ,  0.419 ,  0.329 ,  0.166 ,  -0.251 ,  -0.247 ,  0.046 ,  0.242 ,  -0.057 ,  -0.089 ,  -0.194 ,  -0.314 ,  0.202 ,  -0.117 ,  0.068 ,  0.429 ,  -0.211 ,  0.106 ,  -0.011 ,  -0.073 ,  -0.039 ,  -0.090 ,  -0.165 ,  0.169 ,  0.287 ,  0.149 ,  0.041 ,  0.098 ,  0.427 ,  0.279 ,  0.194 ,  -0.320 ,  -0.122 ,  0.231 ,  0.022 ,  0.586 ,  -0.260 ,  0.074 ,  0.149 ,  -0.388 ,  0.081 ,  -0.079 ,  -0.487 ,  -0.545 ,  -0.090 ,  -0.157 ,  -0.224 ,  0.147 ,  0.291 ,  0.384 ,  0.067 ,  -0.312 ,  0.337 ,  -0.241 ,  0.129 ,  0.501 ,  0.123 ,  0.235 ,  -0.255 ,  0.020 ,  0.104 ,  0.223 ,  -0.278 ,  0.021 ,  -0.037 ,  
                                                                                        -0.276 ,  -0.038 ,  -0.240 ,  -0.185 ,  0.305 ,  -0.275 ,  -0.501 ,  -0.264 ,  -0.390 ,  0.232 ,  -0.253 ,  -0.112 ,  -0.107 ,  -0.034 ,  0.096 ,  0.247 ,  0.432 ,  0.126 ,  -0.055 ,  0.073 ,  0.214 ,  0.055 ,  0.304 ,  -0.344 ,  0.059 ,  0.288 ,  0.121 ,  0.351 ,  0.233 ,  0.040 ,  -0.189 ,  -0.029 ,  -0.022 ,  0.230 ,  0.063 ,  0.338 ,  -0.333 ,  0.100 ,  0.347 ,  -0.204 ,  -0.141 ,  0.271 ,  -0.104 ,  -0.343 ,  0.564 ,  -0.113 ,  0.451 ,  -0.009 ,  -0.240 ,  0.029 ,  0.182 ,  -0.216 ,  0.014 ,  -0.323 ,  -0.130 ,  0.128 ,  0.074 ,  0.140 ,  -0.042 ,  0.065 ,  -0.110 ,  0.079 ,  -0.045 ,  0.252 ,  -0.011 ,  -0.468 ,  -0.124 ,  0.172 ,  0.156 ,  0.157 ,  -0.081 ,  0.234 ,  0.265 ,  -0.165 ,  -0.170 ,  -0.121 ,  -0.234 ,  0.011 ,  0.013 ,  0.050 ,  0.150 ,  0.026 ,  -0.054 ,  -0.260 ,  0.343 ,  -0.373 ,  0.206 ,  -0.270 ,  0.164 ,  0.230 ,  0.024 ,  0.068 ,  0.165 ,  -0.075 ,  0.145 ,  0.114 ,  0.016 ,  -0.369 ,  -0.192 ,  0.275 ,  -0.341 ,  -0.062 ,  0.235 ,  0.318 ,  -0.134 ,  0.049 ,  -0.022 ,  0.191 ,  0.008 ,  0.598 ,  0.128 ,  0.146 ,  -0.146 ,  0.320 ,  -0.072 , 
                                                                                        -0.137 ,  0.083 ,  -0.138 ,  -0.135 ,  0.397 ,  -0.037 ,  0.068 ,  -0.076 ,  0.084 ,  -0.066 ,  -0.187 ,  -0.186 ,  0.062 ,  0.008 ,  0.268 ,  0.147 ,  0.110 ,  0.046 ,  -0.186 ,  -0.310 ,  -0.428 ,  -0.122 ,  0.197 ,  -0.221 ,  -0.038 ,  -0.251 ,  -0.196 ,  0.254 ,  0.297 ,  0.250 ,  0.104 ,  -0.036 ,  -0.272 ,  -0.226 ,  0.178 ,  -0.225 ,  -0.051 ,  0.062 ,  0.200 ,  0.042 ,  0.053 ,  0.019 ,  -0.100 ,  -0.121 ,  -0.354 ,  0.314 ,  -0.085 ,  0.034 ,  -0.116 ,  0.349 ,  -0.357 ,  -0.031 ,  -0.330 ,  -0.023 ,  0.091 ,  0.104 ,  -0.416 ,  -0.120 , ],requires_grad=True).expand(key_shape))
                elif prompt_key_init == 'uniform':

                    self.prompt_key = nn.Parameter(nn.init.uniform_(torch.randn(key_shape), -0.5, 0.5) +torch.tensor([  -0.613 ,  -0.344 ,  0.478 ,  -0.673 ,  -0.064 ,  -1.576 ,  0.646 ,  -0.114 ,  1.043 ,  0.158 ,  -0.545 , 
                                                                                                                    -0.167 ,  0.484 ,  0.447 ,  -1.539 ,  -1.020 ,  0.745 ,  -0.001 ,  -0.186 ,  0.077 ,  0.253 ,  -0.727 ,  -0.852 ,  0.962 ,  -1.424 ,  -0.920 ,  0.089 ,  0.832 ,  -0.216 ,  1.075 ,  -0.021 ,  0.355 ,  0.122 ,  0.388 ,  0.061 ,  0.257 ,  -0.141 ,  0.456 ,  0.090 ,  0.388 ,  -1.383 ,  0.281 ,  
                                                                                                                    -0.505 ,  0.576 ,  -0.443 ,  -1.386 ,  -0.450 ,  0.218 ,  -0.574 ,  0.641 ,  -0.103 ,  -0.300 ,  -0.180 ,  -0.052 ,  0.483 ,  -0.435 ,  0.071 ,  -1.483 ,  -0.644 ,  0.103 ,  -0.726 ,  0.190 ,  0.595 ,  0.683 ,  -0.541 ,  0.338 ,  -0.543 ,  -0.653 ,  -1.010 ,  -0.277 ,  0.348 ,  0.340 ,  0.176 ,  
                                                                                                                    0.263 ,  0.502 ,  -0.073 ,  0.374 ,  -0.444 ,  0.394 ,  -0.093 ,  -0.121 ,  -0.748 ,  -0.514 ,  1.306 ,  0.285 ,  -0.708 ,  -0.567 ,  -0.455 ,  -0.215 ,  0.112 ,  -0.578 ,  0.229 ,  -0.469 ,  0.469 ,  -0.068 ,  -0.745 ,  -0.618 ,  -0.610 ,  -0.430 ,  -2.769 ,  -0.070 ,  -0.166 ,  0.117 ,  -0.343 , 
                                                                                                                        -0.781 ,  1.382 ,  -0.925 ,  0.271 ,  0.148 ,  -0.492 ,  -1.443 ,  0.702 ,  -0.610 ,  0.258 ,  -0.426 ,  1.584 ,  -0.885 ,  0.060 ,  0.114 ,  0.281 ,  -1.270 ,  1.767 ,  1.515 ,  0.113 ,  0.558 ,  0.947 ,  -0.155 ,  -0.106 ,  0.013 ,  -0.075 ,  0.145 ,  -0.491 ,  0.192 ,  0.159 ,  -0.181 ,  0.725 , 
                                                                                                                        0.201 ,  0.391 ,  -0.471 ,  0.069 ,  -0.144 ,  -0.139 ,  -0.061 ,  0.422 ,  -0.034 ,  -0.029 ,  -0.101 ,  0.128 ,  -0.483 ,  0.180 ,  -0.321 ,  0.253 ,  -0.216 ,  -0.157 ,  -0.191 ,  0.069 ,  -0.296 ,  -0.329 ,  0.048 ,  0.307 ,  0.176 ,  0.035 ,  -0.317 ,  0.169 ,  -0.062 ,  0.694 ,  -0.167 ,  -0.059 , 
                                                                                                                        -0.166 ,  -0.124 ,  -0.047 ,  -0.008 ,  0.160 ,  0.355 ,  0.071 ,  0.001 ,  0.137 ,  0.088 ,  0.083 ,  0.139 ,  0.361 ,  -0.234 ,  -0.099 ,  0.156 ,  0.270 ,  0.341 ,  -0.528 ,  0.271 ,  0.079 ,  -0.053 ,  0.802 ,  -0.102 ,  -0.055 ,  0.048 ,  0.791 ,  -0.042 ,  0.105 ,  0.103 ,  -0.326 ,  -0.127 ,  0.128 , 
                                                                                                                            -0.083 ,  -0.399 ,  0.609 ,  -0.010 ,  -0.313 ,  0.196 ,  -0.143 ,  0.178 ,  -0.242 ,  -0.467 ,  -0.097 ,  -0.078 ,  -0.270 ,  0.036 ,  -0.086 ,  -0.074 ,  0.462 ,  0.290 ,  0.278 ,  0.019 ,  -0.146 ,  0.202 ,  0.073 ,  0.492 ,  -0.064 ,  0.102 ,  -0.470 ,  0.070 ,  -0.261 ,  -0.024 ,  0.449 ,  -0.014 ,  -0.115 , 
                                                                                                                            -0.018 ,  -0.306 ,  0.093 ,  -0.071 ,  0.011 ,  -0.627 ,  0.151 ,  0.578 ,  0.034 ,  0.170 ,  0.026 ,  0.040 ,  -0.164 ,  -0.147 ,  -0.362 ,  -0.083 ,  0.173 ,  -0.775 ,  -0.118 ,  0.716 ,  0.405 ,  -0.105 ,  -0.017 ,  0.229 ,  -0.010 ,  -0.080 ,  -0.111 ,  -0.034 ,  0.211 ,  0.192 ,  0.159 ,  -0.190 ,  0.223 , 
                                                                                                                                0.146 ,  0.113 ,  0.080 ,  -0.166 ,  -0.275 ,  0.108 ,  -0.304 ,  0.173 ,  0.281 ,  -0.208 ,  -0.292 ,  0.107 ,  -0.032 ,  -0.015 ,  -0.238 ,  -0.144 ,  0.258 ,  0.203 ,  -0.003 ,  0.127 ,  -0.130 ,  -0.016 ,  0.644 ,  0.151 ,  0.070 ,  -0.372 ,  0.295 ,  0.335 ,  0.014 ,  -0.192 ,  0.218 ,  0.222 ,  -0.050 ,
                                                                                                                                    -0.089 ,  0.081 ,  -0.035 ,  0.169 ,  0.070 ,  0.275 ,  -0.126 ,  -0.006 ,  0.236 ,  0.186 ,  -0.387 ,  -0.110 ,  -0.216 ,  -0.125 ,  -0.233 ,  0.273 ,  -0.074 ,  0.108 ,  -0.111 ,  -0.144 ,  0.080 ,  -0.368 ,  -0.255 ,  0.101 ,  0.135 ,  -0.579 ,  0.417 ,  -0.132 ,  0.017 ,  0.040 ,  0.078 ,  0.242 ,  0.314 ,
                                                                                                                                        -0.169 ,  0.005 ,  -0.087 ,  0.341 ,  -0.466 ,  -0.298 ,  -0.222 ,  -0.118 ,  -0.347 ,  0.277 ,  -0.329 ,  -0.227 ,  -0.035 ,  -0.119 ,  0.359 ,  0.000 ,  0.169 ,  0.014 ,  -0.421 ,  0.102 ,  0.385 ,  0.191 ,  -0.104 ,  0.185 ,  -0.217 ,  0.050 ,  0.132 ,  0.178 ,  0.253 ,  0.205 ,  0.265 ,  0.010 ,  0.267 , 
                                                                                                                                        0.137 ,  0.019 ,  -0.145 ,  0.033 ,  -0.174 ,  -0.221 ,  0.089 ,  0.047 ,  0.396 ,  0.098 ,  0.026 ,  -0.461 ,  0.036 ,  0.061 ,  0.413 ,  -0.451 ,  -0.111 ,  -0.217 ,  0.123 ,  0.225 ,  0.113 ,  0.088 ,  -0.012 ,  -0.045 ,  -0.069 ,  -0.484 ,  0.855 ,  0.262 ,  0.023 ,  0.374 ,  0.151 ,  0.172 ,  -0.345 ,  
                                                                                                                                        0.140 ,  -0.741 ,  -0.215 ,  -0.330 ,  0.419 ,  0.329 ,  0.166 ,  -0.251 ,  -0.247 ,  0.046 ,  0.242 ,  -0.057 ,  -0.089 ,  -0.194 ,  -0.314 ,  0.202 ,  -0.117 ,  0.068 ,  0.429 ,  -0.211 ,  0.106 ,  -0.011 ,  -0.073 ,  -0.039 ,  -0.090 ,  -0.165 ,  0.169 ,  0.287 ,  0.149 ,  0.041 ,  0.098 ,  0.427 ,  0.279 ,
                                                                                                                                            0.194 ,  -0.320 ,  -0.122 ,  0.231 ,  0.022 ,  0.586 ,  -0.260 ,  0.074 ,  0.149 ,  -0.388 ,  0.081 ,  -0.079 ,  -0.487 ,  -0.545 ,  -0.090 ,  -0.157 ,  -0.224 ,  0.147 ,  0.291 ,  0.384 ,  0.067 ,  -0.312 ,  0.337 ,  -0.241 ,  0.129 ,  0.501 ,  0.123 ,  0.235 ,  -0.255 ,  0.020 ,  0.104 ,  0.223 ,  -0.278 , 
                                                                                                                                                0.021 ,  -0.037 ,  -0.276 ,  -0.038 ,  -0.240 ,  -0.185 ,  0.305 ,  -0.275 ,  -0.501 ,  -0.264 ,  -0.390 ,  0.232 ,  -0.253 ,  -0.112 ,  -0.107 ,  -0.034 ,  0.096 ,  0.247 ,  0.432 ,  0.126 ,  -0.055 ,  0.073 ,  0.214 ,  0.055 ,  0.304 ,  -0.344 ,  0.059 ,  0.288 ,  0.121 ,  0.351 ,  0.233 ,  0.040 ,  -0.189 , 
                                                                                                                                                -0.029 ,  -0.022 ,  0.230 ,  0.063 ,  0.338 ,  -0.333 ,  0.100 ,  0.347 ,  -0.204 ,  -0.141 ,  0.271 ,  -0.104 ,  -0.343 ,  0.564 ,  -0.113 ,  0.451 ,  -0.009 ,  -0.240 ,  0.029 ,  0.182 ,  -0.216 ,  0.014 ,  -0.323 ,  -0.130 ,  0.128 ,  0.074 ,  0.140 ,  -0.042 ,  0.065 ,  -0.110 ,  0.079 ,  -0.045 ,  0.252 ,  -0.011 , 
                                                                                                                                                    -0.468 ,  -0.124 ,  0.172 ,  0.156 ,  0.157 ,  -0.081 ,  0.234 ,  0.265 ,  -0.165 ,  -0.170 ,  -0.121 ,  -0.234 ,  0.011 ,  0.013 ,  0.050 ,  0.150 ,  0.026 ,  -0.054 ,  -0.260 ,  0.343 ,  -0.373 ,  0.206 ,  -0.270 ,  0.164 ,  0.230 ,  0.024 ,  0.068 ,  0.165 ,  -0.075 ,  0.145 ,  0.114 ,  0.016 ,  -0.369 ,  -0.192 ,  0.275 ,  -0.341 ,
                                                                                                                                                        -0.062 ,  0.235 ,  0.318 ,  -0.134 ,  0.049 ,  -0.022 ,  0.191 ,  0.008 ,  0.598 ,  0.128 ,  0.146 ,  -0.146 ,  0.320 ,  -0.072 ,  -0.137 ,  0.083 ,  -0.138 ,  -0.135 ,  0.397 ,  -0.037 ,  0.068 ,  -0.076 ,  0.084 ,  -0.066 ,  -0.187 ,  -0.186 ,  0.062 ,  0.008 ,  0.268 ,  0.147 ,  0.110 ,  0.046 ,  -0.186 ,  -0.310 ,  -0.428 ,  -0.122 ,
                                                                                                                                                            0.197 ,  -0.221 ,  -0.038 ,  -0.251 ,  -0.196 ,  0.254 ,  0.297 ,  0.250 ,  0.104 ,  -0.036 ,  -0.272 ,  -0.226 ,  0.178 ,  -0.225 ,  -0.051 ,  0.062 ,  0.200 ,  0.042 ,  0.053 ,  0.019 ,  -0.100 ,  -0.121 ,  -0.354 ,  0.314 ,  -0.085 ,  0.034 ,  
                                                                                                                    -0.116 ,  0.349 ,  -0.357 ,  -0.031 ,  -0.330 ,  -0.023 ,  0.091 ,  0.104 ,  -0.416 ,  -0.120 ,  ],requires_grad=True).expand(key_shape))
                    
                    

            else:
                # else use mean of prompt as key
                # only compatible with prompt, not prefix
                prompt_mean = torch.mean(self.prompt, dim=1)
                self.prompt_key = prompt_mean
                # shape: (pool_size, 1, embedding_size)
                #print(self.prompt_key.requires_grad)
        
        self.prompt_timesused = torch.ones(pool_size)
        self.prompt_timesnotused = torch.ones(pool_size)
        self.prompt_weight = torch.ones(pool_size)
        
        #self.prompt_key = self.prompt_key + promt_base.expand(size=(self.prompt_key.shape))
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.max(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, selected_id = None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':

                x_embed_mean_std = torch.cat((torch.mean(x_embed, dim=1),torch.std(x_embed, dim=1)),dim=1)

            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")


            x_embed_mean_std_norm = normalize(x_embed_mean_std.cpu().numpy(),axis=1)
            
            x_embed_mean_std_norm_gpu = torch.from_numpy(x_embed_mean_std_norm).to(torch.device('cuda', 0))

            similarity = -torch.cdist(x_embed_mean_std_norm_gpu, self.prompt_key, p=2.0)
            
            if selected_id is None:
                if prompt_mask is None:
                    _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k

                    if self.batchwise_prompt:
                        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                        # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                        # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                        # Unless dimension is specified, this will be flattend if it is not already 1D.
                        if prompt_id.shape[0] < self.pool_size:
                            prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                            id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                        _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                        major_prompt_id = prompt_id[major_idx] # top_k
                        # expand to batch
                        idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
                else:
                    idx = prompt_mask # B, top_k
            else:
                idx = selected_id

            batched_prompt_raw = self.prompt[idx] #top_k, B, length, C

            batch_size, topk,  length, c = batched_prompt_raw.shape

            batched_prompt = batched_prompt_raw.reshape(batch_size,topk, self.prompt_size * length, int(c/self.prompt_size)) # B, top_k * length, C

            out['prompt_idx'] = idx
            # shape: (pool_size, top_k)


            out['reduce_sim'] = 1.0
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)


        out['prompt_embedding'] = batched_prompt.reshape(batch_size,topk,self.prompt_size,6,self.embed_dim)
        # shape: (batch, top_k * length, embedding)

        return out
    

##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# VRP Model
class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = VRP_Encoder(**model_params)
        self.decoder = VRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

        self.promptpool = Prompt(length=1,  # not clear
                                 embed_dim=model_params['embedding_dim'], 
                                 embedding_key='mean', 
                                 prompt_init='uniform', 
                                 prompt_pool=True,
                                 load_key= True, 
                                 prompt_key=True, 
                                 key_path=model_params['key_path'],
                                 pool_size=model_params['pool_size'], 
                                 top_k= model_params['top_k'], 
                                 prompt_size=5,
                                 key_div_bound = 1000,
                                 batchwise_prompt=False, 
                                 prompt_key_init='uniform')

        self.prompt_mask_grad = None

        self.prompt_key_mask_grad = None

        self.prompt_id_last = None

        self.prompt_id_select = None

        self.prompts = None


    def pre_forward_prompt(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)


        self.encoded_nodes, encoded_nodes_prompt = self.encoder(depot_xy, node_xy_demand, None)

        if (not self.prompt_id_last == None ):

            self.prompt_mask_grad[self.prompt_id_last] = self.promptpool.prompt[self.prompt_id_last]

            #self.prompt_key_mask_grad[self.prompt_id_last] = self.promptpool.prompt_key[self.prompt_id_last]

            self.promptpool.prompt.data = self.prompt_mask_grad

            self.promptpool.prompt_key.data = self.prompt_key_mask_grad
        
        out = self.promptpool(encoded_nodes_prompt)


        self.prompts = out['prompt_embedding']

        # print(self.prompts.shape)
        # self.encoded_nodes, encoded_nodes_prompt = self.encoder(depot_xy, node_xy_demand, prompt_embeddin_mean)


        # node_size = node_xy.shape[1]+1
        # self.decoder.set_kv(self.encoded_nodes[:,:node_size,:])
        
        self.prompt_mask_grad = self.promptpool.prompt.clone()

        self.prompt_key_mask_grad = self.promptpool.prompt_key.clone()

        self.prompt_id_last = out['prompt_idx']


        return out['reduce_sim'], self.prompt_id_last

    def pre_forward(self, reset_state, prompt_id):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)


        self.encoded_nodes, encoded_nodes_prompt = self.encoder(depot_xy, node_xy_demand, self.prompts[:,prompt_id,:,:,:])

        node_size = node_xy.shape[1]+1
        self.decoder.set_kv(self.encoded_nodes[:,:node_size,:])

        return 



    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)


        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)

            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)


            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():

                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_TW, embedded_prompt):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand_TW)
        # input shape: (batch, problem, 5)
        # 6 features are: x_coord, y_coord, demands, earlyTW, lateTW
        # embedded_node shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)     

        n = 0
        for layer in self.layers:        
            if not embedded_prompt == None:

                #if n == 5:
                out = torch.cat((out, embedded_prompt[:,:,n,:]), dim=1)
                # shape: (batch, problem+1+k-top, embedding)

                out, out_prompt = layer(out)
            #if n == 0:

            else:
                if n < 1:
                    out, out_prompt = layer(out)
                else:  
                    out, out_prompt_layer = layer(out)
                    out_prompt = torch.cat((out_prompt,out_prompt_layer),2)

              
            # else:
            #     out, _ = layer(out)  
            n = n +1


        #out = self.layerlast (out)

        return out, out_prompt
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)


        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3, multi_head_out
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class VRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, load,  ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+3)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # q_last shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)


        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)


        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
    
##### ##### ##### ##### ##### ##### ##### ##### ##### #####
#Solution
class CVRPSolver:

    def __init__(self, env_params,model_params,solver_params):
        
        self.env_params = env_params
        self.model_params = model_params
        self.solver_params = solver_params

        # cuda
        USE_CUDA = self.solver_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.solver_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        if env_params:
            self.env=VRPEnv(**env_params)
        else:
            self.env=None

        self.model = VRPModel(**model_params)
        # Restore
        model_load = self.solver_params['model_load']
        checkpoint_fullname = model_load['model_path']
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    def solve_cvrp(self,depot_xy, node_xy, node_demand,Up_Bound=None, Demand_scaler=None):
        # Augmentation
        if self.solver_params['augmentation_enable']:
            aug_factor = self.solver_params['aug_factor']
        else:
            aug_factor = 1

        if self.env is None:
            problem_size=node_xy.size(1)
            pomo_size=problem_size
            self.env=VRPEnv(problem_size=problem_size, pomo_size=pomo_size)

        # Ready
        self.model.eval()
        batch_size = depot_xy.size(0)
        with torch.no_grad():
            self.env.load_raw_problems(depot_xy=depot_xy, node_xy=node_xy, node_demand=node_demand, Up_Bound=Up_Bound, Demand_scaler=Demand_scaler, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward_prompt(reset_state)

        # For each top-k iteration we will pick the single best across all augmentations AND all pomo ids
        best_reward_per_topk = []      # list of tensors shape (batch,)
        best_aug_idx_list = []        # list of tensors shape (batch,)
        best_pomo_idx_list = []       # list of tensors shape (batch,)
        best_selected_per_topk = []   # list of tensors shape (batch, seq_len)

        for i in range(self.model_params['top_k']):
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state, i)
            # POMO Rollout
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

            # reward: (aug_factor * batch_size, pomo)
            # reshape to (aug, batch, pomo)
            aug_reward = reward.view(aug_factor, batch_size, self.env.pomo_size)

            # Flatten aug and pomo dims to find global best per batch
            # permute to (batch, aug, pomo) then reshape to (batch, aug*pomo)
            flat = aug_reward.permute(1, 0, 2).reshape(batch_size, -1)
            best_vals, best_flat_idx = flat.max(dim=1)  # (batch,)

            # decode aug_idx and pomo_idx
            pomo = self.env.pomo_size
            aug_idx = (best_flat_idx // pomo)
            pomo_idx = (best_flat_idx % pomo)

            # capture selected_node_list and reshape to (aug, batch, pomo, seq_len)
            sel_list = self.env.selected_node_list.view(aug_factor, batch_size, self.env.pomo_size, -1)
            # seq_len = sel_list.shape[-1]

            # use advanced indexing to pick per-batch selected sequence
            batch_idx = torch.arange(batch_size, device=sel_list.device)
            chosen = sel_list[aug_idx, batch_idx, pomo_idx]  # shape: (batch, seq_len)

            best_reward_per_topk.append(best_vals)
            best_aug_idx_list.append(aug_idx)
            best_pomo_idx_list.append(pomo_idx)
            best_selected_per_topk.append(chosen)

        # Stack across top_k -> shapes (top_k, batch, ...)
        best_reward_per_topk = torch.stack(best_reward_per_topk, dim=0)       # (top_k, batch)
        # best_aug_idx_all = torch.stack(best_aug_idx_list, dim=0)              # (top_k, batch)
        # best_pomo_idx_all = torch.stack(best_pomo_idx_list, dim=0)            # (top_k, batch)
        best_selected_all = torch.stack(best_selected_per_topk, dim=0)        # (top_k, batch, seq_len)

        # Now select for each batch the best across top_k
        best_topk_reward, best_topk_idx = best_reward_per_topk.max(dim=0)  # (batch,)

        # Gather final selected sequences per batch
        batch_idx = torch.arange(batch_size, device=best_selected_all.device)
        final_selected = best_selected_all[best_topk_idx, batch_idx]  # (batch, seq_len)

        distances=-best_topk_reward # negative to get positive values
        routes= journey2loops(final_selected)
        return distances, routes

import os
import pickle
def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def load_dataset(filename, disable_print=False):

    with open(check_extension(filename), 'rb') as f:
        data = pickle.load(f)
        if not disable_print:
            print(">> Load {} data ({}) from {}".format(len(data), type(data), filename))
        return data
    
def get_CVRPSolver(model_path, keys_path, problem_size=None, pomo_size=None, use_cuda=True, cuda_device_num=0,):
    '''
    
    :param model_path: the directory path where the pre-trained model is saved
    :param model_name: the name of the pre-trained model to load
    :param keys_path: the path to the keys file `keys_new_16` for the prompt model
    :param problem_size: the number of customer nodes in the CVRP problem
    :param pomo_size: the number of POMO samples, usually set equal to `problem_size`
    :param use_cuda: whether to use CUDA for computation
    :param cuda_device_num: the CUDA device number to use, `0` by default
    '''
    # print(f"problem_size: {problem_size}, pomo_size: {pomo_size}")
    if problem_size is None and pomo_size is None:
        env_params={}
    else:
        env_params = {
        #'problem_type': 'CVRP', 
        'problem_size': problem_size, 
        'pomo_size':pomo_size, 
        }
    model_params = {
        'pool_size': 16, # size of prompt pool
        'top_k': 1, # try the top k prompts
        'key_path': keys_path, # path to keys_new_16
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
    }

    solver_params = {
    'use_cuda': use_cuda,
    'cuda_device_num': cuda_device_num,
    'model_load': {
        'model_path': model_path,  # path of pre-trained model and log files saved.
    },
    'augmentation_enable': True,
    'aug_factor': 8,
    }
    Solver= CVRPSolver(env_params,model_params,solver_params)
    return Solver
    
if __name__ == "__main__":
    DEBUG_MODE = False
    USE_CUDA = not DEBUG_MODE
    CUDA_DEVICE_NUM = 0

    # env_params = {
    # #'problem_type': 'CVRP',
    # 'problem_size': 50, # not used 
    # 'pomo_size':50, # not used
    # }

    # model_params = {
    #     'pool_size': 16, # size of prompt pool
    #     'top_k': 1, # try the top k prompts
    #     'key_path': './keys_new_16', # path to keys_new_16
    #     'embedding_dim': 128,
    #     'sqrt_embedding_dim': 128**(1/2),
    #     'encoder_layer_num': 6,
    #     'qkv_dim': 16,
    #     'head_num': 8,
    #     'logit_clipping': 10,
    #     'ff_hidden_dim': 512,
    #     'eval_type': 'argmax',
    # }

    # solver_params = {
    # 'use_cuda': USE_CUDA,
    # 'cuda_device_num': CUDA_DEVICE_NUM,
    # 'model_load': {
    #     'path': '../checkpoints/prompt_vrp',  # directory path of pre-trained model and log files saved.
    #     'epoch': 10000,  # epoch version of pre-trained model to laod.
    # },
    # 'augmentation_enable': True,
    # 'aug_factor': 8,
    # } 

    Solver= get_CVRPSolver(model_path='../checkpoints/prompt_vrp/checkpoint-10000.pt', keys_path='./keys_new_16', problem_size=50, pomo_size=50, use_cuda=USE_CUDA, cuda_device_num=CUDA_DEVICE_NUM)
    problem_data = load_dataset('./data/vrp_cluster50_10000.pkl')
    print(f"Loaded {len(problem_data)} problems.")
    batch_size = 256

    problems = problem_data[:batch_size]

    depot_xy = torch.Tensor([p[0] for p in problems]).to(Solver.device)
    if depot_xy.dim() == 2:
        depot_xy = depot_xy.unsqueeze(1)
    node_xy = torch.Tensor([p[1] for p in problems]).to(Solver.device)

    node_demand = torch.Tensor([p[2] for p in problems]).to(Solver.device)
    capacity = torch.Tensor([p[3] for p in problems]).to(Solver.device)
    print(f"{depot_xy.shape}, {node_xy.shape}, {node_demand.shape}, {capacity.shape}")
    Demand_scaler = torch.max(capacity, dim=0)[0]
    print(Demand_scaler)
    distances, routes = Solver.solve_cvrp(depot_xy, node_xy, node_demand, Up_Bound=1.0, Demand_scaler=Demand_scaler)
    num_routes=5
    for i in range(num_routes):
        print(f"Problem {i+1}:")
        print("distance:", distances[i].item())
        print("routes:", routes[i])