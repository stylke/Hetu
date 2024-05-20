import math
import heapq
import pulp
from typing import List, Dict, Any
from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from .utils import Args, TrainerCtxs, TrainerStrategyArgs
from .parallel_config import generate_gpt_3d_config, config_spread_zero

DEVICES_PER_NODE = 8

class TPGroup(Args):
    def __init__(
        self,
        ctxs: TrainerCtxs,
        node_idx: int,
        tp: int,
        devices: List[int],
        sr: List[float],
    ):
        assert len(devices) == len(sr), "length mismatches"
        tp_devices_num = len(devices)
        self.node_idx = node_idx
        self.tp = tp
        self.devices = sorted(devices)
        self.hetero_ratio = self.tp // tp_devices_num
        self.alpha = ctxs.hetero_tp_alpha[int(math.log2(self.tp // tp_devices_num))]
        self.sr = max(sr) * self.alpha
        # 减少搜索空间
        if self.sr < ctxs.straggler_threshold:
            self.sr = 1.0

class StrategyModel:
    def __init__(
        self, 
        ctxs: TrainerCtxs,
        old_strategy_args: TrainerStrategyArgs, 
        used_devices_sr: Dict[int, float], 
        suspended_devices_sr: Dict[int, float],
        unused_devices: List[int]
    ):
        self.all_devices_num = old_strategy_args.dp * old_strategy_args.tp * old_strategy_args.pp
        self.all_nodes_num = self.all_devices_num // DEVICES_PER_NODE
        assert len(used_devices_sr) + len(suspended_devices_sr) + len(unused_devices) ==  self.all_devices_num, \
            "the sum of used devices and unused devices should be equal to all devices"
        self.ctxs = ctxs
        self.dp = old_strategy_args.dp
        self.tp = old_strategy_args.tp
        self.pp = old_strategy_args.pp
        self.zero = old_strategy_args.zero
        self.used_devices_sr = used_devices_sr
        self.suspended_devices_sr = suspended_devices_sr
        self.unused_devices = unused_devices
        # Results
        self.strategies = None
        self.ds_parallel_configs = None
        
    def __eq__(self, other):
        if not isinstance(other, StrategyModel):
            return False
        flag = (
            self.ctxs == other.ctxs and 
            self.dp == other.dp and 
            self.tp == other.tp and
            self.pp == other.pp and
            self.zero == other.zero and
            self.unused_devices == other.unused_devices
        )
        if flag == False:
            return flag_1
        for k, v in self.used_devices_sr.items():
            if k not in other.used_devices_sr:
                return False
            if abs(v - other.used_devices_sr[k]) > self.ctxs.straggler_safe_gap:
                return False
        for k, v in self.suspended_devices_sr.items():
            if k not in other.suspended_devices_sr:
                return False
            if abs(v - other.suspended_devices_sr[k]) > self.ctxs.straggler_safe_gap:
                return False
        return True
     
    def make_plans(self):
        if self.strategies != None and self.ds_parallel_configs != None:
            return self.strategies, self.ds_parallel_configs
        all_tp_groups, new_suspended_devices, new_unused_devices = self.sovle_tp_arrangments()
        pipelines_list, l_values_list, m_values_list = self.enumerate_pp_pattern(all_tp_groups)
        total_strategy_num = len(pipelines_list)
        self.strategies = []
        self.ds_parallel_configs = []
        for strategy_id in range(total_strategy_num):
            rank_to_device_mapping = {}
            suspended_rank_list = []
            unused_rank_list = []
            pipelines = pipelines_list[strategy_id]
            for stage_idx in range(self.pp):
                for pipeline_idx in range(self.dp):
                    start_rank_idx = stage_idx * self.tp * self.dp + pipeline_idx * self.tp
                    for rank_idx in range(start_rank_idx, start_rank_idx + self.tp):
                        tp_size = len(pipelines[pipeline_idx][stage_idx].devices)
                        if rank_idx - start_rank_idx < tp_size:
                            rank_to_device_mapping[rank_idx] = pipelines[pipeline_idx][stage_idx].devices[rank_idx - start_rank_idx]
                        else:
                            if len(suspended_rank_list) < len(new_suspended_devices):
                                # 随便建立一个映射即可
                                rank_to_device_mapping[rank_idx] = new_suspended_devices[len(suspended_rank_list)]
                                suspended_rank_list.append(rank_idx)
                            elif len(unused_rank_list) < len(new_unused_devices):
                                # 随便建立一个映射即可
                                rank_to_device_mapping[rank_idx] = new_unused_devices[len(unused_rank_list)]
                                unused_rank_list.append(rank_idx)
                            else:
                                assert False, "unreachable"
            strategy = TrainerStrategyArgs(
                dp=self.dp,
                tp=self.tp,
                pp=self.pp,
                zero=self.zero,
                rank_to_device_mapping=rank_to_device_mapping,
                suspended_rank_list=suspended_rank_list,
                unused_rank_list=unused_rank_list,
                hetero_data=True,
                hetero_layers=l_values_list[strategy_id],
                hetero_micro_batch_num_list=m_values_list[strategy_id]
            )
            ds_parallel_config = config_spread_zero(
                generate_gpt_3d_config(
                    rank_to_device_mapping, 
                    suspended_rank_list + unused_rank_list, 
                    l_values_list[strategy_id], 
                    self.pp * self.ctxs.normal_layers, 
                    self.dp * self.tp * self.pp, 
                    self.dp, 
                    self.tp, 
                    self.pp, 
                    self.zero
                )
            )
            self.strategies.append(strategy)
            self.ds_parallel_configs.append(ds_parallel_config)
        return self.strategies, self.ds_parallel_configs
        
    def sovle_tp_arrangments(self):
        available_devices_sr = self.used_devices_sr.copy()
        available_devices_sr.update(self.suspended_devices_sr)
        # print(f"available_devices_sr = {available_devices_sr}")
        all_tp_groups = []
        new_suspended_devices = []
        # 对于所有node
        for node_idx in range(self.all_nodes_num):
            node_available_devices_num = 0
            node_devices_num = DEVICES_PER_NODE
            node_devices_range = range(node_idx * node_devices_num, (node_idx + 1) * node_devices_num)
            for device_idx in node_devices_range:
                if device_idx in self.unused_devices:
                    continue
                node_available_devices_num += 1
            node_max_homo_devices_num = node_devices_num - self.tp
            assert node_available_devices_num > node_max_homo_devices_num, \
                "unused ranks currently shouldn't be larger than tp - 1"
            node_available_devices_sr = {k: v for k, v in available_devices_sr.items() if k in node_devices_range}
            # 非straggler按照device的序号升序排列
            # straggler按照sr升序排
            node_available_devices_sr = sorted(
                node_available_devices_sr.items(), 
                key=lambda x: x[0] if x[1] < self.ctxs.straggler_threshold else x[1] * node_devices_num, 
            )
            print("node_available_devices_sr =", node_available_devices_sr)
            hetero_tp = 1
            start_idx = node_max_homo_devices_num - 1
            hetero_tp_val = []
            while True:
                idx = start_idx + hetero_tp
                if idx > node_available_devices_num - 1:
                    break
                relative_hetero_idx = int(math.log2(self.tp // hetero_tp))
                alpha = self.ctxs.hetero_tp_alpha[relative_hetero_idx]
                weight = self.ctxs.hetero_tp_weight[relative_hetero_idx]
                sr = node_available_devices_sr[idx][1]
                hetero_tp_val.append(alpha * sr * weight)
                hetero_tp *= 2
            best_hetero_tp_val = min(hetero_tp_val)
            final_relative_hetero_idx = hetero_tp_val.index(best_hetero_tp_val)
            final_hetero_tp = 2 ** final_relative_hetero_idx
            final_used_devices = node_devices_num - self.tp + final_hetero_tp
            # print("hetero_tp_val =", hetero_tp_val, "final_used_devices =", final_used_devices)
            # 求解tp组
            for tp_idx in range(math.ceil(final_used_devices / self.tp)):
                tp_devices = []
                tp_sr = []
                for idx in range(tp_idx * self.tp, min((tp_idx + 1) * self.tp, final_used_devices)):
                    tp_devices.append(node_available_devices_sr[idx][0])
                    tp_sr.append(node_available_devices_sr[idx][1])
                all_tp_groups.append(TPGroup(self.ctxs, node_idx, self.tp, tp_devices, tp_sr))    
            # 求解新的suspended devices
            for idx in range(final_used_devices, node_available_devices_num):
                new_suspended_devices.append(node_available_devices_sr[idx][0])
        return all_tp_groups, new_suspended_devices, self.unused_devices
    
    def enumerate_pp_pattern(self, all_tp_groups: List[TPGroup]):
        # 启发式
        # 大部分tp是正常的
        # straggler tp groups按straggler ratio从大到小排
        Args.print_args_list(all_tp_groups)     
        straggler_tp_groups = [tp_group for tp_group in all_tp_groups if tp_group.sr > 1.0]
        straggler_tp_groups.sort(
            key=lambda x: x.sr, 
            reverse=True
        )
        # normal tp groups按node分后再按device编号从小到大排
        normal_tp_groups_mapping = {
            node_idx: [tp_group for tp_group in all_tp_groups if tp_group.sr == 1.0 and tp_group.node_idx == node_idx]
                for node_idx in range(self.all_nodes_num)
        }
        total_normal_tp_group_num_mapping = {
            node_idx: len(normal_tp_groups_mapping[node_idx])
                for node_idx in range(self.all_nodes_num)
        }
        for node_idx in range(self.all_nodes_num):
            normal_tp_groups_mapping[node_idx].sort(
                key=lambda x: max(x.devices)
            )
        
        # step 1
        # 先把straggler tp group的位置枚举出来
        # 其余都用None代替
        all_pipelines_template = []
        pipelines = []
        pipelines_straggler_tp_group_num = []
        total_straggler_tp_group_num = len(straggler_tp_groups)
        total_normal_tp_group_num = sum(total_normal_tp_group_num_mapping.values())
        visited_straggler_tp_group = {}
        visited_normal_tp_group_num = 0
        # 启发式搜素
        # 所有pipeline的straggler数目从大到小排
        # 每个pipeline内部straggler tp group都放在前头的stage并从大到小排
        # 且要对pipeline permuation不变的进行剪枝
        # 否则搜索空间会溢出
        def dfs(pipeline_idx):
            nonlocal all_pipelines_template, pipelines
            if pipeline_idx == self.dp:
                all_pipelines_template.append(pipelines.copy())
                return 
            pipeline = []
            pipeline_straggler_tp_group_num = 0
            def pipeline_dfs(
                stage_idx, 
                min_straggler_tp_group_idx, 
            ):
                nonlocal pipeline, pipeline_straggler_tp_group_num, \
                    pipelines, pipelines_straggler_tp_group_num, \
                    visited_straggler_tp_group, visited_normal_tp_group_num
                # 减枝
                # 靠后的pipeline不能比靠前的pipeline有更多的straggler tp group
                # 如果数目相等则靠后的pipeline开头的straggler tp group的sr要小于等于前者
                if pipeline_idx != 0 and pipeline_straggler_tp_group_num > pipelines_straggler_tp_group_num[pipeline_idx - 1]:
                    return
                if pipeline_idx != 0 and pipeline_straggler_tp_group_num >= 1 and pipeline_straggler_tp_group_num == pipelines_straggler_tp_group_num[pipeline_idx - 1]:
                    if pipeline[0].sr > pipelines[pipeline_idx - 1][0].sr:
                        return
                # 搜完一条pipeline接着搜下一条
                if stage_idx == self.pp:
                    pipelines.append(pipeline.copy())
                    pipelines_straggler_tp_group_num.append(pipeline_straggler_tp_group_num)
                    # print(f"dfs pipeline {pipeline_idx} finished, visited_normal_tp_group_num = {visited_normal_tp_group_num}")
                    dfs(pipeline_idx + 1)  
                    pipelines.pop()
                    pipelines_straggler_tp_group_num.pop()
                    return 
                # 从大到小选straggler tp group
                for straggler_tp_group_idx in range(min_straggler_tp_group_idx, total_straggler_tp_group_num):
                    if straggler_tp_group_idx in visited_straggler_tp_group:
                        continue
                    visited_straggler_tp_group[straggler_tp_group_idx] = True
                    pipeline_straggler_tp_group_num += 1
                    pipeline.append(straggler_tp_groups[straggler_tp_group_idx])
                    # pipeline的下一个stage的straggler ratio需要更小
                    pipeline_dfs(stage_idx + 1, straggler_tp_group_idx + 1)
                    pipeline.pop()
                    pipeline_straggler_tp_group_num -= 1
                    del visited_straggler_tp_group[straggler_tp_group_idx]
                # 不选straggler tp group
                if visited_normal_tp_group_num >= total_normal_tp_group_num:
                    return
                pipeline.append(None)
                visited_normal_tp_group_num += 1
                # 之后必须都选None
                pipeline_dfs(stage_idx + 1, total_straggler_tp_group_num)
                visited_normal_tp_group_num -= 1
                pipeline.pop()
            # 从stage 0开始搜该条pipeline
            pipeline_dfs(0, 0) 
        # 从pipeline 0开始搜
        dfs(0)    
        
        # step 2
        # 先筛选一波
        # 对于每一种pipeline template
        # 计算理想最优解
        assert len(all_pipelines_template) > 0, "something wrong, no available strategy"
        objective_list = []
        l_values_list = []
        m_values_list = []
        for pipelines_template in all_pipelines_template:
            print("pipelines_template =", pipelines_template)
            y_values = [
                [(tp_group.sr if tp_group != None else 1.0) for tp_group in pipeline]
                    for pipeline in pipelines_template
            ]
            hetero_values = [
                [(tp_group.hetero_ratio if tp_group != None else 1.0) for tp_group in pipeline]
                    for pipeline in pipelines_template
            ]
            objective, l_values, m_values = self.solve_pp_arrangement(y_values, hetero_values)
            objective_list.append(objective)
            l_values_list.append(l_values)
            m_values_list.append(m_values)
        top_k = min(self.ctxs.top_k, len(objective_list))
        top_k_idx = heapq.nsmallest(top_k, range(len(objective_list)), key=lambda i: objective_list[i])
        top_k_l_values = [l_values_list[i] for i in top_k_idx]
        top_k_m_values = [m_values_list[i] for i in top_k_idx]
        top_k_pipelines_template = [all_pipelines_template[i] for i in top_k_idx]
        
        # step 3
        # 对于top k的pipeline template再依次分配每个normal tp group
        # 采用启发式策略
        # 从stage靠前的地方往后枚举
        # 尽量让跨node的grad reduce尽量少
        # 当不得不存在跨node通信时
        # 尽量让node选取和最慢的straggler一致
        for pipelines_template in top_k_pipelines_template:
            visited_normal_tp_group_num_mapping = {
                node_idx: 0
                    for node_idx in range(self.all_nodes_num)
            }
            # step 3.1
            # 第一遍pass
            # 先尽量减少跨机grad reduce
            # 把能填的None填了
            for stage_id in range(self.pp):
                if pipelines_template[0][stage_id] == None:
                    continue
                suggested_node_idx = pipelines_template[0][stage_id].node_idx
                for pipeline_id in range(self.dp):
                    if pipelines_template[pipeline_id][stage_id] == None:
                        curr_num = visited_normal_tp_group_num_mapping[suggested_node_idx]
                        if curr_num < total_normal_tp_group_num_mapping[suggested_node_idx]:
                            pipelines_template[pipeline_id][stage_id] = normal_tp_groups_mapping[suggested_node_idx][curr_num]
                            visited_normal_tp_group_num_mapping[suggested_node_idx] += 1
            # step 3.2
            # 第二遍pass
            # 把剩下的None按顺序填了
            for stage_id in range(self.pp):
                 for pipeline_id in range(self.dp):
                    if pipelines_template[pipeline_id][stage_id] == None:
                        replace_none = False
                        for node_idx in range(self.all_nodes_num):
                            curr_num = visited_normal_tp_group_num_mapping[node_idx]
                            if curr_num < total_normal_tp_group_num_mapping[node_idx]:
                                pipelines_template[pipeline_id][stage_id] = normal_tp_groups_mapping[node_idx][curr_num]
                                visited_normal_tp_group_num_mapping[node_idx] += 1
                                replace_none = True
                                break
                        assert replace_none == True, "can't find a normal tp group to place here"

        # 最终返回top k个搜索出的策略
        return top_k_pipelines_template, top_k_l_values, top_k_m_values
                                                            
    def solve_pp_arrangement(
        self, 
        y_values: List[List[float]], 
        hetero_values: List[List[float]]
    ):    
        # 第一个整数线性规划问题
        dp = self.dp
        pp = self.pp
        L = self.pp * self.ctxs.normal_layers
        C = self.ctxs.memory_bound - self.ctxs.memory_safe_gap
        k_values = self.ctxs.memory_k
        d_values = self.ctxs.memory_d
        # 最优解的值
        o_values = [] 
        l_values = []
        # 第二个整数线性规划问题
        B_div_b = self.dp * self.ctxs.normal_mbn
        # 最优解的值
        m_values = [] 
        
        pulp.LpSolverDefault.msg = 0
        # 求解第一个整数线性规划问题
        for i in range(dp):
            # 定义整数线性规划问题
            model = LpProblem(name="Integer_Linear_Programming_Problem_1", sense=LpMinimize)
            # 变量
            l_vars = [LpVariable(f"l_{i}_{j}", lowBound=0, cat="Integer") for j in range(pp)]
            max_y_times_l = LpVariable("max_y_times_l", lowBound=0, cat="Continuous")  # 最大值变量
            # 添加目标函数
            model += max_y_times_l
            # 约束条件
            model += lpSum(l_vars) == L
            for j in range(pp):
                model += max_y_times_l >= y_values[i][j] * l_vars[j]
                model += k_values[j] * l_vars[j] + d_values[j] <= C / hetero_values[i][j]
            # 求解问题
            model.solve()
            # 打印结果
            # print(f"Q1, num = {i}, Optimal Objective Value:", model.objective.value())
            o_values.append(model.objective.value())
            l_i_values = []
            for var in model.variables():
                # print(var.name, "=", var.varValue)
                if 'l_' in var.name:
                    l_i_values.append(int(var.varValue))
            l_values.append(l_i_values)
            
        # 求解第二个整数线性规划问题
        # 定义整数线性规划问题
        model = LpProblem(name="Integer_Linear_Programming_Problem_2", sense=LpMinimize)
        # 变量
        m_vars = [LpVariable(f"m_{i}", lowBound=pp, cat="Integer") for i in range(dp)]
        max_o_times_m = LpVariable("max_o_times_m", lowBound=0, cat="Continuous")  # 最大值变量
        # 添加目标函数
        model += max_o_times_m
        # 约束条件
        model += lpSum(m_vars) == B_div_b
        for i in range(dp):
            model += max_o_times_m >= o_values[i] * m_vars[i]
            model += m_vars[i] >= pp
        # 求解问题
        model.solve()
        # 打印结果
        # print("Q2, Optimal Objective Value:", model.objective.value())
        for var in model.variables():
            # print(var.name, "=", var.varValue)
            if 'm_' in var.name:
                m_values.append(int(var.varValue))

        print("**********************************************") 
        print("Result:")
        print("objective =", model.objective.value()) 
        print("l_values =", l_values)  
        print("m_values =", m_values)  
        return model.objective.value(), l_values, m_values
                
                        
                    
                
        